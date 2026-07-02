from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ac4mpc.acados_solvers import AcadosSnowHillSolver, AcadosSnowHillSolverConfig
from ac4mpc.algorithms import AC4MPCRTIConfig, AC4MPCRTIController, simulate_closed_loop
from ac4mpc.rollout import actor_rollout
from ac4mpc.sb3_wrappers import NumpySACActor, NumpySACQCost, export_sb3_sac_weights
from ac4mpc.snow_hill import SnowHillConfig, SnowHillSystem, snow_hill_residual_acceleration
from ac4mpc.types import ControllerStep


class ActorOnly:
    def __init__(self, system, actor, horizon: int) -> None:
        self.system = system
        self.actor = actor
        self.horizon = horizon

    def step(self, measured_state) -> ControllerStep:
        traj = actor_rollout(self.system, measured_state, self.actor, self.horizon, source="actor")
        return ControllerStep(traj.controls[0], traj, {"actor": traj}, {"actor": 0.0}, "actor")


def parse_training_log(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    block: dict[str, float] = {}
    patterns = {
        "ep_rew_mean": re.compile(r"\|\s+ep_rew_mean\s+\|\s+([-+0-9.eE]+)"),
        "fps": re.compile(r"\|\s+fps\s+\|\s+([-+0-9.eE]+)"),
        "time_elapsed": re.compile(r"\|\s+time_elapsed\s+\|\s+([-+0-9.eE]+)"),
        "total_timesteps": re.compile(r"\|\s+total_timesteps\s+\|\s+([-+0-9.eE]+)"),
    }
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("---"):
            if "total_timesteps" in block and "ep_rew_mean" in block:
                rows.append(dict(block))
            block = {}
            continue
        for key, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                block[key] = float(match.group(1))
    if "total_timesteps" in block and "ep_rew_mean" in block:
        rows.append(dict(block))
    return rows


def resolve_sac_weights(model_path: Path | None, weights_path: Path | None) -> Path:
    if weights_path is not None:
        return weights_path
    if model_path is None:
        raise ValueError("Provide --sac-model or --weights.")
    output_dir = ROOT / "runs" / "exported_sac_weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.stem}.npz"
    if not output_path.exists() or output_path.stat().st_mtime < model_path.stat().st_mtime:
        export_sb3_sac_weights(model_path, output_path)
    return output_path


def build_rti_controller(args, system, actor, value) -> AC4MPCRTIController:
    code_dir = ROOT / "generated" / "acados" / "snow_hill_rti"
    dll_exists = (code_dir / "acados_ocp_solver_snow_hill_rti.dll").exists()
    json_exists = (code_dir / "snow_hill_rti.json").exists()
    reuse = dll_exists and json_exists and not args.force_generate

    common = dict(
        horizon=args.horizon,
        terminal_rollout=args.terminal_rollout,
        gamma=system.config.gamma,
        value_kind="q",
        critic_scale=args.critic_scale,
        name="snow_hill_rti",
        code_export_directory=str(code_dir),
        acados_dir=str(args.acados_dir) if args.acados_dir else None,
        mingw_bin=str(args.mingw_bin) if args.mingw_bin else None,
    )
    active_solver = AcadosSnowHillSolver(
        system,
        actor,
        value,
        AcadosSnowHillSolverConfig(**common, generate=not reuse, build=not reuse),
    )
    parallel_solver = AcadosSnowHillSolver(
        system,
        actor,
        value,
        AcadosSnowHillSolverConfig(**common, generate=False, build=False),
    )
    return AC4MPCRTIController(
        system,
        actor,
        value,
        active_solver,
        parallel_solver,
        AC4MPCRTIConfig(
            horizon=args.horizon,
            active_sqp_iterations=args.solver_iterations,
            parallel_sqp_iterations=args.solver_iterations,
            reinitialization_period=args.period,
            alpha=args.alpha,
            evaluation_rollout=args.eval_rollout,
            gamma=system.config.gamma,
            value_kind="q",
            critic_scale=args.critic_scale,
        ),
    )


def save_rollout_npz(out_dir: Path, name: str, rollout: dict[str, np.ndarray | list]) -> None:
    np.savez(
        out_dir / f"{name}_rollout.npz",
        states=np.asarray(rollout["states"]),
        controls=np.asarray(rollout["controls"]),
        costs=np.asarray(rollout["costs"]),
        sources=np.asarray(rollout["sources"], dtype=object),
    )


def plot_training_curve(out_dir: Path, training_rows: list[dict[str, float]]) -> None:
    if not training_rows:
        return
    steps = [row["total_timesteps"] for row in training_rows]
    rewards = [row["ep_rew_mean"] for row in training_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax.plot(steps, rewards, color="#146C94", linewidth=2.2)
    ax.scatter(steps[-1], rewards[-1], color="#C0392B", s=32, zorder=3)
    ax.set_xlabel("SAC training steps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("SAC Training Curve")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png")
    plt.close(fig)


def plot_phase(out_dir: Path, actor_rollout_data, rti_rollout_data) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=160)
    actor_states = np.asarray(actor_rollout_data["states"])
    rti_states = np.asarray(rti_rollout_data["states"])
    p_grid = np.linspace(-10.0, 2.0, 400)
    a_res = snow_hill_residual_acceleration(p_grid)
    ax.plot(actor_states[:, 0], actor_states[:, 1], color="#D35400", label="SAC actor", linewidth=2.0)
    ax.plot(rti_states[:, 0], rti_states[:, 1], color="#146C94", label="AC4MPC-RTI", linewidth=2.0)
    ax.scatter([-5.0], [-1.0], color="#222222", marker="o", label="start", zorder=4)
    ax.scatter([0.0], [0.0], color="#1E8449", marker="*", s=100, label="goal", zorder=4)
    ax.set_xlabel("position p")
    ax.set_ylabel("velocity v")
    ax.set_title("Snow Hill Phase Trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = ax.twinx()
    ax2.plot(p_grid, a_res, color="#777777", alpha=0.25, linewidth=1.0)
    ax2.set_ylabel("residual acceleration", color="#777777")
    ax2.tick_params(axis="y", colors="#777777")
    fig.tight_layout()
    fig.savefig(out_dir / "phase_trajectory.png")
    plt.close(fig)


def plot_timeseries(out_dir: Path, actor_rollout_data, rti_rollout_data, dt: float) -> None:
    actor_states = np.asarray(actor_rollout_data["states"])
    rti_states = np.asarray(rti_rollout_data["states"])
    actor_controls = np.asarray(actor_rollout_data["controls"])
    rti_controls = np.asarray(rti_rollout_data["controls"])
    actor_costs = np.asarray(actor_rollout_data["costs"])
    rti_costs = np.asarray(rti_rollout_data["costs"])
    t_x = dt * np.arange(actor_states.shape[0])
    t_u = dt * np.arange(actor_controls.shape[0])

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.8), dpi=160, sharex=False)
    for idx, label in enumerate(("position p", "velocity v")):
        axes[idx].plot(t_x, actor_states[:, idx], color="#D35400", label="SAC actor")
        axes[idx].plot(t_x, rti_states[:, idx], color="#146C94", label="AC4MPC-RTI")
        axes[idx].axhline(0.0, color="#222222", linewidth=0.8, alpha=0.5)
        axes[idx].set_ylabel(label)
        axes[idx].grid(True, alpha=0.25)
    axes[2].step(t_u, actor_controls[:, 0], where="post", color="#D35400", label="SAC actor")
    axes[2].step(t_u, rti_controls[:, 0], where="post", color="#146C94", label="AC4MPC-RTI")
    axes[2].set_ylabel("control u")
    axes[2].grid(True, alpha=0.25)
    axes[3].plot(t_u, np.cumsum(actor_costs), color="#D35400", label="SAC actor")
    axes[3].plot(t_u, np.cumsum(rti_costs), color="#146C94", label="AC4MPC-RTI")
    axes[3].set_ylabel("cumulative cost")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("Closed-Loop Time Series", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_timeseries.png")
    plt.close(fig)


def plot_sources(out_dir: Path, sources: list[str]) -> None:
    source_to_y = {"actor": 0, "parallel": 1, "active": 2}
    colors = {"actor": "#D35400", "parallel": "#8E44AD", "active": "#146C94"}
    y = np.array([source_to_y.get(source, -1) for source in sources], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 2.4), dpi=160)
    for source, value in source_to_y.items():
        idx = np.where(y == value)[0]
        ax.scatter(idx, y[idx], s=14, color=colors[source], label=source)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["actor", "parallel", "active"])
    ax.set_xlabel("closed-loop step")
    ax.set_title("AC4MPC-RTI Selected Source")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "rti_sources.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate snow hill evaluation visualizations.")
    parser.add_argument("--sac-model", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--train-log", type=Path, default=ROOT / "runs" / "sac_50k_acados" / "train_stdout.log")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "visualizations" / "snowhill_50k")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--terminal-rollout", type=int, default=0)
    parser.add_argument("--eval-rollout", type=int, default=20)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--solver-iterations", type=int, default=1)
    parser.add_argument("--critic-scale", type=float, default=1.0)
    parser.add_argument("--initial-state", type=float, nargs=2, default=(-5.0, -1.0))
    parser.add_argument("--q", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--r", type=float, nargs=1, default=(0.01,))
    parser.add_argument("--acados-dir", type=Path, default=Path(os.environ["ACADOS_SOURCE_DIR"]) if os.environ.get("ACADOS_SOURCE_DIR") else None)
    parser.add_argument("--mingw-bin", type=Path, default=Path(os.environ["MINGW_BIN"]) if os.environ.get("MINGW_BIN") else None)
    parser.add_argument("--force-generate", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    system = SnowHillSystem(SnowHillConfig(q=tuple(args.q), r=tuple(args.r)))
    weights_path = resolve_sac_weights(args.sac_model, args.weights)
    actor = NumpySACActor.load(weights_path)
    value = NumpySACQCost.load(weights_path)

    actor_controller = ActorOnly(system, actor, args.horizon)
    actor_result = simulate_closed_loop(system, actor_controller, np.array(args.initial_state), args.steps)

    rti_controller = build_rti_controller(args, system, actor, value)
    rti_result = simulate_closed_loop(system, rti_controller, np.array(args.initial_state), args.steps)

    save_rollout_npz(args.out, "actor", actor_result)
    save_rollout_npz(args.out, "ac4mpc_rti", rti_result)
    training_rows = parse_training_log(args.train_log)
    plot_training_curve(args.out, training_rows)
    plot_phase(args.out, actor_result, rti_result)
    plot_timeseries(args.out, actor_result, rti_result, system.config.dt)
    plot_sources(args.out, list(rti_result["sources"]))

    actor_cost = float(np.asarray(actor_result["costs"]).sum())
    rti_cost = float(np.asarray(rti_result["costs"]).sum())
    summary = {
        "steps": args.steps,
        "dt": system.config.dt,
        "initial_state": list(args.initial_state),
        "training": {
            "checkpoint": str(weights_path),
            "log": str(args.train_log),
            "last_logged_timestep": training_rows[-1]["total_timesteps"] if training_rows else None,
            "last_logged_mean_episode_reward": training_rows[-1]["ep_rew_mean"] if training_rows else None,
            "last_logged_elapsed_seconds": training_rows[-1].get("time_elapsed") if training_rows else None,
        },
        "actor": {
            "total_cost": actor_cost,
            "final_state": np.asarray(actor_result["states"])[-1].tolist(),
        },
        "ac4mpc_rti": {
            "total_cost": rti_cost,
            "final_state": np.asarray(rti_result["states"])[-1].tolist(),
            "source_counts": dict(Counter(rti_result["sources"])),
        },
        "improvement": {
            "absolute_cost_reduction": actor_cost - rti_cost,
            "relative_cost_reduction": (actor_cost - rti_cost) / actor_cost,
        },
        "figures": [
            "training_curve.png",
            "phase_trajectory.png",
            "closed_loop_timeseries.png",
            "rti_sources.png",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
