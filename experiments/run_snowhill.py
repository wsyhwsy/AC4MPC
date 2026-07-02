from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ac4mpc.algorithms import (
    AC4MPCConfig,
    AC4MPCController,
    AC4MPCRTIConfig,
    AC4MPCRTIController,
    simulate_closed_loop,
)
from ac4mpc.models import LinearFeedbackActor, ZeroValue
from ac4mpc.sb3_wrappers import NumpySACActor, NumpySACQCost, SB3SACActor, SB3SACQCost
from ac4mpc.snow_hill import SnowHillConfig, SnowHillSystem
from ac4mpc.acados_solvers import AcadosSnowHillSolver, AcadosSnowHillSolverConfig
from ac4mpc.solvers import MultipleShootingSolverConfig, ScipyMultipleShootingSolver


def ensure_sac_weights(model_path: Path, explicit_weights_path: Path | None) -> Path:
    if explicit_weights_path is not None:
        return explicit_weights_path
    output_dir = ROOT / "runs" / "exported_sac_weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_path.stem.replace(".", "_")
    output_path = output_dir / f"{safe_name}.npz"
    if not output_path.exists() or output_path.stat().st_mtime < model_path.stat().st_mtime:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("export_sac_weights.py")),
                "--model",
                str(model_path),
                "--out",
                str(output_path),
            ],
            check=True,
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snow hill AC4MPC/AC4MPC-RTI.")
    parser.add_argument("--controller", choices=["actor", "ac4mpc", "rti"], default="rti")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--terminal-rollout", type=int, default=0)
    parser.add_argument("--eval-rollout", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--solver-iterations", type=int, default=1)
    parser.add_argument("--solver-backend", choices=["scipy", "acados"], default="scipy")
    parser.add_argument("--acados-dir", type=Path)
    parser.add_argument("--mingw-bin", type=Path)
    parser.add_argument("--critic-scale", type=float, default=1.0)
    parser.add_argument("--sac-model", type=Path)
    parser.add_argument("--sac-weights", type=Path)
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--initial-state", type=float, nargs=2, default=(-5.0, -1.0))
    parser.add_argument("--q", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--r", type=float, nargs=1, default=(0.01,))
    args = parser.parse_args()

    system = SnowHillSystem(SnowHillConfig(q=tuple(args.q), r=tuple(args.r)))
    if (args.sac_model is not None or args.sac_weights is not None) and args.ppo_checkpoint is not None:
        raise ValueError("Use only one of --sac-model or --ppo-checkpoint.")
    if args.ppo_checkpoint is not None:
        from ac4mpc.ppo import load_ppo_cost_actor_critic

        actor, value, _ = load_ppo_cost_actor_critic(args.ppo_checkpoint)
        value_kind = "value"
    elif args.sac_weights is not None or (args.sac_model is not None and args.solver_backend == "acados"):
        if args.sac_model is None and args.sac_weights is None:
            raise ValueError("Missing SAC model or exported SAC weights.")
        weights_path = ensure_sac_weights(args.sac_model, args.sac_weights) if args.sac_model else args.sac_weights
        actor = NumpySACActor.load(weights_path)
        value = NumpySACQCost.load(weights_path)
        value_kind = "q"
    elif args.sac_model is not None:
        actor = SB3SACActor.load(str(args.sac_model))
        value = SB3SACQCost.load(str(args.sac_model))
        value_kind = "q"
    else:
        actor = LinearFeedbackActor(control_limit=system.config.control_limit)
        value = ZeroValue()
        value_kind = "value"

    if args.solver_backend == "acados":
        acados_code_dir = ROOT / "generated" / "acados" / "snow_hill_rti"
        acados_name = "snow_hill_rti"
        active_solver = AcadosSnowHillSolver(
            system,
            actor,
            value,
            AcadosSnowHillSolverConfig(
                horizon=args.horizon,
                terminal_rollout=args.terminal_rollout,
                gamma=system.config.gamma,
                value_kind=value_kind,
                critic_scale=args.critic_scale,
                name=acados_name,
                code_export_directory=str(acados_code_dir),
                acados_dir=str(args.acados_dir) if args.acados_dir else None,
                mingw_bin=str(args.mingw_bin) if args.mingw_bin else None,
            ),
        )
        parallel_solver = AcadosSnowHillSolver(
            system,
            actor,
            value,
            AcadosSnowHillSolverConfig(
                horizon=args.horizon,
                terminal_rollout=args.terminal_rollout,
                gamma=system.config.gamma,
                value_kind=value_kind,
                critic_scale=args.critic_scale,
                name=acados_name,
                code_export_directory=str(acados_code_dir),
                acados_dir=str(args.acados_dir) if args.acados_dir else None,
                mingw_bin=str(args.mingw_bin) if args.mingw_bin else None,
                generate=False,
                build=False,
            ),
        )
    else:
        solver_config = MultipleShootingSolverConfig(
            horizon=args.horizon,
            terminal_rollout=args.terminal_rollout,
            gamma=system.config.gamma,
            value_kind=value_kind,
            critic_scale=args.critic_scale,
            max_iterations=max(args.solver_iterations, 1),
        )
        active_solver = ScipyMultipleShootingSolver(system, actor, value, solver_config)
        parallel_solver = ScipyMultipleShootingSolver(system, actor, value, solver_config)

    if args.controller == "actor":
        class ActorOnly:
            def step(self, measured_state):
                from ac4mpc.rollout import actor_rollout
                from ac4mpc.types import ControllerStep

                traj = actor_rollout(system, measured_state, actor, args.horizon, source="actor")
                return ControllerStep(traj.controls[0], traj, {"actor": traj}, {"actor": 0.0}, "actor")

        controller = ActorOnly()
    elif args.controller == "ac4mpc":
        controller = AC4MPCController(
            system,
            actor,
            value,
            active_solver,
            AC4MPCConfig(
                horizon=args.horizon,
                terminal_rollout=args.terminal_rollout,
                gamma=system.config.gamma,
                value_kind=value_kind,
                critic_scale=args.critic_scale,
                solver_iterations=args.solver_iterations,
            ),
        )
    else:
        controller = AC4MPCRTIController(
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
                value_kind=value_kind,
                critic_scale=args.critic_scale,
            ),
        )

    result = simulate_closed_loop(system, controller, args.initial_state, args.steps)
    summary = {
        "controller": args.controller,
        "solver_backend": args.solver_backend,
        "initial_state": list(args.initial_state),
        "steps": args.steps,
        "total_cost": float(result["costs"].sum()),
        "final_state": result["states"][-1].tolist(),
        "sources": result["sources"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
