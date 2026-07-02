from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import Actor, LinearFeedbackActor, ValueFunction, ZeroValue
from .rollout import actor_rollout, max_dynamics_gap
from .sb3_wrappers import NumpySACActor, NumpySACQCost, SB3SACActor, SB3SACQCost
from .types import Trajectory


_DLL_DIRECTORY_HANDLES: list[Any] = []


@dataclass
class AcadosSnowHillSolverConfig:
    horizon: int = 20
    terminal_rollout: int = 0
    gamma: float = 1.0
    value_kind: str = "q"
    critic_scale: float = 1.0
    terminal_hessian_epsilon: float = 1e-6
    nlp_solver_type: str = "SQP_RTI"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    qp_solver_cond_N: int = 5
    split_rti: bool = True
    print_level: int = 0
    ext_fun_compile_flags: str = "-O0"
    code_export_directory: str | None = None
    acados_dir: str | None = None
    mingw_bin: str | None = None
    name: str = "snow_hill_ac4mpc"
    verbose: bool = False
    generate: bool = True
    build: bool = True


class AcadosSnowHillSolver:
    """acados SQP/RTI multiple-shooting solver for the paper snow-hill OCP."""

    def __init__(
        self,
        system,
        actor: Actor,
        value_function: ValueFunction,
        config: AcadosSnowHillSolverConfig,
    ) -> None:
        if abs(config.gamma - 1.0) > 1e-12:
            raise ValueError("The paper snow-hill experiments use gamma=1; acados backend enforces this.")
        if config.value_kind not in {"value", "q"}:
            raise ValueError("value_kind must be 'value' or 'q'")
        self.system = system
        self.actor = actor
        self.value_function = value_function
        self.config = config
        self._configure_runtime_paths()
        self._build_solver()

    @property
    def horizon(self) -> int:
        return self.config.horizon

    def solve(
        self,
        state: np.ndarray,
        initial: Trajectory | None = None,
        *,
        max_iterations: int | None = None,
        source: str = "acados",
    ) -> Trajectory:
        state = np.asarray(state, dtype=float).reshape(self.system.state_dim)
        if initial is None:
            initial = actor_rollout(self.system, state, self.actor, self.horizon, source=source)
        if initial.horizon != self.horizon:
            raise ValueError(f"initial horizon {initial.horizon} does not match solver horizon {self.horizon}")

        iterations = 1 if max_iterations is None else int(max_iterations)
        if iterations <= 0:
            out = initial.copy(source=source)
            out.feasible_gap = max_dynamics_gap(self.system, out)
            out.info["solver_status"] = "not_run"
            return out

        self._set_initial_guess(state, initial)
        status = 0
        for _ in range(iterations):
            if self.config.nlp_solver_type == "SQP_RTI" and self.config.split_rti:
                self.solver.options_set("rti_phase", 1)
                status = int(self.solver.solve())
                self.solver.options_set("rti_phase", 2)
                status = int(self.solver.solve())
            else:
                status = int(self.solver.solve())

        trajectory = self._read_trajectory(source=source)
        trajectory.objective = self._get_cost_or_none()
        trajectory.feasible_gap = max_dynamics_gap(self.system, trajectory)
        trajectory.info.update(self._solver_info(status))
        return trajectory

    def _configure_runtime_paths(self) -> None:
        root = Path(__file__).resolve().parents[2]
        if self.config.acados_dir:
            acados_dir = Path(self.config.acados_dir)
        elif os.environ.get("ACADOS_SOURCE_DIR"):
            acados_dir = Path(os.environ["ACADOS_SOURCE_DIR"])
        else:
            acados_dir = root / "acados"
        os.environ["ACADOS_SOURCE_DIR"] = str(acados_dir)

        path_entries = [acados_dir / "bin", acados_dir / "lib"]
        mingw_env = os.environ.get("MINGW_BIN")
        mingw_bin = Path(self.config.mingw_bin) if self.config.mingw_bin else None
        if mingw_bin is None and mingw_env:
            mingw_bin = Path(mingw_env)
        if mingw_bin is not None and mingw_bin.exists():
            path_entries.append(mingw_bin)
            os.environ.setdefault("CC", str(mingw_bin / "gcc.exe"))
            os.environ.setdefault("CXX", str(mingw_bin / "g++.exe"))

        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(str(path) for path in path_entries if path.exists()) + os.pathsep + existing_path

        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for path in path_entries:
                if path.exists():
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))

    def _build_solver(self) -> None:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        nx, nu = self.system.state_dim, self.system.control_dim
        model = AcadosModel()
        x = ca.SX.sym("x", nx)
        u = ca.SX.sym("u", nu)
        model.x = x
        model.u = u
        model.disc_dyn_expr = self._rk4_expr(x, u)
        model.name = self._sanitize_name(self.config.name)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.config.horizon
        ocp.solver_options.tf = self.config.horizon * float(self.system.config.dt)
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = self.config.nlp_solver_type
        ocp.solver_options.qp_solver = self.config.qp_solver
        if self.config.qp_solver == "PARTIAL_CONDENSING_HPIPM":
            ocp.solver_options.qp_solver_cond_N = min(self.config.qp_solver_cond_N, self.config.horizon)
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.print_level = self.config.print_level
        ocp.solver_options.ext_fun_compile_flags = self.config.ext_fun_compile_flags

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = self._stage_cost_residual_expr(x, u)
        ocp.cost.W = np.eye(2)
        ocp.cost.yref = np.zeros(2)

        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = self._terminal_cost_expr(x)
        ocp.model.cost_expr_ext_cost_custom_hess_e = self.config.terminal_hessian_epsilon * ca.DM.eye(nx)

        limit = float(self.system.config.control_limit)
        ocp.constraints.x0 = np.zeros(nx)
        ocp.remove_x0_elimination()
        ocp.constraints.lbu = -limit * np.ones(nu)
        ocp.constraints.ubu = limit * np.ones(nu)
        ocp.constraints.idxbu = np.arange(nu)

        if self.config.code_export_directory is None:
            code_dir = Path(__file__).resolve().parents[1] / "generated" / "acados" / model.name
        else:
            code_dir = Path(self.config.code_export_directory)
        code_dir.mkdir(parents=True, exist_ok=True)
        ocp.code_gen_opts.code_export_directory = str(code_dir.resolve())

        self.solver = AcadosOcpSolver(
            ocp,
            json_file=str((code_dir / f"{model.name}.json").resolve()),
            verbose=self.config.verbose,
            generate=self.config.generate,
            build=self.config.build,
        )

    def _set_initial_guess(self, state: np.ndarray, initial: Trajectory) -> None:
        self.solver.constraints_set(0, "lbx", state)
        self.solver.constraints_set(0, "ubx", state)
        for k in range(self.horizon + 1):
            self.solver.set(k, "x", np.asarray(initial.states[k], dtype=float))
        for k in range(self.horizon):
            self.solver.set(k, "u", np.asarray(initial.controls[k], dtype=float))

    def _read_trajectory(self, *, source: str) -> Trajectory:
        states = np.zeros((self.horizon + 1, self.system.state_dim), dtype=float)
        controls = np.zeros((self.horizon, self.system.control_dim), dtype=float)
        for k in range(self.horizon):
            states[k] = self.solver.get(k, "x")
            controls[k] = self.solver.get(k, "u")
        states[self.horizon] = self.solver.get(self.horizon, "x")
        return Trajectory(states, controls, source=source)

    def _get_cost_or_none(self) -> float | None:
        try:
            return float(self.solver.get_cost())
        except Exception:
            return None

    def _solver_info(self, status: int) -> dict[str, Any]:
        info: dict[str, Any] = {"solver_status": status}
        for name in ("time_tot", "time_lin", "time_qp", "time_reg", "sqp_iter"):
            try:
                value = self.solver.get_stats(name)
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                info[name] = value
            except Exception:
                pass
        return info

    def _continuous_dynamics_expr(self, x, u):
        import casadi as ca

        p = x[0]
        v = x[1]
        return ca.vertcat(v, u[0] + self._residual_acceleration_expr(p))

    def _rk4_expr(self, x, u):
        dt = float(self.system.config.dt)
        k1 = self._continuous_dynamics_expr(x, u)
        k2 = self._continuous_dynamics_expr(x + 0.5 * dt * k1, u)
        k3 = self._continuous_dynamics_expr(x + 0.5 * dt * k2, u)
        k4 = self._continuous_dynamics_expr(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _residual_acceleration_expr(self, p):
        h1 = self._huber_expr(p + 7.0)
        h2 = self._huber_expr(p + 3.0)
        return 1.25 * (((h1 - h2) ** 2) / 16.0 - 1.0)

    @staticmethod
    def _huber_expr(z, delta: float = 1.0):
        import casadi as ca

        abs_z = ca.fabs(z)
        return ca.if_else(abs_z <= delta, 0.5 * z * z, delta * (abs_z - 0.5 * delta))

    def _stage_cost_expr(self, x, u):
        import casadi as ca

        q = np.asarray(self.system.config.q, dtype=float)
        r = np.asarray(self.system.config.r, dtype=float)
        return ca.sqrt(q[0] * x[0] ** 2 + q[1] * x[1] ** 2 + 1.0) + r[0] * u[0] ** 2

    def _stage_cost_residual_expr(self, x, u):
        import casadi as ca

        q = np.asarray(self.system.config.q, dtype=float)
        r = np.asarray(self.system.config.r, dtype=float)
        state_quadratic = q[0] * x[0] ** 2 + q[1] * x[1] ** 2 + 1.0
        state_residual = ca.sqrt(2.0) * state_quadratic ** 0.25
        control_residual = ca.sqrt(2.0 * r[0]) * u[0]
        return ca.vertcat(state_residual, control_residual)

    def _terminal_cost_expr(self, terminal_state):
        cost = 0.0
        current = terminal_state
        for _ in range(self.config.terminal_rollout):
            action = self._actor_expr(current)
            cost = cost + self._stage_cost_expr(current, action)
            current = self._rk4_expr(current, action)
        return cost + self.config.critic_scale * self._critic_cost_expr(current)

    def _actor_expr(self, x):
        import casadi as ca

        if isinstance(self.actor, (SB3SACActor, NumpySACActor)):
            if isinstance(self.actor, NumpySACActor):
                sd = self.actor.weights
                action_high = np.asarray(self.actor.action_high, dtype=float).reshape(-1)
            else:
                sd = self._state_dict_numpy(self.actor.model.policy.actor)
                action_high = np.asarray(self.actor.model.action_space.high, dtype=float).reshape(-1)
            h = self._linear_expr(sd["latent_pi.0.weight"], sd["latent_pi.0.bias"], x)
            h = ca.tanh(h)
            h = self._linear_expr(sd["latent_pi.2.weight"], sd["latent_pi.2.bias"], h)
            h = ca.tanh(h)
            mu = self._linear_expr(sd["mu.weight"], sd["mu.bias"], h)
            return ca.DM(action_high) * ca.tanh(mu)
        if isinstance(self.actor, LinearFeedbackActor):
            gain = np.asarray(self.actor.gain, dtype=float).reshape(1, -1)
            raw = self._linear_expr(gain, np.zeros(1), x)
            limit = float(self.actor.control_limit)
            return ca.fmin(ca.fmax(raw, -limit), limit)
        raise TypeError(
            "acados backend needs a symbolic actor; SB3SACActor, NumpySACActor and LinearFeedbackActor are supported."
        )

    def _critic_cost_expr(self, x):
        import casadi as ca

        if isinstance(self.value_function, ZeroValue):
            return ca.SX(0.0)
        if self.config.value_kind != "q":
            raise TypeError("Only ZeroValue is supported for symbolic value critics; use SAC Q critics for value_kind='q'.")
        if not isinstance(self.value_function, (SB3SACQCost, NumpySACQCost)):
            raise TypeError("acados SAC terminal cost requires SB3SACQCost or NumpySACQCost.")

        action = self._actor_expr(x)
        critic_input = ca.vertcat(x, action)
        sd = (
            self.value_function.weights
            if isinstance(self.value_function, NumpySACQCost)
            else self._state_dict_numpy(self.value_function.model.policy.critic)
        )
        q0 = self._q_network_expr(sd, "qf0", critic_input)
        q1 = self._q_network_expr(sd, "qf1", critic_input)
        q_reward = ca.fmin(q0, q1)
        return -q_reward if self.value_function.reward_to_cost else q_reward

    def _q_network_expr(self, state_dict: dict[str, np.ndarray], prefix: str, x):
        import casadi as ca

        h = self._linear_expr(state_dict[f"{prefix}.0.weight"], state_dict[f"{prefix}.0.bias"], x)
        h = ca.tanh(h)
        h = self._linear_expr(state_dict[f"{prefix}.2.weight"], state_dict[f"{prefix}.2.bias"], h)
        h = ca.tanh(h)
        return self._linear_expr(state_dict[f"{prefix}.4.weight"], state_dict[f"{prefix}.4.bias"], h)[0]

    @staticmethod
    def _linear_expr(weight: np.ndarray, bias: np.ndarray, x):
        import casadi as ca

        return ca.mtimes(ca.DM(weight), x) + ca.DM(bias)

    @staticmethod
    def _state_dict_numpy(module) -> dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy().astype(float) for key, value in module.state_dict().items()}

    @staticmethod
    def _sanitize_name(name: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
        if not sanitized or sanitized[0].isdigit():
            sanitized = f"m_{sanitized}"
        return sanitized
