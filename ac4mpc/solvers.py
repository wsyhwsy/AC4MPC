from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, minimize

from .evaluation import finite_horizon_cost
from .models import Actor, ValueFunction
from .rollout import actor_rollout, max_dynamics_gap
from .types import Trajectory


@dataclass
class MultipleShootingSolverConfig:
    horizon: int = 20
    terminal_rollout: int = 0
    gamma: float = 1.0
    value_kind: str = "value"
    critic_scale: float = 1.0
    max_iterations: int = 200
    ftol: float = 1e-8
    display: bool = False


class ScipyMultipleShootingSolver:
    """Local multiple-shooting NLP solver for the AC4MPC OCP.

    The paper used CasADi/Ipopt for converged AC4MPC and acados SQP-RTI for
    AC4MPC-RTI. This class solves the same multiple-shooting NLP with SciPy
    SLSQP so the reproduction is self-contained and accepts arbitrary Python
    actor/critic callables. Limiting `max_iterations` gives an SQP-style
    intermediate iterate for algorithmic RTI tests.
    """

    def __init__(
        self,
        system,
        actor: Actor,
        value_function: ValueFunction,
        config: MultipleShootingSolverConfig,
    ) -> None:
        self.system = system
        self.actor = actor
        self.value_function = value_function
        self.config = config

    @property
    def horizon(self) -> int:
        return self.config.horizon

    def solve(
        self,
        state: np.ndarray,
        initial: Trajectory | None = None,
        *,
        max_iterations: int | None = None,
        source: str = "mpc",
    ) -> Trajectory:
        if initial is None:
            initial = actor_rollout(self.system, state, self.actor, self.horizon)
        if initial.horizon != self.horizon:
            raise ValueError(f"initial horizon {initial.horizon} does not match solver horizon {self.horizon}")

        iterations = self.config.max_iterations if max_iterations is None else int(max_iterations)
        if iterations <= 0:
            out = initial.copy(source=source)
            out.objective = finite_horizon_cost(
                self.system,
                out,
                self.actor,
                self.value_function,
                terminal_rollout=self.config.terminal_rollout,
                gamma=self.config.gamma,
                value_kind=self.config.value_kind,
                critic_scale=self.config.critic_scale,
            )
            out.feasible_gap = max_dynamics_gap(self.system, out)
            out.info["solver_status"] = "not_run"
            return out

        z0 = self._pack(initial)
        bounds = self._bounds()

        def objective(z: np.ndarray) -> float:
            traj = self._unpack(z, source=source)
            return finite_horizon_cost(
                self.system,
                traj,
                self.actor,
                self.value_function,
                terminal_rollout=self.config.terminal_rollout,
                gamma=self.config.gamma,
                value_kind=self.config.value_kind,
                critic_scale=self.config.critic_scale,
            )

        def equality(z: np.ndarray) -> np.ndarray:
            traj = self._unpack(z, source=source)
            residuals = [traj.states[0] - np.asarray(state, dtype=float).reshape(self.system.state_dim)]
            for k in range(self.horizon):
                residuals.append(self.system.step(traj.states[k], traj.controls[k]) - traj.states[k + 1])
            return np.concatenate(residuals)

        result = minimize(
            objective,
            z0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": equality},
            options={"maxiter": iterations, "ftol": self.config.ftol, "disp": self.config.display},
        )
        trajectory = self._unpack(result.x, source=source)
        trajectory.objective = float(result.fun) if np.isfinite(result.fun) else objective(result.x)
        trajectory.feasible_gap = max_dynamics_gap(self.system, trajectory)
        trajectory.info.update(
            {
                "solver_success": bool(result.success),
                "solver_status": str(result.message),
                "solver_iterations": int(getattr(result, "nit", -1)),
            }
        )
        return trajectory

    def _pack(self, trajectory: Trajectory) -> np.ndarray:
        return np.concatenate([trajectory.states.reshape(-1), trajectory.controls.reshape(-1)])

    def _unpack(self, z: np.ndarray, *, source: str) -> Trajectory:
        nx, nu, n = self.system.state_dim, self.system.control_dim, self.horizon
        state_size = (n + 1) * nx
        states = np.asarray(z[:state_size], dtype=float).reshape(n + 1, nx)
        controls = np.asarray(z[state_size:], dtype=float).reshape(n, nu)
        return Trajectory(states, controls, source=source)

    def _bounds(self) -> Bounds:
        nx, n = self.system.state_dim, self.horizon
        state_bounds = [(-np.inf, np.inf) for _ in range((n + 1) * nx)]
        control_bounds = self.system.bounds_for_controls(n)
        lower, upper = zip(*(state_bounds + control_bounds))
        return Bounds(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

