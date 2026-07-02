from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluation import ac4eval, finite_horizon_cost
from .models import Actor, ValueFunction
from .rollout import actor_rollout, shift_trajectory
from .solvers import ScipyMultipleShootingSolver
from .types import ControllerStep, Trajectory


@dataclass
class AC4MPCConfig:
    horizon: int = 20
    terminal_rollout: int = 0
    gamma: float = 1.0
    value_kind: str = "value"
    critic_scale: float = 1.0
    solver_iterations: int | None = None


class AC4MPCController:
    """Algorithm 1 controller with actor roll-out warm start and critic terminal cost."""

    def __init__(
        self,
        system,
        actor: Actor,
        value_function: ValueFunction,
        solver: ScipyMultipleShootingSolver,
        config: AC4MPCConfig | None = None,
    ) -> None:
        self.system = system
        self.actor = actor
        self.value_function = value_function
        self.solver = solver
        self.config = config or AC4MPCConfig(horizon=solver.horizon)
        self._warm: Trajectory | None = None
        self.iteration = 0

    def reset(self) -> None:
        self._warm = None
        self.iteration = 0

    def step(self, measured_state: np.ndarray) -> ControllerStep:
        state = np.asarray(measured_state, dtype=float).reshape(self.system.state_dim)
        policy = actor_rollout(self.system, state, self.actor, self.config.horizon, source="actor")
        warm = policy if self._warm is None else self._replace_initial_state(self._warm, state)
        solved = self.solver.solve(
            state,
            warm,
            max_iterations=self.config.solver_iterations,
            source="mpc",
        )

        candidates = {"warm": warm, "mpc": solved, "actor": policy}
        costs = {
            name: finite_horizon_cost(
                self.system,
                traj,
                self.actor,
                self.value_function,
                terminal_rollout=self.config.terminal_rollout,
                gamma=self.config.gamma,
                value_kind=self.config.value_kind,
                critic_scale=self.config.critic_scale,
            )
            for name, traj in candidates.items()
        }
        chosen_name = min(costs, key=costs.get)
        chosen = candidates[chosen_name].copy(source=chosen_name)
        chosen.objective = costs[chosen_name]
        self._warm = shift_trajectory(self.system, chosen, self.actor, source="shifted")
        self.iteration += 1
        return ControllerStep(
            action=chosen.controls[0].copy(),
            chosen=chosen,
            candidates=candidates,
            costs=costs,
            source=chosen_name,
            info={"iteration": self.iteration},
        )

    def _replace_initial_state(self, trajectory: Trajectory, state: np.ndarray) -> Trajectory:
        """Keep shifted controls but resimulate a feasible warm trajectory from measurement."""

        states = self.system.simulate_controls(state, trajectory.controls)
        return Trajectory(states, trajectory.controls.copy(), source=trajectory.source)


@dataclass
class AC4MPCRTIConfig:
    horizon: int = 20
    active_sqp_iterations: int = 1
    parallel_sqp_iterations: int = 1
    reinitialization_period: int = 5
    alpha: float = 1.0
    evaluation_rollout: int = 20
    gamma: float = 1.0
    value_kind: str = "value"
    critic_scale: float = 1.0
    evaluate_policy_rollout: bool = True


class AC4MPCRTIController:
    """Algorithm 2 AC4MPC-RTI controller with active and parallel MPC instances."""

    def __init__(
        self,
        system,
        actor: Actor,
        value_function: ValueFunction,
        active_solver: ScipyMultipleShootingSolver,
        parallel_solver: ScipyMultipleShootingSolver,
        config: AC4MPCRTIConfig | None = None,
    ) -> None:
        self.system = system
        self.actor = actor
        self.value_function = value_function
        self.active_solver = active_solver
        self.parallel_solver = parallel_solver
        self.config = config or AC4MPCRTIConfig(horizon=active_solver.horizon)
        self.active: Trajectory | None = None
        self.parallel: Trajectory | None = None
        self.iteration = 0

    def reset(self) -> None:
        self.active = None
        self.parallel = None
        self.iteration = 0

    def step(self, measured_state: np.ndarray) -> ControllerStep:
        state = np.asarray(measured_state, dtype=float).reshape(self.system.state_dim)
        policy = actor_rollout(self.system, state, self.actor, self.config.horizon, source="actor")

        if self.parallel is None or self.iteration % self.config.reinitialization_period == 0:
            self.parallel = policy.copy(source="parallel_init_actor")
        else:
            self.parallel = self._replace_initial_state(self.parallel, state)

        if self.active is None:
            self.active = policy.copy(source="active_init_actor")
        else:
            self.active = self._replace_initial_state(self.active, state)

        active_solution = self.active_solver.solve(
            state,
            self.active,
            max_iterations=self.config.active_sqp_iterations,
            source="active",
        )
        parallel_solution = self.parallel_solver.solve(
            state,
            self.parallel,
            max_iterations=self.config.parallel_sqp_iterations,
            source="parallel",
        )

        candidates = {"active": active_solution, "parallel": parallel_solution}
        if self.config.evaluate_policy_rollout:
            candidates["actor"] = policy

        costs = {
            name: ac4eval(
                self.system,
                traj,
                self.actor,
                self.value_function,
                alpha=self.config.alpha,
                rollout_length=self.config.evaluation_rollout,
                gamma=self.config.gamma,
                value_kind=self.config.value_kind,
                critic_scale=self.config.critic_scale,
            )
            for name, traj in candidates.items()
        }
        chosen_name = min(costs, key=costs.get)
        chosen = candidates[chosen_name].copy(source=chosen_name)
        chosen.objective = costs[chosen_name]

        self.active = shift_trajectory(self.system, chosen, self.actor, source="active_shifted")
        self.parallel = shift_trajectory(self.system, parallel_solution, self.actor, source="parallel_shifted")
        self.iteration += 1
        return ControllerStep(
            action=chosen.controls[0].copy(),
            chosen=chosen,
            candidates=candidates,
            costs=costs,
            source=chosen_name,
            info={"iteration": self.iteration},
        )

    def _replace_initial_state(self, trajectory: Trajectory, state: np.ndarray) -> Trajectory:
        states = self.system.simulate_controls(state, trajectory.controls)
        return Trajectory(states, trajectory.controls.copy(), source=trajectory.source)


def simulate_closed_loop(system, controller, initial_state: np.ndarray, steps: int) -> dict[str, np.ndarray | list]:
    states = np.zeros((steps + 1, system.state_dim), dtype=float)
    controls = np.zeros((steps, system.control_dim), dtype=float)
    costs = np.zeros(steps, dtype=float)
    sources: list[str] = []
    states[0] = np.asarray(initial_state, dtype=float).reshape(system.state_dim)
    for j in range(steps):
        result = controller.step(states[j])
        action = system.clip_control(result.action)
        controls[j] = action
        costs[j] = system.stage_cost(states[j], action)
        states[j + 1] = system.step(states[j], action)
        sources.append(result.source)
    return {"states": states, "controls": controls, "costs": costs, "sources": sources}

