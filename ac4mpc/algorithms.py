from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .evaluation import (
    Ac4EvalResult,
    PMPResidualCertificate,
    ac4eval,
    build_pmp_taylor_certificate,
    finite_horizon_cost,
)
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

    # Optional PMP diagnostics.  Both options are disabled by default, so the
    # historical AC4MPC-RTI path and candidate ranking remain unchanged.
    enable_pmp_diagnostics: bool = False
    enable_pmp_screening: bool = False
    pmp_derivative_eps: float = 1e-5

    # Screening is fail-closed.  These values must be justified for the full
    # ac4eval objective over the validated segment, not guessed from samples.
    certificate_lipschitz: float | None = None
    certificate_gradient_error: float | None = None
    certificate_radius: float | None = None
    certificate_tolerance: float = 0.0
    certificate_segment_validator: Callable[[Trajectory, Trajectory, Ac4EvalResult], bool] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.pmp_derivative_eps) or self.pmp_derivative_eps <= 0.0:
            raise ValueError("pmp_derivative_eps must be finite and positive")
        if not np.isfinite(self.certificate_tolerance) or self.certificate_tolerance < 0.0:
            raise ValueError("certificate_tolerance must be finite and non-negative")
        if self.enable_pmp_screening:
            missing = []
            if self.certificate_lipschitz is None:
                missing.append("certificate_lipschitz")
            if self.certificate_gradient_error is None:
                missing.append("certificate_gradient_error")
            if self.certificate_radius is None:
                missing.append("certificate_radius")
            if self.certificate_segment_validator is None:
                missing.append("certificate_segment_validator")
            if missing:
                raise ValueError(
                    "PMP screening requires validated values for: " + ", ".join(missing)
                )
            if self.certificate_lipschitz < 0.0:
                raise ValueError("certificate_lipschitz must be non-negative")
            if self.certificate_gradient_error < 0.0:
                raise ValueError("certificate_gradient_error must be non-negative")
            if self.certificate_radius <= 0.0:
                raise ValueError("certificate_radius must be positive")


class AC4MPCRTIController:
    """Algorithm 2 AC4MPC-RTI controller with optional PMP candidate screening."""

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

    def _evaluate(self, trajectory: Trajectory, *, return_info: bool, compute_adjoint: bool):
        result = ac4eval(
            self.system,
            trajectory,
            self.actor,
            self.value_function,
            alpha=self.config.alpha,
            rollout_length=self.config.evaluation_rollout,
            gamma=self.config.gamma,
            value_kind=self.config.value_kind,
            critic_scale=self.config.critic_scale,
            return_info=return_info,
            compute_adjoint=compute_adjoint,
            derivative_eps=self.config.pmp_derivative_eps,
        )
        if isinstance(result, Ac4EvalResult):
            return result.cost, result
        return float(result), None

    def step(self, measured_state: np.ndarray) -> ControllerStep:
        step_start = perf_counter()
        timings: dict[str, float] = {}
        state = np.asarray(measured_state, dtype=float).reshape(self.system.state_dim)

        t0 = perf_counter()
        policy = actor_rollout(self.system, state, self.actor, self.config.horizon, source="actor")
        timings["policy_rollout_seconds"] = perf_counter() - t0

        t0 = perf_counter()
        if self.parallel is None or self.iteration % self.config.reinitialization_period == 0:
            self.parallel = policy.copy(source="parallel_init_actor")
        else:
            self.parallel = self._replace_initial_state(self.parallel, state)

        if self.active is None:
            self.active = policy.copy(source="active_init_actor")
        else:
            self.active = self._replace_initial_state(self.active, state)
        timings["warm_start_seconds"] = perf_counter() - t0

        t0 = perf_counter()
        active_solution = self.active_solver.solve(
            state,
            self.active,
            max_iterations=self.config.active_sqp_iterations,
            source="active",
        )
        timings["active_solve_seconds"] = perf_counter() - t0

        t0 = perf_counter()
        parallel_solution = self.parallel_solver.solve(
            state,
            self.parallel,
            max_iterations=self.config.parallel_sqp_iterations,
            source="parallel",
        )
        timings["parallel_solve_seconds"] = perf_counter() - t0

        candidates = {"active": active_solution, "parallel": parallel_solution}
        if self.config.evaluate_policy_rollout:
            candidates["actor"] = policy

        costs: dict[str, float] = {}
        evaluation_details: dict[str, Ac4EvalResult] = {}
        certificates: dict[str, PMPResidualCertificate] = {}
        candidate_status: dict[str, str] = {}
        candidate_timings: dict[str, float] = {}
        certificate_seconds = 0.0
        evaluation_seconds = 0.0

        pmp_enabled = self.config.enable_pmp_diagnostics or self.config.enable_pmp_screening
        reference_name = "active" if "active" in candidates else next(iter(candidates))

        # Evaluate one reference candidate exactly and compute its adjoint once.
        t0 = perf_counter()
        reference_cost, reference_result = self._evaluate(
            candidates[reference_name],
            return_info=pmp_enabled,
            compute_adjoint=pmp_enabled,
        )
        elapsed = perf_counter() - t0
        evaluation_seconds += elapsed
        candidate_timings[reference_name] = elapsed
        costs[reference_name] = reference_cost
        candidate_status[reference_name] = "evaluated_reference"
        if reference_result is not None:
            evaluation_details[reference_name] = reference_result

        best_cost = reference_cost
        best_name = reference_name

        for name, trajectory in candidates.items():
            if name == reference_name:
                continue

            if self.config.enable_pmp_screening:
                cert_start = perf_counter()
                certificate = build_pmp_taylor_certificate(
                    candidates[reference_name],
                    trajectory,
                    reference_result,
                    lipschitz_constant=float(self.config.certificate_lipschitz),
                    gradient_error_bound=float(self.config.certificate_gradient_error),
                    radius=float(self.config.certificate_radius),
                    segment_validator=self.config.certificate_segment_validator,
                )
                certificate_seconds += perf_counter() - cert_start
                certificates[name] = certificate
                if (
                    certificate.certified
                    and certificate.lower_bound is not None
                    and certificate.lower_bound >= best_cost + self.config.certificate_tolerance
                ):
                    candidate_status[name] = "screened_by_pmp_lower_bound"
                    continue

            t0 = perf_counter()
            cost, detail = self._evaluate(
                trajectory,
                return_info=self.config.enable_pmp_diagnostics,
                compute_adjoint=self.config.enable_pmp_diagnostics,
            )
            elapsed = perf_counter() - t0
            evaluation_seconds += elapsed
            candidate_timings[name] = elapsed
            costs[name] = cost
            candidate_status[name] = "evaluated"
            if detail is not None:
                evaluation_details[name] = detail
            if cost < best_cost:
                best_cost = cost
                best_name = name

        chosen_name = min(costs, key=costs.get)
        # ``costs`` contains exact evaluations only.  A screened candidate is
        # omitted from this mapping and described by candidate_status and its
        # certificate in ControllerStep.info; no lower bound is mislabeled as
        # an exact objective value.
        chosen = candidates[chosen_name].copy(source=chosen_name)
        chosen.objective = costs[chosen_name]

        self.active = shift_trajectory(self.system, chosen, self.actor, source="active_shifted")
        self.parallel = shift_trajectory(self.system, parallel_solution, self.actor, source="parallel_shifted")
        self.iteration += 1

        timings["candidate_evaluation_seconds"] = evaluation_seconds
        timings["certificate_seconds"] = certificate_seconds
        timings["controller_total_seconds"] = perf_counter() - step_start
        info = {
            "iteration": self.iteration,
            "timings": timings,
            "candidate_timings": candidate_timings,
            "candidate_status": candidate_status,
            "screened_candidates": [
                name for name, status in candidate_status.items()
                if status == "screened_by_pmp_lower_bound"
            ],
            "certificates": certificates,
            "pmp_evaluation_details": evaluation_details,
            "pmp_reference_candidate": reference_name if pmp_enabled else None,
            "best_exact_candidate": best_name,
            "exact_costs_only": bool(self.config.enable_pmp_screening),
        }
        return ControllerStep(
            action=chosen.controls[0].copy(),
            chosen=chosen,
            candidates=candidates,
            costs=costs,
            source=chosen_name,
            info=info,
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
