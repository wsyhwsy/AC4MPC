from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .models import Actor, ValueFunction
from .types import Trajectory


def _critic_value(
    value_function: ValueFunction,
    actor: Actor,
    state: np.ndarray,
    *,
    value_kind: str,
    critic_scale: float,
) -> float:
    if value_kind == "value":
        value = value_function(state, None)
    elif value_kind == "q":
        value = value_function(state, actor(state))
    else:
        raise ValueError("value_kind must be 'value' or 'q'")
    return critic_scale * float(value)


def terminal_actor_rollout_cost(
    system,
    state: np.ndarray,
    actor: Actor,
    value_function: ValueFunction,
    *,
    rollout_length: int,
    gamma: float,
    value_kind: str = "value",
    critic_scale: float = 1.0,
) -> float:
    """Terminal cost V_f from Eq. (V_f), with optional Q(s,pi(s)) critic."""

    if rollout_length < 0:
        raise ValueError("rollout_length must be non-negative")
    current = np.asarray(state, dtype=float).reshape(system.state_dim)
    cost = 0.0
    discount = 1.0
    for _ in range(rollout_length):
        action = system.clip_control(actor(current))
        cost += discount * system.stage_cost(current, action)
        current = system.step(current, action)
        discount *= gamma
    cost += discount * _critic_value(
        value_function,
        actor,
        current,
        value_kind=value_kind,
        critic_scale=critic_scale,
    )
    return float(cost)


def finite_horizon_cost(
    system,
    trajectory: Trajectory,
    actor: Actor,
    value_function: ValueFunction,
    *,
    terminal_rollout: int,
    gamma: float,
    value_kind: str = "value",
    critic_scale: float = 1.0,
) -> float:
    """MPC objective V_N with the AC4MPC terminal cost."""

    cost = 0.0
    discount = 1.0
    for k in range(trajectory.horizon):
        cost += discount * system.stage_cost(trajectory.states[k], trajectory.controls[k])
        discount *= gamma
    terminal = terminal_actor_rollout_cost(
        system,
        trajectory.states[-1],
        actor,
        value_function,
        rollout_length=terminal_rollout,
        gamma=gamma,
        value_kind=value_kind,
        critic_scale=critic_scale,
    )
    return float(cost + discount * terminal)


@dataclass
class PMPAdjointInfo:
    """Discrete adjoint and candidate-variable gradient for ``ac4eval``.

    The derivative backend is central finite differences of the local maps.
    Consequently ``gradient_vector`` is a numerical PMP gradient, not an
    automatically certified exact derivative.  A rigorous screening call must
    additionally supply a valid bound on its gradient error.
    """

    state_gradient: np.ndarray
    control_gradient: np.ndarray
    gradient_vector: np.ndarray
    adjoints: np.ndarray
    terminal_gradient: np.ndarray
    transition_state_jacobians: np.ndarray
    transition_control_jacobians: np.ndarray
    anchor_state_jacobians: np.ndarray
    candidate_control_jacobians: np.ndarray
    derivative_method: str = "central_finite_difference"
    gradient_error_bound: float | None = None


@dataclass
class Ac4EvalResult:
    """Detailed result of the AC4MPC auxiliary-law candidate evaluation."""

    cost: float
    corrected_states: np.ndarray
    corrected_controls: np.ndarray
    stage_costs: np.ndarray
    terminal_value: float
    step_logs: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    smoothness: dict[str, Any] = field(default_factory=dict)
    pmp: PMPAdjointInfo | None = None


@dataclass
class PMPResidualCertificate:
    """One-sided Taylor lower-bound certificate for a candidate trajectory."""

    certified: bool
    lower_bound: float | None
    reference_cost: float
    linear_term: float | None
    curvature_penalty: float | None
    gradient_error_penalty: float | None
    delta_norm: float | None
    lipschitz_constant: float | None
    gradient_error_bound: float | None
    reason: str


def _numeric_jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float) -> np.ndarray:
    """Central-difference Jacobian; output shape is (output_dim, input_dim)."""

    x = np.asarray(x, dtype=float).reshape(-1)
    f0 = np.asarray(fun(x), dtype=float).reshape(-1)
    jac = np.empty((f0.size, x.size), dtype=float)
    for j in range(x.size):
        h = eps * max(1.0, abs(float(x[j])))
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        fp = np.asarray(fun(xp), dtype=float).reshape(-1)
        fm = np.asarray(fun(xm), dtype=float).reshape(-1)
        jac[:, j] = (fp - fm) / (2.0 * h)
    return jac


def _numeric_gradient(fun: Callable[[np.ndarray], float], x: np.ndarray, eps: float) -> np.ndarray:
    return _numeric_jacobian(lambda z: np.array([float(fun(z))]), x, eps).reshape(-1)


def trajectory_decision_vector(trajectory: Trajectory) -> np.ndarray:
    """Variables used by ``ac4eval``: s_1..s_{N-1} and u_0..u_{N-1}.

    ``ac4eval`` fixes s_0 and does not use the candidate terminal shooting state
    s_N; including either in the certificate displacement would be incorrect.
    """

    return np.concatenate(
        [trajectory.states[1:-1].reshape(-1), trajectory.controls.reshape(-1)]
    )


def _compute_pmp_adjoint(
    system,
    trajectory: Trajectory,
    actor: Actor,
    value_function: ValueFunction,
    *,
    alpha: float,
    rollout_length: int,
    gamma: float,
    value_kind: str,
    critic_scale: float,
    corrected_states: np.ndarray,
    corrected_controls: np.ndarray,
    raw_controls: np.ndarray,
    eps: float,
    step_logs: list[dict[str, Any]],
) -> tuple[PMPAdjointInfo, dict[str, Any], dict[str, float]]:
    """Compute the discrete PMP adjoint of the exact ac4eval rollout.

    The candidate shooting states s_1,...,s_{N-1} enter the auxiliary control
    law as anchor states.  Candidate s_0 is fixed by measurement and s_N is
    unused by Algorithm 3's auxiliary-law evaluation, so their gradients are
    zero by construction.
    """

    derivative_start = perf_counter()
    n = trajectory.horizon
    r = rollout_length
    total = n + r
    nx = system.state_dim
    nu = system.control_dim

    fx = np.zeros((total, nx, nx), dtype=float)
    fa = np.zeros((total, nx, nx), dtype=float)
    fu = np.zeros((total, nx, nu), dtype=float)
    lx = np.zeros((total, nx), dtype=float)
    la = np.zeros((total, nx), dtype=float)
    lu = np.zeros((total, nu), dtype=float)
    clip_kink = False

    for k in range(total):
        t0 = perf_counter()
        x = corrected_states[k]
        u = corrected_controls[k]
        raw = raw_controls[k]

        A = _numeric_jacobian(lambda z: system.step(z, u), x, eps)
        B = _numeric_jacobian(lambda z: system.step(x, z), u, eps)
        stage_x = _numeric_gradient(lambda z: system.stage_cost(z, u), x, eps)
        stage_u = _numeric_gradient(lambda z: system.stage_cost(x, z), u, eps)

        clip_jac = _numeric_jacobian(
            lambda z: system.clip_control(z), raw, eps
        )
        pi_x = _numeric_jacobian(
            lambda z: np.asarray(actor(z), dtype=float).reshape(-1), x, eps
        )

        if k < n:
            anchor = trajectory.states[k]
            pi_anchor = _numeric_jacobian(
                lambda z: np.asarray(actor(z), dtype=float).reshape(-1), anchor, eps
            )
            control_from_state = alpha * clip_jac @ pi_x
            control_from_anchor = -alpha * clip_jac @ pi_anchor
            control_from_candidate_u = clip_jac
        else:
            control_from_state = clip_jac @ pi_x
            control_from_anchor = np.zeros((nu, nx), dtype=float)
            control_from_candidate_u = np.zeros((nu, nu), dtype=float)

        fx[k] = A + B @ control_from_state
        fa[k] = B @ control_from_anchor
        fu[k] = B @ control_from_candidate_u
        lx[k] = stage_x + control_from_state.T @ stage_u
        la[k] = control_from_anchor.T @ stage_u
        lu[k] = control_from_candidate_u.T @ stage_u

        # A central difference of a clip exactly at its kink commonly returns
        # a fractional slope.  Such a point is not certified as smooth.
        fractional = np.any((np.abs(clip_jac) > 1e-5) & (np.abs(clip_jac) < 1.0 - 1e-5))
        changed = not np.allclose(raw, u, rtol=0.0, atol=1e-9)
        clip_kink = clip_kink or bool(fractional) or bool(changed)
        if k < len(step_logs):
            step_logs[k]["derivative_seconds"] = perf_counter() - t0
            step_logs[k]["clip_kink_or_active"] = bool(fractional or changed)

    terminal_start = perf_counter()
    terminal_gradient = _numeric_gradient(
        lambda z: _critic_value(
            value_function,
            actor,
            z,
            value_kind=value_kind,
            critic_scale=critic_scale,
        ),
        corrected_states[-1],
        eps,
    )
    terminal_seconds = perf_counter() - terminal_start

    adjoint_start = perf_counter()
    discounts = np.power(float(gamma), np.arange(total + 1, dtype=float))
    p = np.zeros((total + 1, nx), dtype=float)
    p[total] = discounts[total] * terminal_gradient
    grad_states = np.zeros_like(trajectory.states)
    grad_controls = np.zeros_like(trajectory.controls)

    for k in range(total - 1, -1, -1):
        p[k] = discounts[k] * lx[k] + fx[k].T @ p[k + 1]
        if k < n:
            if 0 < k < n:
                grad_states[k] = discounts[k] * la[k] + fa[k].T @ p[k + 1]
            grad_controls[k] = discounts[k] * lu[k] + fu[k].T @ p[k + 1]

    # Match trajectory_decision_vector ordering exactly.
    gradient_vector = np.concatenate(
        [grad_states[1:-1].reshape(-1), grad_controls.reshape(-1)]
    )
    adjoint_seconds = perf_counter() - adjoint_start
    derivative_seconds = perf_counter() - derivative_start

    info = PMPAdjointInfo(
        state_gradient=grad_states,
        control_gradient=grad_controls,
        gradient_vector=gradient_vector,
        adjoints=p,
        terminal_gradient=terminal_gradient,
        transition_state_jacobians=fx,
        transition_control_jacobians=fu,
        anchor_state_jacobians=fa,
        candidate_control_jacobians=fu.copy(),
    )
    smoothness = {
        "locally_smooth_at_reference": not clip_kink,
        "clip_active_or_kink": clip_kink,
        "note": "Huber dynamics are C1,1; control clipping kinks disable the default certificate path.",
    }
    timing = {
        "derivative_seconds": derivative_seconds,
        "terminal_gradient_seconds": terminal_seconds,
        "adjoint_seconds": adjoint_seconds,
    }
    return info, smoothness, timing


def ac4eval(
    system,
    trajectory: Trajectory,
    actor: Actor,
    value_function: ValueFunction,
    *,
    alpha: float,
    rollout_length: int,
    gamma: float,
    value_kind: str = "value",
    critic_scale: float = 1.0,
    return_info: bool = False,
    compute_adjoint: bool = False,
    derivative_eps: float = 1e-5,
) -> float | Ac4EvalResult:
    """Evaluate a possibly infeasible multiple-shooting trajectory.

    The default return remains the historical scalar ``float``.  With
    ``return_info=True`` this returns corrected rollouts, stage costs, timings,
    smoothness diagnostics, and (optionally) a discrete PMP adjoint.

    The PMP gradient is currently assembled by central finite differences of
    the local discrete maps.  It is useful for diagnostics and screening, but
    it is not an exact-autodiff gradient and does not carry an automatic error
    certificate.  Certified screening must supply a separately justified
    gradient-error bound and a segment validator.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if rollout_length < 0:
        raise ValueError("rollout_length must be non-negative")
    if compute_adjoint and not return_info:
        raise ValueError("compute_adjoint=True requires return_info=True")
    if derivative_eps <= 0.0:
        raise ValueError("derivative_eps must be positive")

    total_start = perf_counter()
    n = trajectory.horizon
    r = rollout_length
    nx = system.state_dim
    nu = system.control_dim
    total = n + r

    simulated = np.asarray(trajectory.states[0], dtype=float).reshape(nx).copy()
    corrected_states = np.zeros((total + 1, nx), dtype=float)
    corrected_controls = np.zeros((total, nu), dtype=float)
    raw_controls = np.zeros((total, nu), dtype=float)
    stage_costs = np.zeros(total, dtype=float)
    corrected_states[0] = simulated
    logs: list[dict[str, Any]] = []
    forward_start = perf_counter()
    cost = 0.0
    discount = 1.0

    for k in range(total):
        step_start = perf_counter() if return_info else 0.0
        if k < n:
            correction = actor(simulated) - actor(trajectory.states[k])
            raw_control = trajectory.controls[k] + alpha * correction
            phase = "auxiliary_mpc"
        else:
            raw_control = actor(simulated)
            phase = "actor_rollout"
        raw_control = np.asarray(raw_control, dtype=float).reshape(nu)
        simulated_control = np.asarray(
            system.clip_control(raw_control), dtype=float
        ).reshape(nu)
        stage_cost = float(system.stage_cost(simulated, simulated_control))
        cost += discount * stage_cost
        simulated_next = np.asarray(
            system.step(simulated, simulated_control), dtype=float
        ).reshape(nx)

        raw_controls[k] = raw_control
        corrected_controls[k] = simulated_control
        stage_costs[k] = stage_cost
        corrected_states[k + 1] = simulated_next
        if return_info:
            logs.append(
                {
                    "step": k,
                    "phase": phase,
                    "stage_cost": stage_cost,
                    "discount": discount,
                    "raw_control": raw_control.copy(),
                    "applied_control": simulated_control.copy(),
                    "clip_active": not np.allclose(
                        raw_control, simulated_control, rtol=0.0, atol=1e-9
                    ),
                    "forward_seconds": perf_counter() - step_start,
                }
            )

        simulated = simulated_next
        discount *= gamma

    forward_seconds = perf_counter() - forward_start
    terminal_start = perf_counter()
    terminal_value = _critic_value(
        value_function,
        actor,
        simulated,
        value_kind=value_kind,
        critic_scale=critic_scale,
    )
    terminal_seconds = perf_counter() - terminal_start
    cost += discount * terminal_value

    if not return_info:
        return float(cost)

    timings: dict[str, float] = {
        "forward_seconds": forward_seconds,
        "terminal_value_seconds": terminal_seconds,
        "derivative_seconds": 0.0,
        "terminal_gradient_seconds": 0.0,
        "adjoint_seconds": 0.0,
        "total_seconds": 0.0,
    }
    smoothness: dict[str, Any] = {
        "locally_smooth_at_reference": not any(log["clip_active"] for log in logs),
        "clip_active_or_kink": any(log["clip_active"] for log in logs),
        "note": "A segment validator is still required to certify the entire reference-candidate segment.",
    }
    # Twin-Q critics use min(Q1,Q2), which is nonsmooth at branch crossings.
    # If a critic wrapper exposes branch_gap(state, action), record the terminal
    # branch margin so a segment validator can enforce a non-crossing condition.
    if value_kind == "q" and callable(getattr(value_function, "branch_gap", None)):
        terminal_action = np.asarray(actor(simulated), dtype=float).reshape(nu)
        try:
            smoothness["terminal_critic_branch_gap"] = float(
                value_function.branch_gap(simulated, terminal_action)
            )
        except Exception as exc:
            smoothness["terminal_critic_branch_gap"] = None
            smoothness["terminal_critic_branch_gap_error"] = f"{type(exc).__name__}: {exc}"
    pmp = None
    if compute_adjoint:
        pmp, derivative_smoothness, derivative_timing = _compute_pmp_adjoint(
            system,
            trajectory,
            actor,
            value_function,
            alpha=alpha,
            rollout_length=rollout_length,
            gamma=gamma,
            value_kind=value_kind,
            critic_scale=critic_scale,
            corrected_states=corrected_states,
            corrected_controls=corrected_controls,
            raw_controls=raw_controls,
            eps=derivative_eps,
            step_logs=logs,
        )
        smoothness.update(derivative_smoothness)
        timings.update(derivative_timing)
    timings["total_seconds"] = perf_counter() - total_start

    return Ac4EvalResult(
        cost=float(cost),
        corrected_states=corrected_states,
        corrected_controls=corrected_controls,
        stage_costs=stage_costs,
        terminal_value=float(terminal_value),
        step_logs=logs,
        timings=timings,
        smoothness=smoothness,
        pmp=pmp,
    )


def build_pmp_taylor_certificate(
    reference_trajectory: Trajectory,
    candidate_trajectory: Trajectory,
    reference_result: Ac4EvalResult,
    *,
    lipschitz_constant: float,
    gradient_error_bound: float,
    radius: float,
    segment_validator: Callable[[Trajectory, Trajectory, Ac4EvalResult], bool] | None,
) -> PMPResidualCertificate:
    """Build a conservative one-sided Taylor lower bound.

    If ``g_hat`` is the returned numerical PMP gradient, ``||g_hat-grad J|| <=
    eps_g`` and ``grad J`` is L-Lipschitz on the complete line segment, then

        J(z+d) >= J(z) + g_hat.T d - eps_g ||d|| - (L/2)||d||^2.

    The caller must provide *validated* L, eps_g and a validator that certifies
    the entire segment lies in the domain where those bounds hold.  This
    function intentionally refuses to certify when any item is missing.
    """

    def rejected(reason: str, delta_norm: float | None = None) -> PMPResidualCertificate:
        return PMPResidualCertificate(
            False, None, reference_result.cost, None, None, None, delta_norm,
            lipschitz_constant, gradient_error_bound, reason
        )

    if reference_result.pmp is None:
        return rejected("reference evaluation has no PMP adjoint")
    if segment_validator is None:
        return rejected("no segment validator supplied")
    if not np.isfinite(lipschitz_constant) or lipschitz_constant < 0.0:
        return rejected("invalid gradient-Lipschitz bound")
    if not np.isfinite(gradient_error_bound) or gradient_error_bound < 0.0:
        return rejected("invalid gradient-error bound")
    if not np.isfinite(radius) or radius <= 0.0:
        return rejected("invalid certification radius")
    if not reference_result.smoothness.get("locally_smooth_at_reference", False):
        return rejected("reference rollout touches a control-clipping kink")
    if reference_trajectory.horizon != candidate_trajectory.horizon:
        return rejected("candidate horizon differs from reference")
    if reference_trajectory.state_dim != candidate_trajectory.state_dim:
        return rejected("candidate state dimension differs from reference")
    if reference_trajectory.control_dim != candidate_trajectory.control_dim:
        return rejected("candidate control dimension differs from reference")
    if not np.allclose(reference_trajectory.states[0], candidate_trajectory.states[0], atol=1e-10, rtol=0.0):
        return rejected("candidate initial state differs from reference")

    delta = trajectory_decision_vector(candidate_trajectory) - trajectory_decision_vector(reference_trajectory)
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > radius:
        return rejected("candidate is outside the certified radius", delta_norm)
    try:
        segment_valid = bool(segment_validator(reference_trajectory, candidate_trajectory, reference_result))
    except Exception as exc:  # fail closed: validator errors must never enable screening
        return rejected(f"segment validator raised {type(exc).__name__}: {exc}", delta_norm)
    if not segment_valid:
        return rejected("segment validator rejected the reference-candidate segment", delta_norm)

    gradient = reference_result.pmp.gradient_vector
    if gradient.shape != delta.shape:
        return rejected("PMP gradient and candidate displacement dimensions differ", delta_norm)
    linear = float(gradient @ delta)
    curvature = 0.5 * float(lipschitz_constant) * delta_norm**2
    gradient_error = float(gradient_error_bound) * delta_norm
    lower = float(reference_result.cost + linear - curvature - gradient_error)
    return PMPResidualCertificate(
        True,
        lower,
        float(reference_result.cost),
        linear,
        curvature,
        gradient_error,
        delta_norm,
        float(lipschitz_constant),
        float(gradient_error_bound),
        "certified under caller-supplied L, gradient-error bound, and segment validator",
    )
