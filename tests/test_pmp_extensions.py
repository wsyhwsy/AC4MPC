from __future__ import annotations

import numpy as np

from ac4mpc.evaluation import (
    Ac4EvalResult,
    ac4eval,
    build_pmp_taylor_certificate,
    trajectory_decision_vector,
)
from ac4mpc.models import LinearFeedbackActor, ZeroValue
from ac4mpc.rollout import actor_rollout
from ac4mpc.snow_hill import SnowHillSystem
from ac4mpc.types import Trajectory


def _make_case(horizon: int = 4):
    system = SnowHillSystem()
    actor = LinearFeedbackActor(gain=(-0.22, -0.31), control_limit=0.8)
    value = ZeroValue()
    trajectory = actor_rollout(system, np.array([-5.0, -0.7]), actor, horizon)
    # Keep the test away from control saturation and create nonzero shooting gaps.
    controls = trajectory.controls.copy() + np.linspace(-0.03, 0.025, horizon).reshape(-1, 1)
    controls = np.clip(controls, -0.5, 0.5)
    states = trajectory.states.copy()
    for k in range(1, horizon):
        states[k] += np.array([0.01 * k, -0.004 * k])
    return system, actor, value, Trajectory(states, controls)


def _replace_decision_vector(trajectory: Trajectory, z: np.ndarray) -> Trajectory:
    out = trajectory.copy()
    n = out.horizon
    nx = out.state_dim
    state_size = max(n - 1, 0) * nx
    out.states[1:-1] = z[:state_size].reshape(n - 1, nx)
    out.controls = z[state_size:].reshape(n, out.control_dim)
    return out


def test_ac4eval_legacy_scalar_and_info_result_match():
    system, actor, value, trajectory = _make_case()
    kwargs = dict(alpha=0.8, rollout_length=3, gamma=0.99)
    scalar = ac4eval(system, trajectory, actor, value, **kwargs)
    result = ac4eval(system, trajectory, actor, value, return_info=True, **kwargs)
    assert isinstance(scalar, float)
    assert isinstance(result, Ac4EvalResult)
    assert np.isclose(scalar, result.cost, rtol=0.0, atol=1e-12)
    assert result.corrected_states.shape == (trajectory.horizon + 3 + 1, system.state_dim)
    assert result.corrected_controls.shape == (trajectory.horizon + 3, system.control_dim)
    assert len(result.step_logs) == trajectory.horizon + 3
    assert "forward_seconds" in result.timings


def test_discrete_adjoint_gradient_matches_full_objective_finite_difference():
    system, actor, value, trajectory = _make_case()
    kwargs = dict(alpha=0.7, rollout_length=2, gamma=0.995)
    result = ac4eval(
        system,
        trajectory,
        actor,
        value,
        return_info=True,
        compute_adjoint=True,
        derivative_eps=2e-6,
        **kwargs,
    )
    z = trajectory_decision_vector(trajectory)
    numerical = np.empty_like(z)
    outer_eps = 2e-5
    for j in range(z.size):
        zp = z.copy()
        zm = z.copy()
        zp[j] += outer_eps
        zm[j] -= outer_eps
        fp = ac4eval(system, _replace_decision_vector(trajectory, zp), actor, value, **kwargs)
        fm = ac4eval(system, _replace_decision_vector(trajectory, zm), actor, value, **kwargs)
        numerical[j] = (fp - fm) / (2.0 * outer_eps)
    assert result.pmp is not None
    rel = np.linalg.norm(result.pmp.gradient_vector - numerical) / max(1.0, np.linalg.norm(numerical))
    assert rel < 2e-4, rel
    assert result.pmp.adjoints.shape == (trajectory.horizon + 2 + 1, system.state_dim)


def test_certificate_fails_closed_without_segment_validator():
    system, actor, value, trajectory = _make_case()
    candidate = trajectory.copy()
    candidate.controls[0, 0] += 0.01
    result = ac4eval(
        system, trajectory, actor, value,
        alpha=0.5, rollout_length=1, gamma=1.0,
        return_info=True, compute_adjoint=True,
    )
    cert = build_pmp_taylor_certificate(
        trajectory,
        candidate,
        result,
        lipschitz_constant=10.0,
        gradient_error_bound=1e-4,
        radius=1.0,
        segment_validator=None,
    )
    assert not cert.certified
    assert "no segment validator" in cert.reason


def test_certificate_uses_gradient_error_and_curvature_penalties():
    system, actor, value, trajectory = _make_case()
    candidate = trajectory.copy()
    candidate.controls[0, 0] += 0.01
    result = ac4eval(
        system, trajectory, actor, value,
        alpha=0.5, rollout_length=1, gamma=1.0,
        return_info=True, compute_adjoint=True,
    )
    cert = build_pmp_taylor_certificate(
        trajectory,
        candidate,
        result,
        lipschitz_constant=100.0,
        gradient_error_bound=0.02,
        radius=1.0,
        segment_validator=lambda _a, _b, _r: True,
    )
    assert cert.certified
    assert cert.lower_bound is not None
    assert np.isclose(
        cert.lower_bound,
        result.cost + cert.linear_term - cert.curvature_penalty - cert.gradient_error_penalty,
    )


def test_rti_controller_diagnostics_and_zero_displacement_screening():
    from ac4mpc.algorithms import AC4MPCRTIConfig, AC4MPCRTIController

    system = SnowHillSystem()
    actor = LinearFeedbackActor(gain=(-0.1, -0.1), control_limit=0.8)
    value = ZeroValue()

    class IdentitySolver:
        horizon = 4
        def solve(self, state, initial, *, max_iterations=None, source="solver"):
            return initial.copy(source=source)

    config = AC4MPCRTIConfig(
        horizon=4,
        active_sqp_iterations=0,
        parallel_sqp_iterations=0,
        evaluation_rollout=2,
        enable_pmp_screening=True,
        certificate_lipschitz=0.0,
        certificate_gradient_error=0.0,
        certificate_radius=1.0,
        certificate_segment_validator=lambda _a, _b, _r: True,
    )
    controller = AC4MPCRTIController(
        system, actor, value, IdentitySolver(), IdentitySolver(), config
    )
    step = controller.step(np.array([-5.0, -0.7]))
    assert step.source == "active"
    assert step.info["candidate_status"]["parallel"] == "screened_by_pmp_lower_bound"
    assert step.info["candidate_status"]["actor"] == "screened_by_pmp_lower_bound"
    details = step.info["pmp_evaluation_details"]["active"]
    assert details.pmp is not None
    assert details.pmp.adjoints.shape[0] == config.horizon + config.evaluation_rollout + 1
    assert step.info["timings"]["controller_total_seconds"] >= 0.0


def test_default_rti_path_keeps_all_exact_candidate_costs():
    from ac4mpc.algorithms import AC4MPCRTIConfig, AC4MPCRTIController

    system = SnowHillSystem()
    actor = LinearFeedbackActor(gain=(-0.1, -0.1), control_limit=0.8)
    value = ZeroValue()

    class IdentitySolver:
        horizon = 4
        def solve(self, state, initial, *, max_iterations=None, source="solver"):
            return initial.copy(source=source)

    controller = AC4MPCRTIController(
        system,
        actor,
        value,
        IdentitySolver(),
        IdentitySolver(),
        AC4MPCRTIConfig(horizon=4, evaluation_rollout=2),
    )
    step = controller.step(np.array([-5.0, -0.7]))
    assert set(step.costs) == {"active", "parallel", "actor"}
    assert step.info["screened_candidates"] == []
    assert step.info["pmp_evaluation_details"] == {}


def test_q_critic_path_includes_actor_dependence_in_terminal_gradient():
    system, actor, _value, trajectory = _make_case()

    class QuadraticQ:
        def __call__(self, state, action=None):
            x = np.asarray(state, dtype=float).reshape(-1)
            u = np.asarray(action, dtype=float).reshape(-1)
            return float(0.2 * (x @ x) + 0.3 * (u @ u) + 0.05 * x[0] * u[0])

    q_value = QuadraticQ()
    kwargs = dict(alpha=0.6, rollout_length=2, gamma=0.98, value_kind="q", critic_scale=0.7)
    result = ac4eval(
        system, trajectory, actor, q_value,
        return_info=True, compute_adjoint=True, derivative_eps=2e-6, **kwargs
    )
    z = trajectory_decision_vector(trajectory)
    j = min(2, z.size - 1)
    zp, zm = z.copy(), z.copy()
    zp[j] += 2e-5
    zm[j] -= 2e-5
    fp = ac4eval(system, _replace_decision_vector(trajectory, zp), actor, q_value, **kwargs)
    fm = ac4eval(system, _replace_decision_vector(trajectory, zm), actor, q_value, **kwargs)
    fd = (fp - fm) / 4e-5
    assert np.isclose(result.pmp.gradient_vector[j], fd, rtol=2e-4, atol=2e-5)


def test_certificate_rejects_reference_control_clip_activity():
    system, actor, value, trajectory = _make_case()
    trajectory.controls[:] = 3.0
    result = ac4eval(
        system, trajectory, actor, value,
        alpha=1.0, rollout_length=1, gamma=1.0,
        return_info=True, compute_adjoint=True,
    )
    candidate = trajectory.copy()
    cert = build_pmp_taylor_certificate(
        trajectory, candidate, result,
        lipschitz_constant=10.0,
        gradient_error_bound=1e-4,
        radius=1.0,
        segment_validator=lambda _a, _b, _r: True,
    )
    assert not cert.certified
    assert "control-clipping kink" in cert.reason
