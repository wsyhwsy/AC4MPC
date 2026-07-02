from __future__ import annotations

import numpy as np

from .models import Actor
from .types import Trajectory


def actor_rollout(system, state: np.ndarray, actor: Actor, horizon: int, *, source: str = "actor") -> Trajectory:
    """Compute Phi_hat and Psi_hat from Eq. (cl_pi)."""

    states = np.zeros((horizon + 1, system.state_dim), dtype=float)
    controls = np.zeros((horizon, system.control_dim), dtype=float)
    states[0] = np.asarray(state, dtype=float).reshape(system.state_dim)
    for k in range(horizon):
        controls[k] = system.clip_control(actor(states[k]))
        states[k + 1] = system.step(states[k], controls[k])
    return Trajectory(states, controls, source=source)


def shift_trajectory(system, trajectory: Trajectory, actor: Actor, *, source: str | None = None) -> Trajectory:
    """Shift states and controls using xi and zeta from the AC4MPC-RTI section."""

    if trajectory.horizon <= 0:
        return trajectory.copy(source=source)

    shifted_states = np.zeros_like(trajectory.states)
    shifted_controls = np.zeros_like(trajectory.controls)
    shifted_states[:-1] = trajectory.states[1:]
    shifted_controls[:-1] = trajectory.controls[1:]

    last_state = trajectory.states[-1]
    last_control = system.clip_control(actor(last_state))
    shifted_controls[-1] = last_control
    shifted_states[-1] = system.step(last_state, last_control)

    return Trajectory(
        shifted_states,
        shifted_controls,
        source=trajectory.source if source is None else source,
        info=dict(trajectory.info),
    )


def max_dynamics_gap(system, trajectory: Trajectory) -> float:
    """Maximum multiple-shooting gap ||F(s_k,u_k)-s_{k+1}||_inf."""

    if trajectory.horizon == 0:
        return 0.0
    gaps = []
    for k in range(trajectory.horizon):
        next_state = system.step(trajectory.states[k], trajectory.controls[k])
        gaps.append(float(np.max(np.abs(next_state - trajectory.states[k + 1]))))
    return max(gaps)

