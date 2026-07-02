from __future__ import annotations

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
) -> float:
    """Evaluate a possibly infeasible multiple-shooting trajectory.

    Implements the auxiliary control law from Eq. (auxlaw):

        u_bar_k = u_k + alpha * (pi(s_bar_k) - pi(s_k))

    The terminal actor roll-out uses exactly R extra actor controls as in
    Eq. (V_f). The pseudo-code loop in main.tex is inclusive at N+R, while the
    equations and surrounding text define R actor steps; this implementation is
    equation-consistent.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if rollout_length < 0:
        raise ValueError("rollout_length must be non-negative")

    simulated_state = trajectory.states[0].copy()
    cost = 0.0
    discount = 1.0
    for k in range(trajectory.horizon):
        correction = actor(simulated_state) - actor(trajectory.states[k])
        simulated_control = trajectory.controls[k] + alpha * correction
        simulated_control = system.clip_control(simulated_control)
        cost += discount * system.stage_cost(simulated_state, simulated_control)
        simulated_state = system.step(simulated_state, simulated_control)
        discount *= gamma

    for _ in range(rollout_length):
        actor_control = system.clip_control(actor(simulated_state))
        cost += discount * system.stage_cost(simulated_state, actor_control)
        simulated_state = system.step(simulated_state, actor_control)
        discount *= gamma

    cost += discount * _critic_value(
        value_function,
        actor,
        simulated_state,
        value_kind=value_kind,
        critic_scale=critic_scale,
    )
    return float(cost)

