from __future__ import annotations

import numpy as np

from ac4mpc.evaluation import ac4eval, finite_horizon_cost
from ac4mpc.models import LinearFeedbackActor, ZeroValue
from ac4mpc.rollout import actor_rollout, max_dynamics_gap, shift_trajectory
from ac4mpc.snow_hill import SnowHillSystem


def test_ac4eval_matches_feasible_open_loop_when_alpha_zero_and_no_terminal() -> None:
    system = SnowHillSystem()
    actor = LinearFeedbackActor()
    value = ZeroValue()
    trajectory = actor_rollout(system, np.array([-5.0, -1.0]), actor, horizon=4)
    assert max_dynamics_gap(system, trajectory) < 1e-12
    expected = finite_horizon_cost(
        system,
        trajectory,
        actor,
        value,
        terminal_rollout=0,
        gamma=1.0,
    )
    got = ac4eval(
        system,
        trajectory,
        actor,
        value,
        alpha=0.0,
        rollout_length=0,
        gamma=1.0,
    )
    assert np.isclose(got, expected)


def test_shift_preserves_horizon_and_appends_actor_control() -> None:
    system = SnowHillSystem()
    actor = LinearFeedbackActor()
    trajectory = actor_rollout(system, np.array([-5.0, -1.0]), actor, horizon=5)
    shifted = shift_trajectory(system, trajectory, actor)
    assert shifted.horizon == trajectory.horizon
    assert shifted.states.shape == trajectory.states.shape
    assert shifted.controls.shape == trajectory.controls.shape

