from __future__ import annotations

import numpy as np

from ac4mpc.snow_hill import SnowHillSystem, snow_hill_residual_acceleration


def test_snow_hill_force_matches_paper_shape() -> None:
    assert np.isclose(snow_hill_residual_acceleration(-10.0), 0.0)
    assert np.isclose(snow_hill_residual_acceleration(0.0), 0.0)
    assert np.isclose(snow_hill_residual_acceleration(-5.0), -1.25)


def test_rk4_step_has_expected_dimensions() -> None:
    system = SnowHillSystem()
    next_state = system.step(np.array([-5.0, -1.0]), np.array([1.0]))
    assert next_state.shape == (2,)
    assert np.all(np.isfinite(next_state))


def test_stage_cost_is_nonnegative() -> None:
    system = SnowHillSystem()
    assert system.stage_cost(np.array([0.0, 0.0]), np.array([0.0])) >= 0.0

