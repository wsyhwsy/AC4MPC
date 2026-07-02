from __future__ import annotations

import numpy as np

from ac4mpc.algorithms import AC4MPCRTIConfig, AC4MPCRTIController
from ac4mpc.models import LinearFeedbackActor, ZeroValue
from ac4mpc.snow_hill import SnowHillSystem
from ac4mpc.solvers import MultipleShootingSolverConfig, ScipyMultipleShootingSolver


def test_rti_controller_returns_bounded_action_with_zero_solver_iterations() -> None:
    system = SnowHillSystem()
    actor = LinearFeedbackActor()
    value = ZeroValue()
    solver_config = MultipleShootingSolverConfig(horizon=3, max_iterations=1)
    active = ScipyMultipleShootingSolver(system, actor, value, solver_config)
    parallel = ScipyMultipleShootingSolver(system, actor, value, solver_config)
    controller = AC4MPCRTIController(
        system,
        actor,
        value,
        active,
        parallel,
        AC4MPCRTIConfig(horizon=3, active_sqp_iterations=0, parallel_sqp_iterations=0),
    )
    step = controller.step(np.array([-5.0, -1.0]))
    assert step.action.shape == (1,)
    assert abs(step.action[0]) <= system.config.control_limit + 1e-12
    assert step.source in {"active", "parallel", "actor"}

