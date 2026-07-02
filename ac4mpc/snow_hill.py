from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def huber(x: np.ndarray | float, delta: float = 1.0) -> np.ndarray:
    """Huber function H_delta from the paper preliminaries."""

    x_arr = np.asarray(x, dtype=float)
    abs_x = np.abs(x_arr)
    return np.where(abs_x <= delta, 0.5 * x_arr * x_arr, delta * (abs_x - 0.5 * delta))


def snow_hill_residual_acceleration(position: np.ndarray | float) -> np.ndarray:
    """Residual acceleration a_res(p) from the commented paper formula."""

    p = np.asarray(position, dtype=float)
    force_shape = (huber(p + 7.0, delta=1.0) - huber(p + 3.0, delta=1.0)) ** 2
    return 1.25 * (force_shape / 16.0 - 1.0)


@dataclass(frozen=True)
class SnowHillConfig:
    """Snow hill parameters specified by the paper plus explicit cost config.

    The paper gives the cost form sqrt(s^T Q s + 1) + u^T R u but not the
    numerical Q/R entries in main.tex. The defaults are smoke-test values.
    """

    dt: float = 0.1
    control_limit: float = 1.0
    q: tuple[float, float] = (1.0, 1.0)
    r: tuple[float] = (0.01,)
    gamma: float = 1.0
    simulation_steps: int = 200
    eval_position_bounds: tuple[float, float] = (-12.0, 4.0)
    eval_velocity_bounds: tuple[float, float] = (-3.0, 3.0)


class SnowHillSystem:
    """One-dimensional point mass in the paper snow hill environment."""

    state_dim = 2
    control_dim = 1

    def __init__(self, config: SnowHillConfig | None = None) -> None:
        self.config = config or SnowHillConfig()
        self.Q = np.diag(np.asarray(self.config.q, dtype=float))
        self.R = np.diag(np.asarray(self.config.r, dtype=float))

    def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(2)
        control = np.asarray(control, dtype=float).reshape(1)
        p, v = state
        u = control[0]
        return np.array([v, u + float(snow_hill_residual_acceleration(p))], dtype=float)

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Discrete-time transition F(s,u) via RK4 with t_d = 0.1 s."""

        x = np.asarray(state, dtype=float).reshape(2)
        u = np.asarray(control, dtype=float).reshape(1)
        dt = self.config.dt
        k1 = self.continuous_dynamics(x, u)
        k2 = self.continuous_dynamics(x + 0.5 * dt * k1, u)
        k3 = self.continuous_dynamics(x + 0.5 * dt * k2, u)
        k4 = self.continuous_dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def clip_control(self, control: np.ndarray) -> np.ndarray:
        limit = self.config.control_limit
        return np.clip(np.asarray(control, dtype=float).reshape(1), -limit, limit)

    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        state = np.asarray(state, dtype=float).reshape(2)
        control = np.asarray(control, dtype=float).reshape(1)
        state_term = float(np.sqrt(state.T @ self.Q @ state + 1.0))
        control_term = float(control.T @ self.R @ control)
        return state_term + control_term

    def simulate_controls(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=float).reshape(-1, 1)
        states = np.zeros((controls.shape[0] + 1, self.state_dim), dtype=float)
        states[0] = np.asarray(state, dtype=float).reshape(2)
        for k, u in enumerate(controls):
            states[k + 1] = self.step(states[k], u)
        return states

    def bounds_for_controls(self, horizon: int) -> list[tuple[float, float]]:
        limit = self.config.control_limit
        return [(-limit, limit) for _ in range(horizon * self.control_dim)]


try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency path
    gym = None
    spaces = None


if gym is not None:

    class SnowHillGymEnv(gym.Env):
        """Gymnasium wrapper for training paper-style SAC/PPO policies."""

        metadata = {"render_modes": []}

        def __init__(self, config: SnowHillConfig | None = None, seed: int | None = None) -> None:
            super().__init__()
            self.system = SnowHillSystem(config)
            self.config = self.system.config
            self.rng = np.random.default_rng(seed)
            self.step_count = 0
            p_low, p_high = self.config.eval_position_bounds
            v_low, v_high = self.config.eval_velocity_bounds
            self.observation_space = spaces.Box(
                low=np.array([p_low, v_low], dtype=np.float32),
                high=np.array([p_high, v_high], dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.array([-self.config.control_limit], dtype=np.float32),
                high=np.array([self.config.control_limit], dtype=np.float32),
                dtype=np.float32,
            )
            self.state = np.zeros(2, dtype=float)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            options = options or {}
            if "state" in options:
                self.state = np.asarray(options["state"], dtype=float).reshape(2)
            else:
                p_low, p_high = self.config.eval_position_bounds
                v_low, v_high = self.config.eval_velocity_bounds
                self.state = np.array(
                    [self.rng.uniform(p_low, p_high), self.rng.uniform(v_low, v_high)],
                    dtype=float,
                )
            self.step_count = 0
            return self.state.astype(np.float32), {}

        def step(self, action):
            action = self.system.clip_control(np.asarray(action, dtype=float))
            cost = self.system.stage_cost(self.state, action)
            self.state = self.system.step(self.state, action)
            self.step_count += 1
            terminated = False
            truncated = self.step_count >= self.config.simulation_steps
            return self.state.astype(np.float32), -float(cost), terminated, truncated, {"cost": float(cost)}

else:
    SnowHillGymEnv = None

