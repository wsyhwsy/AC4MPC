from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SB3SACActor:
    """Deterministic maximum-likelihood SAC action wrapper."""

    model: object

    @classmethod
    def load(cls, path: str) -> "SB3SACActor":
        from stable_baselines3 import SAC

        return cls(SAC.load(path))

    def __call__(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(np.asarray(state, dtype=np.float32), deterministic=True)
        return np.asarray(action, dtype=float).reshape(-1)


@dataclass
class SB3SACQCost:
    """Convert an SB3 SAC reward Q critic into a cost Q critic."""

    model: object
    reward_to_cost: bool = True

    @classmethod
    def load(cls, path: str, *, reward_to_cost: bool = True) -> "SB3SACQCost":
        from stable_baselines3 import SAC

        return cls(SAC.load(path), reward_to_cost=reward_to_cost)

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        if action is None:
            raise ValueError("SB3SACQCost requires an action.")
        import torch

        device = self.model.device
        obs = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=device)
        act = torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(1, -1), device=device)
        with torch.no_grad():
            q_values = self.model.critic(obs, act)
            q_reward = torch.min(torch.cat(q_values, dim=1), dim=1).values.item()
        return float(-q_reward if self.reward_to_cost else q_reward)


def export_sb3_sac_weights(model_path: str | Path, output_path: str | Path) -> Path:
    """Export an SB3 SAC checkpoint to exact NumPy weights for acados/CasADi use."""

    from stable_baselines3 import SAC

    model = SAC.load(str(model_path))
    arrays: dict[str, np.ndarray] = {
        "action_low": np.asarray(model.action_space.low, dtype=float),
        "action_high": np.asarray(model.action_space.high, dtype=float),
    }
    for key, value in model.policy.actor.state_dict().items():
        arrays[f"actor::{key}"] = value.detach().cpu().numpy().astype(float)
    for key, value in model.policy.critic.state_dict().items():
        arrays[f"critic::{key}"] = value.detach().cpu().numpy().astype(float)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return output


def _load_npz_dict(path: str | Path, prefix: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            key[len(prefix) :]: np.asarray(data[key], dtype=float)
            for key in data.files
            if key.startswith(prefix)
        }


def _linear_numpy(weight: np.ndarray, bias: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(weight, dtype=float) @ np.asarray(x, dtype=float) + np.asarray(bias, dtype=float)


@dataclass
class NumpySACActor:
    """Pure NumPy deterministic SAC actor exported from an SB3 checkpoint."""

    weights: dict[str, np.ndarray]
    action_high: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "NumpySACActor":
        with np.load(path, allow_pickle=False) as data:
            action_high = np.asarray(data["action_high"], dtype=float)
        return cls(_load_npz_dict(path, "actor::"), action_high)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float).reshape(-1)
        h = np.tanh(_linear_numpy(self.weights["latent_pi.0.weight"], self.weights["latent_pi.0.bias"], x))
        h = np.tanh(_linear_numpy(self.weights["latent_pi.2.weight"], self.weights["latent_pi.2.bias"], h))
        mu = _linear_numpy(self.weights["mu.weight"], self.weights["mu.bias"], h)
        return np.asarray(self.action_high, dtype=float).reshape(-1) * np.tanh(mu)


@dataclass
class NumpySACQCost:
    """Pure NumPy SAC twin-Q cost critic exported from an SB3 checkpoint."""

    weights: dict[str, np.ndarray]
    reward_to_cost: bool = True

    @classmethod
    def load(cls, path: str | Path, *, reward_to_cost: bool = True) -> "NumpySACQCost":
        return cls(_load_npz_dict(path, "critic::"), reward_to_cost=reward_to_cost)

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        if action is None:
            raise ValueError("NumpySACQCost requires an action.")
        x = np.concatenate([np.asarray(state, dtype=float).reshape(-1), np.asarray(action, dtype=float).reshape(-1)])
        q0 = self._q("qf0", x)
        q1 = self._q("qf1", x)
        q_reward = min(q0, q1)
        return float(-q_reward if self.reward_to_cost else q_reward)

    def _q(self, prefix: str, x: np.ndarray) -> float:
        h = np.tanh(_linear_numpy(self.weights[f"{prefix}.0.weight"], self.weights[f"{prefix}.0.bias"], x))
        h = np.tanh(_linear_numpy(self.weights[f"{prefix}.2.weight"], self.weights[f"{prefix}.2.bias"], h))
        q = _linear_numpy(self.weights[f"{prefix}.4.weight"], self.weights[f"{prefix}.4.bias"], h)
        return float(np.asarray(q).reshape(-1)[0])
