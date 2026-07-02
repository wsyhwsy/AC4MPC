from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Actor(Protocol):
    """Deterministic actor policy pi_hat(s)."""

    def __call__(self, state: np.ndarray) -> np.ndarray:
        ...


class ValueFunction(Protocol):
    """State value or state-action Q value used as a cost critic."""

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        ...


@dataclass
class CallableActor:
    fn: Callable[[np.ndarray], np.ndarray]
    control_limit: float | np.ndarray | None = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        action = np.asarray(self.fn(np.asarray(state, dtype=float)), dtype=float).reshape(-1)
        if self.control_limit is not None:
            limit = np.asarray(self.control_limit, dtype=float)
            action = np.clip(action, -limit, limit)
        return action


@dataclass
class CallableValue:
    fn: Callable[[np.ndarray], float]

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        del action
        return float(self.fn(np.asarray(state, dtype=float)))


@dataclass
class CallableQValue:
    fn: Callable[[np.ndarray, np.ndarray], float]

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        if action is None:
            raise ValueError("Q-value critics require an action.")
        return float(self.fn(np.asarray(state, dtype=float), np.asarray(action, dtype=float)))


class ZeroValue:
    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        del state, action
        return 0.0


@dataclass
class LinearFeedbackActor:
    """Small placeholder actor for smoke tests, not a paper result."""

    gain: tuple[float, float] = (-0.45, -0.75)
    control_limit: float = 1.0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float).reshape(-1)
        raw = self.gain[0] * s[0] + self.gain[1] * s[1]
        return np.array([np.clip(raw, -self.control_limit, self.control_limit)], dtype=float)


class TanhMlp:
    """PyTorch two-hidden-layer tanh MLP matching the paper architecture."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        import torch
        from torch import nn

        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self) -> "TanhMlp":
        self.model.eval()
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        tensor = self.torch.as_tensor(np.asarray(x, dtype=np.float32)).view(1, -1)
        with self.torch.no_grad():
            out = self.model(tensor).cpu().numpy()[0]
        return out


@dataclass
class TorchActor:
    module: object
    control_limit: float | np.ndarray = 1.0
    squash_tanh: bool = True

    def __call__(self, state: np.ndarray) -> np.ndarray:
        out = np.asarray(self.module(np.asarray(state, dtype=np.float32)), dtype=float).reshape(-1)
        if self.squash_tanh:
            out = np.tanh(out)
        return np.asarray(self.control_limit, dtype=float) * out


@dataclass
class TorchValue:
    module: object

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        del action
        return float(np.asarray(self.module(np.asarray(state, dtype=np.float32))).reshape(-1)[0])


@dataclass
class TorchQValue:
    module: object

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        if action is None:
            raise ValueError("TorchQValue requires an action.")
        x = np.concatenate([np.asarray(state, dtype=np.float32).reshape(-1), np.asarray(action, dtype=np.float32).reshape(-1)])
        return float(np.asarray(self.module(x)).reshape(-1)[0])

