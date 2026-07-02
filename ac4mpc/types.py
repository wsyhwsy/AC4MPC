from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass
class Trajectory:
    """Multiple-shooting trajectory with states s_0..s_N and controls u_0..u_{N-1}."""

    states: Array
    controls: Array
    source: str = "unknown"
    objective: float | None = None
    feasible_gap: float | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=float)
        self.controls = np.asarray(self.controls, dtype=float)
        if self.states.ndim != 2:
            raise ValueError(f"states must be rank 2, got shape {self.states.shape}")
        if self.controls.ndim != 2:
            raise ValueError(f"controls must be rank 2, got shape {self.controls.shape}")
        if self.states.shape[0] != self.controls.shape[0] + 1:
            raise ValueError(
                "states must contain exactly one more time point than controls: "
                f"{self.states.shape[0]} vs {self.controls.shape[0]}"
            )

    @property
    def horizon(self) -> int:
        return int(self.controls.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def control_dim(self) -> int:
        return int(self.controls.shape[1])

    def copy(self, *, source: str | None = None) -> "Trajectory":
        return Trajectory(
            self.states.copy(),
            self.controls.copy(),
            source=self.source if source is None else source,
            objective=self.objective,
            feasible_gap=self.feasible_gap,
            info=dict(self.info),
        )


@dataclass
class ControllerStep:
    """Diagnostics returned by one closed-loop controller call."""

    action: Array
    chosen: Trajectory
    candidates: dict[str, Trajectory]
    costs: dict[str, float]
    source: str
    info: dict[str, Any] = field(default_factory=dict)

