from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def compute_gae_acmpc(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """GAE copied from the ACMPC PPO loop, kept local for Python 3.8 envs."""

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, time]")
    batch, time = rewards.shape
    if values.shape != (batch, time + 1):
        raise ValueError("values must have shape [batch, time+1]")
    if dones.shape != (batch, time):
        raise ValueError("dones must have shape [batch, time]")

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(time)):
        mask = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * values[:, t + 1] * mask - values[:, t]
        gae = delta + gamma * lam * mask * gae
        advantages[:, t] = gae
    returns = advantages + values[:, :-1]
    return advantages, returns


def _atanh(x: Tensor) -> Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class TanhGaussianActorCritic(nn.Module):
    """Two-layer tanh actor and value critic for the paper AC4MPC PPO path."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        action_limit: float = 1.0,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_limit = float(action_limit)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def _normal(self, state: Tensor) -> torch.distributions.Normal:
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, state: Tensor) -> Tensor:
        return self.critic(state).squeeze(-1)

    def act(self, state: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        dist = self._normal(state)
        z = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(z) * self.action_limit
        log_prob, entropy = self.evaluate_action(state, action)
        return action, log_prob, entropy

    def evaluate_action(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        dist = self._normal(state)
        scaled = (action / self.action_limit).clamp(-0.999999, 0.999999)
        z = _atanh(scaled)
        base_log_prob = dist.log_prob(z).sum(dim=-1)
        correction = torch.log(self.action_limit * (1.0 - scaled.pow(2)) + 1e-8).sum(dim=-1)
        log_prob = base_log_prob - correction
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic_action_np(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state, dtype=np.float32)).view(1, -1)
            action, _, _ = self.act(x, deterministic=True)
        return action.cpu().numpy()[0]


@dataclass
class PPOCostActor:
    model: TanhGaussianActorCritic

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.model.deterministic_action_np(state)


@dataclass
class PPOCostValue:
    """Cost value wrapper J(s) = -V_reward(s)."""

    model: TanhGaussianActorCritic

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> float:
        del action
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state, dtype=np.float32)).view(1, -1)
            reward_value = self.model.value(x).cpu().item()
        return float(-reward_value)


def save_ppo_checkpoint(path: str | Path, model: TanhGaussianActorCritic, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_dim": model.hidden_dim,
            "action_limit": model.action_limit,
            "metadata": metadata,
        },
        path,
    )


def load_ppo_cost_actor_critic(path: str | Path) -> tuple[PPOCostActor, PPOCostValue, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    model = TanhGaussianActorCritic(
        int(checkpoint["state_dim"]),
        int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        action_limit=float(checkpoint["action_limit"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return PPOCostActor(model), PPOCostValue(model), dict(checkpoint.get("metadata", {}))


def ppo_loss(
    model: TanhGaussianActorCritic,
    states: Tensor,
    actions: Tensor,
    old_log_prob: Tensor,
    returns: Tensor,
    advantages: Tensor,
    *,
    clip_param: float,
    entropy_coeff: float,
    value_loss_coeff: float,
    advantage_normalization: bool,
) -> tuple[Tensor, dict[str, float]]:
    if advantage_normalization and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    log_prob, entropy = model.evaluate_action(states, actions)
    values = model.value(states)
    ratio = torch.exp(log_prob - old_log_prob)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy.mean()
    value_loss = F.mse_loss(values, returns)
    total_loss = policy_loss + value_loss_coeff * value_loss
    approx_kl = (old_log_prob - log_prob).mean()
    clip_fraction = ((ratio < 1.0 - clip_param) | (ratio > 1.0 + clip_param)).float().mean()
    metrics = {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.mean().detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
    }
    return total_loss, metrics

