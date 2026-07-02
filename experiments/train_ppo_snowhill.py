from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ac4mpc.ppo import (  # noqa: E402
    TanhGaussianActorCritic,
    compute_gae_acmpc,
    ppo_loss,
    save_ppo_checkpoint,
)
from ac4mpc.snow_hill import SnowHillConfig, SnowHillSystem  # noqa: E402


def reset_states(system: SnowHillSystem, rng: np.random.Generator, num_envs: int) -> np.ndarray:
    p_low, p_high = system.config.eval_position_bounds
    v_low, v_high = system.config.eval_velocity_bounds
    return np.stack(
        [rng.uniform(p_low, p_high, size=num_envs), rng.uniform(v_low, v_high, size=num_envs)],
        axis=-1,
    ).astype(np.float32)


def evaluate_policy(system: SnowHillSystem, model: TanhGaussianActorCritic, initial_state, steps: int) -> dict:
    state = np.asarray(initial_state, dtype=float).reshape(system.state_dim)
    total_cost = 0.0
    for _ in range(steps):
        action = system.clip_control(model.deterministic_action_np(state))
        total_cost += system.stage_cost(state, action)
        state = system.step(state, action)
    return {"total_cost": float(total_cost), "final_state": state.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train snow hill actor/critic with the ACMPC PPO objective.")
    parser.add_argument("--total-iters", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--actor-critic-lr", type=float, default=3e-4)
    parser.add_argument("--clip-param", type=float, default=0.12)
    parser.add_argument("--entropy-coeff", type=float, default=0.04)
    parser.add_argument("--value-loss-coeff", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("ac4mpc_reproduction/runs/ppo_snowhill"))
    parser.add_argument("--q", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--r", type=float, nargs=1, default=(0.01,))
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--initial-state", type=float, nargs=2, default=(-5.0, -1.0))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    system = SnowHillSystem(SnowHillConfig(q=tuple(args.q), r=tuple(args.r), simulation_steps=args.episode_len))
    model = TanhGaussianActorCritic(
        system.state_dim,
        system.control_dim,
        hidden_dim=args.hidden_dim,
        action_limit=system.config.control_limit,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.actor_critic_lr)

    states = reset_states(system, rng, args.num_envs)
    lengths = np.zeros(args.num_envs, dtype=np.int64)
    log_path = args.out / "train_log.jsonl"

    with log_path.open("w", encoding="utf-8") as log_file:
        for iteration in range(1, args.total_iters + 1):
            state_records = []
            action_records = []
            log_prob_records = []
            reward_records = []
            done_records = []
            value_records = []

            for _ in range(args.rollout_steps):
                state_tensor = torch.as_tensor(states, dtype=torch.float32)
                with torch.no_grad():
                    actions_tensor, log_probs, _ = model.act(state_tensor, deterministic=False)
                    values = model.value(state_tensor)
                actions = actions_tensor.cpu().numpy()

                next_states = np.zeros_like(states)
                rewards = np.zeros(args.num_envs, dtype=np.float32)
                dones = np.zeros(args.num_envs, dtype=np.float32)
                for env_id in range(args.num_envs):
                    action = system.clip_control(actions[env_id])
                    cost = system.stage_cost(states[env_id], action)
                    rewards[env_id] = -float(cost)
                    next_states[env_id] = system.step(states[env_id], action).astype(np.float32)
                    lengths[env_id] += 1
                    if lengths[env_id] >= args.episode_len:
                        dones[env_id] = 1.0
                        next_states[env_id] = reset_states(system, rng, 1)[0]
                        lengths[env_id] = 0

                state_records.append(state_tensor)
                action_records.append(actions_tensor.detach())
                log_prob_records.append(log_probs.detach())
                reward_records.append(torch.as_tensor(rewards))
                done_records.append(torch.as_tensor(dones))
                value_records.append(values.detach())
                states = next_states

            with torch.no_grad():
                bootstrap = model.value(torch.as_tensor(states, dtype=torch.float32))
            rewards_t = torch.stack(reward_records, dim=1)
            dones_t = torch.stack(done_records, dim=1)
            values_t = torch.cat([torch.stack(value_records, dim=1), bootstrap[:, None]], dim=1)
            advantages, returns = compute_gae_acmpc(
                rewards_t,
                values_t,
                dones_t,
                gamma=args.gamma,
                lam=args.gae_lambda,
            )

            flat_states = torch.stack(state_records, dim=1).reshape(-1, system.state_dim)
            flat_actions = torch.stack(action_records, dim=1).reshape(-1, system.control_dim)
            flat_old_log_prob = torch.stack(log_prob_records, dim=1).reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_advantages = advantages.reshape(-1)

            sample_count = flat_states.shape[0]
            metrics_accum = []
            for _ in range(args.ppo_epochs):
                order = torch.randperm(sample_count)
                for start in range(0, sample_count, args.mini_batch_size):
                    idx = order[start : start + args.mini_batch_size]
                    loss, metrics = ppo_loss(
                        model,
                        flat_states[idx],
                        flat_actions[idx],
                        flat_old_log_prob[idx],
                        flat_returns[idx],
                        flat_advantages[idx],
                        clip_param=args.clip_param,
                        entropy_coeff=args.entropy_coeff,
                        value_loss_coeff=args.value_loss_coeff,
                        advantage_normalization=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    metrics_accum.append(metrics)

            eval_info = evaluate_policy(system, model, args.initial_state, args.eval_steps)
            mean_metrics = {
                key: float(np.mean([m[key] for m in metrics_accum]))
                for key in metrics_accum[0]
            }
            entry = {
                "iteration": iteration,
                "mean_rollout_reward": float(rewards_t.mean().item()),
                "eval_total_cost": eval_info["total_cost"],
                "eval_final_state": eval_info["final_state"],
                **mean_metrics,
            }
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            print(json.dumps(entry, indent=2))

    ckpt_path = args.out / "ppo_snowhill.pt"
    save_ppo_checkpoint(
        ckpt_path,
        model,
        {
            "algorithm": "ACMPC-style PPO",
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "q": list(args.q),
            "r": list(args.r),
            "iterations": args.total_iters,
        },
    )
    summary = evaluate_policy(system, model, args.initial_state, args.eval_steps)
    (args.out / "summary.json").write_text(json.dumps({"checkpoint": str(ckpt_path), **summary}, indent=2))
    print(json.dumps({"checkpoint": str(ckpt_path), **summary}, indent=2))


if __name__ == "__main__":
    main()

