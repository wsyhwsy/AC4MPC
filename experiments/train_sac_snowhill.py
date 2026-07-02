from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ac4mpc.snow_hill import SnowHillConfig, SnowHillGymEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paper-style SAC on the snow hill environment.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("runs/snowhill_sac"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--r", type=float, nargs=1, default=(0.01,))
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging when tensorboard is installed.")
    args = parser.parse_args()

    if SnowHillGymEnv is None:
        raise RuntimeError("gymnasium is required for training.")

    from stable_baselines3 import SAC

    args.out.mkdir(parents=True, exist_ok=True)
    env = SnowHillGymEnv(SnowHillConfig(q=tuple(args.q), r=tuple(args.r)), seed=args.seed)
    policy_kwargs = {"net_arch": [256, 256], "activation_fn": torch.nn.Tanh}
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=args.verbose,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        tensorboard_log=str(args.out / "tb") if args.tensorboard else None,
    )

    checkpoints = [step for step in (50_000, 100_000) if step <= args.timesteps]
    if args.timesteps not in checkpoints:
        checkpoints.append(args.timesteps)

    trained = 0
    for target in checkpoints:
        model.learn(total_timesteps=target - trained, reset_num_timesteps=(trained == 0))
        trained = target
        model.save(args.out / f"sac_{target}")


if __name__ == "__main__":
    main()
