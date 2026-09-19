"""Measure environment + replay + SAC update throughput, excluding evaluation."""
import argparse
import json
import time

import numpy as np
import torch

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.sac_cuda_graph import SACGraphUpdater
from scripting_utils import make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", "-cfg", required=True)
    parser.add_argument("--cuda_graph", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--which_gpu", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    torch.distributions.Distribution.set_default_validate_args(False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(gpu_id=args.which_gpu)
    if ptu.device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA")
    torch.cuda.set_device(ptu.device)
    config = make_config(args.config_file)
    env = config["make_env"]()
    observation, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    agent = SoftActorCritic(observation.shape, env.action_space.shape[0], **config["agent_kwargs"])
    replay = ReplayBuffer(capacity=10000)
    try:
        for _ in range(2000):
            action = env.action_space.sample()
            nxt, reward, terminated, truncated, _ = env.step(action)
            replay.insert(observation=observation, action=action, reward=reward, next_observation=nxt, done=terminated)
            observation = env.reset()[0] if terminated or truncated else nxt
        updater = agent
        capture_seconds = 0.0
        if args.cuda_graph:
            start = time.perf_counter()
            updater = SACGraphUpdater(agent, ptu.from_numpy(replay.sample(config["batch_size"])))
            torch.cuda.synchronize()
            capture_seconds = time.perf_counter() - start
        elapsed = 0.0
        for step in range(args.warmup + args.steps):
            if step == args.warmup:
                torch.cuda.synchronize()
                start = time.perf_counter()
            action = agent.get_action(observation)
            nxt, reward, terminated, truncated, _ = env.step(action)
            replay.insert(observation=observation, action=action, reward=reward, next_observation=nxt, done=terminated)
            observation = env.reset()[0] if terminated or truncated else nxt
            batch = ptu.from_numpy(replay.sample(config["batch_size"]))
            updater.update(**batch, step=step, log_stats=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(json.dumps({"config": args.config_file, "cuda_graph": args.cuda_graph,
                          "steps": args.steps, "seconds": elapsed, "capture_seconds": capture_seconds,
                          "steps_per_second": args.steps / elapsed,
                          "projected_training_hours_5m": 5e6 / args.steps * elapsed / 3600}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
