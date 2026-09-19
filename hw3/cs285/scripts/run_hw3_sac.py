import os
import time
import yaml

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse
import json


def save_checkpoint(agent, logger, step, seed):
    path = os.path.join(logger._log_dir, "checkpoint.pt")
    torch.save(
        {"agent_state_dict": agent.state_dict(), "step": step, "seed": seed},
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    torch.set_num_threads(args.torch_num_threads)
    if args.no_distribution_validation or args.cuda_graph:
        torch.distributions.Distribution.set_default_validate_args(False)
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    if ptu.device.type == "cuda":
        torch.cuda.set_device(ptu.device)
    if args.cuda_graph and ptu.device.type != "cuda":
        raise ValueError("--cuda_graph requires a CUDA GPU")

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"]

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])
    updater = agent

    env.action_space.seed(args.seed)
    eval_env.reset(seed=args.seed + 1)
    observation, _ = env.reset(seed=args.seed)

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(student): Select an action
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=terminated,
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation, _ = env.reset()
        else:
            observation = next_observation

        # Train the agent
        if step >= config["training_starts"]:
            # Train on a tensor batch from replay, rather than the latest transition.
            batch = ptu.from_numpy(replay_buffer.sample(batch_size))
            if args.cuda_graph and updater is agent:
                from cs285.infrastructure.sac_cuda_graph import SACGraphUpdater
                capture_start = time.perf_counter()
                updater = SACGraphUpdater(agent, batch)
                print(f"CUDA graph ready in {time.perf_counter() - capture_start:.2f}s", flush=True)
            update_info = updater.update(
                observations=batch["observations"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                next_observations=batch["next_observations"],
                dones=batch["dones"],
                step=step,
                log_stats=step % args.log_interval == 0,
            )

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(agent, logger, step, args.seed)

        # Run evaluation
        if step % args.eval_interval == 0 or step == config["total_steps"] - 1:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )

    # Preserve the trained networks alongside TensorBoard logs for later inspection.
    save_checkpoint(agent, logger, step, args.seed)
    logger.flush()
    env.close()
    eval_env.close()
    render_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=50000)
    parser.add_argument(
        "--cuda_graph", action="store_true",
        help="Capture fixed-batch SAC updates on CUDA; requires soft targets and constant LR, disables distribution validation.",
    )
    parser.add_argument(
        "--torch_num_threads", type=int, default=1,
        help="CPU threads for PyTorch; small SAC networks usually benefit from 1.",
    )
    parser.add_argument(
        "--no_distribution_validation", action="store_true",
        help="Skip distribution argument checks to reduce GPU synchronization; omit this flag for debugging.",
    )

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)
    with open(args.config_file) as source:
        config_yaml = yaml.safe_load(source)
    with open(os.path.join(logger._log_dir, "run_config.json"), "w") as output:
        json.dump({"config": config_yaml, "args": vars(args), "torch_version": torch.__version__}, output, indent=2)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
