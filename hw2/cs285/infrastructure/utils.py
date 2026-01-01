from collections import OrderedDict
import numpy as np
import copy
from cs285.networks.policies import MLPPolicy
import gymnasium as gym
import cv2
from cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy (non-vectorized)."""
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render()
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # TODO use the most recent ob and the policy to decide what to do
        ac: np.ndarray = policy.get_action(ob)  # HINT: this is a numpy array

        # TODO: use that action to take a step in the environment
        next_ob, rew, done, _, _ = env.step(ac)

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done: bool = done or (steps >= max_length)

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories_vectorized(
    env: gym.vector.VectorEnv,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy in a vectorized environment."""
    num_envs = env.num_envs
    
    # Initialize storage for each environment
    obs_lists = [[] for _ in range(num_envs)]
    acs_lists = [[] for _ in range(num_envs)]
    rewards_lists = [[] for _ in range(num_envs)]
    next_obs_lists = [[] for _ in range(num_envs)]
    terminals_lists = [[] for _ in range(num_envs)]
    image_obs_lists = [[] for _ in range(num_envs)]
    
    # Track steps per environment
    steps = np.zeros(num_envs, dtype=np.int32)
    
    # Completed trajectories
    trajs = []
    total_timesteps = 0
    
    # Reset all environments
    obs, _ = env.reset()  # obs shape: (num_envs, ob_dim)
    
    while total_timesteps < min_timesteps_per_batch:
        # Render if needed - for vectorized envs, try to get rendered frames
        if render:
            try:
                if hasattr(env, 'call'):
                    # For AsyncVectorEnv, use call method to render from sub-environments
                    imgs = env.call('render')
                elif hasattr(env, 'render'):
                    imgs = env.render()
                else:
                    imgs = None
                
                if imgs is not None:
                    if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
                        # Shape: (num_envs, H, W, C)
                        for i, img in enumerate(imgs):
                            image_obs_lists[i].append(
                                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
                            )
                    elif isinstance(imgs, (list, tuple)):
                        for i, img in enumerate(imgs):
                            if img is not None:
                                image_obs_lists[i].append(
                                    cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
                                )
            except Exception:
                pass  # Skip rendering if it fails
        
        # Get actions for all observations (batched)
        acs = policy.get_action(obs)  # shape: (num_envs, ac_dim) or (num_envs,)
        
        # Step all environments
        next_obs, rews, dones, truncs, infos = env.step(acs)
        steps += 1
        
        # Check for episode end (done, truncated, or max_length)
        rollout_dones = dones | truncs | (steps >= max_length)
        
        # Record data for each environment
        for i in range(num_envs):
            obs_lists[i].append(obs[i])
            acs_lists[i].append(acs[i])
            rewards_lists[i].append(rews[i])
            next_obs_lists[i].append(next_obs[i])
            terminals_lists[i].append(rollout_dones[i])
            
            # If this environment's episode is done, save trajectory
            if rollout_dones[i]:
                traj = {
                    "observation": np.array(obs_lists[i], dtype=np.float32),
                    "image_obs": np.array(image_obs_lists[i], dtype=np.uint8) if image_obs_lists[i] else np.array([], dtype=np.uint8),
                    "reward": np.array(rewards_lists[i], dtype=np.float32),
                    "action": np.array(acs_lists[i], dtype=np.float32),
                    "next_observation": np.array(next_obs_lists[i], dtype=np.float32),
                    "terminal": np.array(terminals_lists[i], dtype=np.float32),
                }
                trajs.append(traj)
                total_timesteps += len(rewards_lists[i])
                
                # Reset storage for this environment
                obs_lists[i] = []
                acs_lists[i] = []
                rewards_lists[i] = []
                next_obs_lists[i] = []
                terminals_lists[i] = []
                image_obs_lists[i] = []
                steps[i] = 0
        
        obs = next_obs
    
    return trajs, total_timesteps


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    # Check if environment is vectorized (handle both gym.vector and gymnasium.experimental.vector)
    is_vectorized = hasattr(env, 'num_envs') and env.num_envs > 1
    if is_vectorized:
        return sample_trajectories_vectorized(env, policy, min_timesteps_per_batch, max_length, render)
    
    # Original non-vectorized implementation
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    # For vectorized envs, use the vectorized sampling and collect enough trajectories
    is_vectorized = hasattr(env, 'num_envs') and env.num_envs > 1
    if is_vectorized:
        trajs = []
        while len(trajs) < ntraj:
            # Sample enough timesteps to likely get at least one trajectory per env
            new_trajs, _ = sample_trajectories_vectorized(
                env, policy, max_length * env.num_envs, max_length, render
            )
            trajs.extend(new_trajs)
        return trajs[:ntraj]
    
    # Original non-vectorized implementation
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])
