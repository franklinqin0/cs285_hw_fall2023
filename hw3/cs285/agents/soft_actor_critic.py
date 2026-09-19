from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
        train_actor: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"
        if target_critic_backup_type in ("doubleq", "redq"):
            assert num_critic_networks >= 2, "Double-Q and REDQ need at least two critics"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy
        self.train_actor = train_actor

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            # Under no_grad, rsample is detached and avoids torch.normal's CPU check.
            action: torch.Tensor = action_distribution.rsample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # TODO(student): Implement the different backup strategies.
        if self.target_critic_backup_type == "doubleq":
            next_qs = torch.roll(next_qs, shifts=1, dims=0)
        elif self.target_critic_backup_type == "min":
            next_qs = next_qs.min(dim=0).values
        elif self.target_critic_backup_type == "mean":
            next_qs = next_qs.mean(dim=0)
        elif self.target_critic_backup_type == "redq":
            # Independently choose two distinct target critics per transition.
            first = torch.randint(num_critic_networks, (1, batch_size), device=next_qs.device)
            second = torch.randint(num_critic_networks - 1, (1, batch_size), device=next_qs.device)
            second = second + (second >= first).long()
            next_qs = torch.minimum(
                next_qs.gather(0, first), next_qs.gather(0, second)
            ).squeeze(0)
        else:
            # Default, we don't need to do anything.
            pass


        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        log_stats: bool = True,
        tensor_stats: bool = False,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        loss, q_values, target_values = self.critic_loss_tensors(obs, action, reward, next_obs, done)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        if not log_stats:
            return {}
        stats = {"critic_loss": loss.detach(), "q_values": q_values.mean(), "target_values": target_values.mean()}
        return stats if tensor_stats else {k: v.item() for k, v in stats.items()}

    def critic_loss_tensors(self, obs, action, reward, next_obs, done):
        """Tensor-only loss computation, also used by the optional compiler."""
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student)
            # Sample from the actor
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.rsample()

            # Compute the next Q-values for the sampled actions
            next_qs = self.target_critic(next_obs, next_action)

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(student): Add entropy bonus to the target values for SAC
                next_action_entropy = self.entropy(next_action_distribution)
                next_qs += self.temperature * next_action_entropy

            # Compute the target Q-value
            target_values: torch.Tensor = reward + self.discount * (1 - done.float()) * next_qs
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            )

        # TODO(student): Update the critic
        # Predict Q-values
        q_values = self.critic(obs=obs, action=action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        return loss, q_values.detach(), target_values.detach()

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        # TODO(student): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        return -action_distribution.log_prob(action_distribution.rsample())

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # TODO(student): Generate an action distribution
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            # REINFORCE needs detached actions. no_grad detaches rsample while
            # avoiding the synchronizing std check in Normal.sample on CUDA.
            action = action_distribution.rsample(sample_shape=(self.num_actor_samples,))
            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), action.shape

            # TODO(student): Compute Q-values for the current state-action pair
            # q_values = self.critic(obs=obs, action=action)
            sampled_obs = obs.unsqueeze(0).expand(self.num_actor_samples, -1, -1)
            q_values = self.critic(obs=sampled_obs, action=action)
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, axis=0)
            advantage = q_values

        # Do REINFORCE: calculate log-probs and use the Q-values
        # TODO(student)
        log_probs = action_distribution.log_prob(action)
        loss = -torch.mean(log_probs * advantage)

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # Pathwise gradient: .rsample() keeps a = mu_theta + sigma_theta * eps differentiable
        # w.r.t. theta (NOT under no_grad here), so grad flows theta -> action -> Q -> loss.
        action = action_distribution.rsample()

        # TODO(student): Compute Q-values for the sampled state-action pair
        q_values = self.critic(obs=obs, action=action)

        # Maximize Q -> minimize -Q. mean() over the whole tensor collapses both the
        # ensemble and batch dims (no min/backup: no bootstrap here, so no overestimation to fight).
        loss = -torch.mean(q_values)

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor, log_stats: bool = True, tensor_stats: bool = False):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        # Keep dQ/da for the policy gradient without computing unused dQ/dw.
        critic_params = list(self.critics.parameters())
        requires_grad = [p.requires_grad for p in critic_params]
        try:
            for p in critic_params:
                p.requires_grad_(False)
            if self.actor_gradient_type == "reparametrize":
                loss, entropy = self.actor_loss_reparametrize(obs)
            elif self.actor_gradient_type == "reinforce":
                loss, entropy = self.actor_loss_reinforce(obs)

            # Add entropy if necessary
            if self.use_entropy_bonus:
                loss -= self.temperature * entropy

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        finally:
            for p, enabled in zip(critic_params, requires_grad):
                p.requires_grad_(enabled)

        if not log_stats:
            return {}
        stats = {"actor_loss": loss.detach(), "entropy": entropy.detach()}
        return stats if tensor_stats else {k: v.item() for k, v in stats.items()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    @torch.no_grad()
    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.lerp_(param, tau)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
        log_stats: bool = True,
        tensor_stats: bool = False,
        update_schedules: bool = True,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos

        # TODO(student): Update the actor
        for _ in range(self.num_critic_updates):
            critic_infos.append(
                self.update_critic(
                    observations, actions, rewards, next_observations, dones,
                    log_stats=log_stats,
                    tensor_stats=tensor_stats,
                )
            )

        # The bootstrapping sanity check evaluates a fixed policy.
        if self.train_actor:
            actor_info = self.update_actor(observations, log_stats=log_stats, tensor_stats=tensor_stats)
        else:
            with torch.no_grad():
                # Preserve the entropy sample (and RNG sequence) between log steps.
                entropy = self.entropy(self.actor(observations)).mean()
                actor_info = {"entropy": entropy if tensor_stats else entropy.item()} if log_stats else {}
        # TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        if self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)
        elif (step + 1) % self.target_update_period == 0:
            self.update_target_critic()

        # Average the critic info over all of the steps
        critic_info = {
            k: (torch.stack([info[k] for info in critic_infos]).mean() if tensor_stats
                else np.mean([info[k] for info in critic_infos])) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        if update_schedules:
            self.step_lr_schedules()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

    def step_lr_schedules(self):
        if self.train_actor:
            self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
