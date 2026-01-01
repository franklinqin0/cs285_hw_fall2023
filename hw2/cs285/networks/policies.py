import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_tensor = ptu.from_numpy(obs)
        action_tensor = self.forward(obs_tensor).sample()
        action = ptu.to_numpy(action_tensor)

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)
        return None

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray = None,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
    ) -> dict:
        """Implements PPO-style policy gradient update with clipping and entropy regularization."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # Compute new log probabilities and entropy
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions) # (batch_size, action_dim)
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(-1) # sum over action_dim if continuous
        
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(-1)
        
        # PPO-style clipped importance sampling
        if old_log_probs is not None:
            old_log_probs = ptu.from_numpy(old_log_probs)
            ratio = torch.exp(log_probs - old_log_probs)
            # Clipped surrogate objective
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        else:
            ratio = torch.ones_like(log_probs)
            policy_loss = -(log_probs * advantages).mean()
        
        # Add entropy bonus for exploration
        entropy_loss = -entropy_coef * entropy.mean()
        
        loss = policy_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "Actor Loss": ptu.to_numpy(policy_loss),
            "Entropy": ptu.to_numpy(entropy.mean()),
            "Importance Ratio Mean": ptu.to_numpy(ratio.mean()),
        }
