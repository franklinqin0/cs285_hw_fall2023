import unittest
import copy
from unittest.mock import patch

import numpy as np
import torch

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.env_configs.sac_config import sac_config
from cs285.infrastructure import pytorch_util as ptu, utils
from cs285.infrastructure.replay_buffer import ReplayBuffer


class SACBootstrappingTest(unittest.TestCase):
    def setUp(self):
        ptu.init_gpu(use_gpu=False)
        torch.manual_seed(1)
        self.config = sac_config(
            env_name="Pendulum-v1",
            hidden_size=8,
            num_layers=1,
            discount=0.9,
            target_update_period=2,
            train_actor=False,
            use_entropy_bonus=False,
        )
        self.agent = SoftActorCritic((3,), 1, **self.config["agent_kwargs"])
        self.batch = dict(
            observations=torch.zeros(2, 3),
            actions=torch.zeros(2, 1),
            rewards=torch.tensor([2.0, 3.0]),
            next_observations=torch.zeros(2, 3),
            dones=torch.tensor([True, False]),
        )

    def test_bellman_target_masks_terminal_transitions(self):
        with torch.no_grad():
            for parameter in self.agent.critics.parameters():
                parameter.zero_()
            for parameter in self.agent.target_critics.parameters():
                parameter.zero_()
            self.agent.target_critics[0].net[-2].bias.fill_(5.0)

        stats = self.agent.update_critic(
            self.batch["observations"], self.batch["actions"],
            self.batch["rewards"], self.batch["next_observations"],
            self.batch["dones"],
        )
        # A terminal target is 2; a continuing target is 3 + 0.9 * 5.
        self.assertAlmostEqual(stats["target_values"], 4.75)
        self.assertAlmostEqual(stats["critic_loss"], 30.125)
        self.assertTrue(all(p.grad is None for p in self.agent.target_critics.parameters()))

    def test_mean_backup_shares_ensemble_average(self):
        self.agent.num_critic_networks = 3
        values = torch.tensor([[1.0, 8.0], [3.0, 2.0], [5.0, 5.0]])
        expected = torch.tensor([[3.0, 5.0]]).expand(3, 2)
        torch.testing.assert_close(self.agent.q_backup_strategy(values), expected)

    def test_redq_uses_a_distinct_pair_per_transition(self):
        self.agent.num_critic_networks = 4
        self.agent.target_critic_backup_type = "redq"
        values = torch.tensor([[1., 10., 100.], [2., 20., 200.], [3., 30., 300.], [4., 40., 400.]])
        # The second raw index skips the first: pairs (0,1), (2,3), (3,1).
        with patch("torch.randint", side_effect=[torch.tensor([[0, 2, 3]]), torch.tensor([[0, 2, 1]])]):
            actual = self.agent.q_backup_strategy(values)
        expected = torch.tensor([[1., 30., 200.]]).expand(4, 3)
        torch.testing.assert_close(actual, expected)

    def test_redq_with_two_critics_is_clipped_double_q(self):
        self.agent.num_critic_networks = 2
        self.agent.target_critic_backup_type = "redq"
        values = torch.randn(2, 100)
        expected = values.min(0).values.expand(2, 100)
        torch.testing.assert_close(self.agent.q_backup_strategy(values), expected)

    def test_fixed_actor_and_hard_target_schedule(self):
        actor_before = [p.detach().clone() for p in self.agent.actor.parameters()]
        target_before = [p.detach().clone() for p in self.agent.target_critics.parameters()]
        self.agent.num_critic_updates = 2
        stats = self.agent.update(**self.batch, step=0)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(target_before, self.agent.critics.parameters())
        ))
        for old, new in zip(target_before, self.agent.target_critics.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

        self.agent.update(**self.batch, step=1)
        for old, new in zip(actor_before, self.agent.actor.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        for source, target in zip(self.agent.critics.parameters(), self.agent.target_critics.parameters()):
            torch.testing.assert_close(source, target, rtol=0, atol=0)

    def test_soft_target_update_interpolates_after_training(self):
        self.agent.target_update_period = None
        self.agent.soft_target_update_rate = 0.25
        target_before = [p.detach().clone() for p in self.agent.target_critics.parameters()]
        self.agent.update(**self.batch, step=0)
        for old, source, target in zip(
            target_before, self.agent.critics.parameters(), self.agent.target_critics.parameters()
        ):
            torch.testing.assert_close(target, 0.75 * old + 0.25 * source)

    def test_replay_batch_runs_with_entropy_and_boolean_dones(self):
        self.agent.use_entropy_bonus = True
        self.agent.temperature = 0.1
        replay = ReplayBuffer(capacity=4)
        for index in range(7):
            replay.insert(
                observation=np.zeros(3, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=float(index),
                next_observation=np.ones(3, dtype=np.float32),
                done=index % 2 == 0,
            )
        batch = ptu.from_numpy(replay.sample(4))
        stats = self.agent.update(**batch, step=0)
        self.assertEqual(len(replay), 4)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_rollouts_respect_truncation_and_length_limit(self):
        env = self.config["make_env"]()
        try:
            path = utils.sample_trajectory(env, self.agent, max_length=1000)
            self.assertEqual(len(path["reward"]), 200)
            self.assertTrue(path["terminal"][-1])
            path = utils.sample_trajectory(env, self.agent, max_length=7)
            self.assertEqual(len(path["reward"]), 7)
        finally:
            env.close()

    def test_skipping_stats_preserves_updates_and_rng(self):
        for train_actor in (False, True):
            for gradient_type in ("reinforce", "reparametrize"):
                with self.subTest(train_actor=train_actor, gradient_type=gradient_type):
                    logged = SoftActorCritic((3,), 1, **self.config["agent_kwargs"])
                    quiet = SoftActorCritic((3,), 1, **self.config["agent_kwargs"])
                    for agent in (logged, quiet):
                        agent.load_state_dict(self.agent.state_dict())
                        agent.train_actor = train_actor
                        agent.actor_gradient_type = gradient_type
                        agent.num_critic_updates = 2
                        agent.use_entropy_bonus = True
                        agent.temperature = 0.1
                    torch.manual_seed(42)
                    logged.update(**self.batch, step=0)
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(42)
                    stats = quiet.update(**self.batch, step=0, log_stats=False)
                    self.assertNotIn("critic_loss", stats)
                    self.assertNotIn("actor_loss", stats)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                    for name, value in logged.state_dict().items():
                        torch.testing.assert_close(value, quiet.state_dict()[name], rtol=0, atol=0)

    def test_actor_update_preserves_policy_gradient_without_critic_gradients(self):
        self.agent.actor_gradient_type = "reparametrize"
        self.agent.use_entropy_bonus = True
        self.agent.temperature = 0.1
        reference = copy.deepcopy(self.agent)
        torch.manual_seed(42)
        loss, entropy = reference.actor_loss_reparametrize(self.batch["observations"])
        loss = loss - reference.temperature * entropy
        reference.actor_optimizer.zero_grad()
        loss.backward()
        reference.actor_optimizer.step()

        before = [p.detach().clone() for p in self.agent.actor.parameters()]
        torch.manual_seed(42)
        self.agent.update_actor(self.batch["observations"], log_stats=False)
        for expected, actual in zip(reference.actor.parameters(), self.agent.actor.parameters()):
            torch.testing.assert_close(expected, actual)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, self.agent.actor.parameters())))
        self.assertTrue(all(p.grad is None for p in self.agent.critics.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.agent.critics.parameters()))


if __name__ == "__main__":
    unittest.main()
