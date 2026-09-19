import unittest
from unittest.mock import patch

import torch

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.env_configs.sac_config import sac_config
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.sac_cuda_graph import SACGraphUpdater


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SACGraphTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        ptu.init_gpu()
        self.validation = torch.distributions.Distribution._validate_args
        torch.distributions.Distribution.set_default_validate_args(False)
        self.config = sac_config(
            "Pendulum-v1", hidden_size=8, num_layers=1,
            num_critic_networks=2, target_critic_backup_type="redq",
            actor_gradient_type="reparametrize", num_critic_updates=2,
            use_soft_target_update=True, soft_target_update_rate=0.05,
        )
        self.batch = dict(
            observations=torch.randn(16, 3, device=ptu.device),
            actions=torch.randn(16, 1, device=ptu.device).tanh(),
            rewards=torch.randn(16, device=ptu.device),
            next_observations=torch.randn(16, 3, device=ptu.device),
            dones=torch.arange(16, device=ptu.device) % 2 == 0,
        )

    def tearDown(self):
        torch.distributions.Distribution.set_default_validate_args(self.validation)

    def make_agent(self):
        return SoftActorCritic((3,), 1, **self.config["agent_kwargs"])

    def test_capture_preserves_state_and_replays_fresh_randomness(self):
        agent = self.make_agent()
        # Exercise restoration of nonempty Adam states, not only fresh optimizers.
        agent.update(**self.batch, step=0)
        weights = {k: v.clone() for k, v in agent.state_dict().items()}
        moments = [{p: {k: v.clone() for k, v in state.items()} for p, state in opt.state.items()}
                   for opt in (agent.actor_optimizer, agent.critic_optimizer)]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        epoch = agent.actor_lr_scheduler.last_epoch
        updater = SACGraphUpdater(agent, self.batch)
        for name, old in weights.items():
            torch.testing.assert_close(old, agent.state_dict()[name], rtol=0, atol=0)
        for opt, states in zip((agent.actor_optimizer, agent.critic_optimizer), moments):
            for p, state in states.items():
                for key, old in state.items():
                    torch.testing.assert_close(old.to(ptu.device), opt.state[p][key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
        self.assertEqual(agent.actor_lr_scheduler.last_epoch, epoch)
        first = updater.update(**self.batch, step=1)
        second = updater.update(**self.batch, step=2)
        self.assertNotEqual(first["entropy"], second["entropy"])
        self.assertEqual(agent.actor_lr_scheduler.last_epoch, epoch + 2)
        self.assertTrue(all(torch.isfinite(v).all() for v in agent.state_dict().values()))
        for opt, states in zip((agent.actor_optimizer, agent.critic_optimizer), moments):
            count = 2 if opt is agent.actor_optimizer else 4
            for p, state in states.items():
                self.assertEqual(opt.state[p]["step"].item(), state["step"].item() + count)
        with self.assertRaises(ValueError):
            updater.update(**{k: v[:8] for k, v in self.batch.items()}, step=3)

    def test_replay_matches_eager_updates_with_controlled_noise(self):
        # Fix Gaussian noise to compare numerical updates independent of CUDA RNG
        # offset allocation. REDQ with two critics always reduces to min-Q.
        def deterministic_sample(distribution, sample_shape=torch.Size()):
            shape = distribution._extended_shape(sample_shape)
            return distribution.loc.expand(shape) + 0.25 * distribution.scale.expand(shape)

        for gradient_type in ("reparametrize", "reinforce"):
            with self.subTest(gradient_type=gradient_type):
                eager, graphed = self.make_agent(), self.make_agent()
                eager.actor_gradient_type = graphed.actor_gradient_type = gradient_type
                graphed.load_state_dict(eager.state_dict())
                with patch.object(torch.distributions.Normal, "rsample", deterministic_sample):
                    updater = SACGraphUpdater(graphed, self.batch)
                    for step in range(3):
                        batch = {k: v.clone() for k, v in self.batch.items()}
                        batch["rewards"] += step
                        expected = eager.update(**batch, step=step)
                        actual = updater.update(**batch, step=step)
                        for key in actual:
                            self.assertAlmostEqual(actual[key], expected[key], delta=2e-5)
                        for name, value in eager.state_dict().items():
                            torch.testing.assert_close(value, graphed.state_dict()[name], rtol=1e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
