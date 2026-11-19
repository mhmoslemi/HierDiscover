from types import SimpleNamespace
import unittest

import numpy as np
import torch

from training.hybrid import hybrid_advantages, reference_logprob_mean
from training.trainers import _clipped_policy_options, historical_policy_loss


class HybridTrainingTest(unittest.TestCase):
    def test_hybrid_advantages_redistribute_sibling_credit(self):
        rewards = np.array([4.87, 4.89, 4.91, 4.93])
        historical = np.expm1(2 * (rewards - 4.93)) / 2
        actual, beta = hybrid_advantages(rewards, historical)
        np.testing.assert_allclose(actual, [-0.080538, -0.059188, -0.026543, 0.051682], atol=1e-6)
        self.assertAlmostEqual(beta, 72.292961, places=5)
        self.assertAlmostEqual(actual.mean(), historical.mean())
        doubled, _ = hybrid_advantages(rewards, historical, sibling_weight=0.20)
        np.testing.assert_allclose(doubled - historical, 2 * (actual - historical))
        disabled, _ = hybrid_advantages(rewards, historical, sibling_weight=0.0)
        np.testing.assert_array_equal(disabled, historical)
        tied, beta = hybrid_advantages([1.0] * 4, [0.25] * 4)
        np.testing.assert_array_equal(tied, [0.25] * 4)
        self.assertEqual(beta, 0.0)

    def test_ttt_token_sum_and_gradient_on_a_fixed_batch(self):
        sampler = [torch.tensor([-1.8, -0.2]), torch.tensor([-1.1, -0.4, -1.7])]
        reference = [torch.tensor([-2.0, -0.1]), torch.tensor([-0.6, -1.3, -2.6])]
        learner = [torch.tensor([-0.9, -0.3], requires_grad=True),
                   torch.tensor([-2.8, -0.2, -1.6], requires_grad=True)]
        advantages = [0.8, -0.5]
        coeff = 0.1
        delta = [old - base for old, base in zip(sampler, reference)]
        expected_center = sum(d.sum() for d in delta) / sum(d.numel() for d in delta)
        center = reference_logprob_mean(
            (len(old), old, base) for old, base in zip(sampler, reference))
        self.assertEqual(center, expected_center.item())

        cfg = SimpleNamespace(advantage_mode="hybrid", kl_penalty_coef=coeff)
        actual = sum(historical_policy_loss(cfg, current, {
            "old_policy_logprobs": old, "reference_logprobs": base,
            "advantage": advantage, "reference_logprob_mean": center,
        }, None)[0] for current, old, base, advantage in zip(learner, sampler, reference, advantages))
        expected = sum(-((current - old).exp() * (advantage + coeff * (expected_center - d))).sum()
                       for current, old, advantage, d in zip(learner, sampler, advantages, delta))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for got, wanted in zip(torch.autograd.grad(actual, learner), torch.autograd.grad(expected, learner)):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)

    def test_spo_rs_still_uses_clipping_and_a_token_mean(self):
        cfg = SimpleNamespace(advantage_mode="spo-rs", coder_model_name="openai/gpt-oss-120b")
        current = torch.tensor([-0.2, -2.0], requires_grad=True)
        old = torch.tensor([-1.0, -1.0])
        loss, _ = historical_policy_loss(cfg, current, {
            "old_policy_logprobs": old, "advantage": 1.0,
        }, _clipped_policy_options(cfg))
        self.assertAlmostEqual(float(loss), -(1.35 + np.exp(-1)) / 2)
        self.assertEqual(torch.autograd.grad(loss, current)[0][0].item(), 0.0)
