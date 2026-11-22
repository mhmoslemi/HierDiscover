"""Check reward invariance and exact step recovery for historical advantages."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import orchestrator
from config_validation import HYBRID_INVARIANT_MODE, INVARIANT_ADVANTAGE_MODES
from problems.base import RewardResult, SeedState
from search.sampler import make_sampler
from training.advantage import ENTROPIC_KL_BUDGET, INVARIANT_LOGIT_CAP
from training.spo_rs import InvariantSPORSTracker


def make_tracker(adapter="adapter_initial"):
    return InvariantSPORSTracker(d_half=0.06, rho_min=0.875, rho_max=0.96, policy_adapter=adapter)


@pytest.mark.parametrize("scale", [3.0, 1e-15, 1e150])
@pytest.mark.parametrize("mix", [None, 0.35])
def test_affine_invariance_with_failures_and_caps(scale, mix):
    trackers = make_tracker(), make_tracker()
    shift = -7.0 * scale
    for step, rewards in enumerate(([1.0, 1.3, 0.0, 1.8, 1.7, 2.0],
                                    [1.3, 1.4, 0.0, 1000.0, 1.4, 1.1])):
        valid = np.array([True, True, False, True, True, True])
        scaled = np.where(valid, np.array(rewards) * scale + shift, 0.0)
        original, transformed = [
            tracker.calculate(values, valid, group_size=3, mixing_weight=mix, target_rms=1.7)
            for tracker, values in zip(trackers, (rewards, scaled))
        ]
        np.testing.assert_allclose(original.applied, transformed.applied, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(original.historical, transformed.historical, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(
            scale * original.effective_rewards + shift, transformed.effective_rewards, atol=0, rtol=1e-10)
        if mix is not None:
            assert np.sqrt(np.mean(original.applied ** 2)) == pytest.approx(1.7)
        if step == 0:
            assert original.info["pooled_start"]
            assert original.info["tilt_kl"] == pytest.approx(ENTROPIC_KL_BUDGET)
        else:
            assert not original.info["pooled_start"]
            assert original.info["theta"] == INVARIANT_LOGIT_CAP
        for tracker, batch in zip(trackers, (original, transformed)):
            tracker.update(batch, divergence=0.01, policy_adapter="adapter_initial", step=step)
    assert trackers[0].state_dict()["weights"] == [2 ** (-0.01 / 0.06)] * 6 + [1.0] * 6


def test_all_failures_keep_empty_history_and_constant_rewards_have_zero_advantage():
    tracker = make_tracker()
    failed = tracker.calculate([0.0, 0.0], [False, False], group_size=2, mixing_weight=0.5)
    assert failed.effective_rewards is None
    np.testing.assert_array_equal(failed.applied, [0.0, 0.0])
    tracker.update(failed, divergence=0.0, policy_adapter="sampled", step=0)
    assert len(tracker) == 0
    assert tracker.last_policy_adapter == "sampled"
    tied = tracker.calculate([2.0, 2.0], [True, True], group_size=2, mixing_weight=0.5)
    np.testing.assert_array_equal(tied.applied, [0.0, 0.0])
    tracker.update(tied, divergence=0.0, policy_adapter="sampled", step=1)
    np.testing.assert_array_equal(
        tracker.calculate([2.0, 2.0], [True, True], group_size=2).applied, [0.0, 0.0])


def test_restored_history_uses_decayed_weights_and_valid_only_failure_variance():
    tracker = make_tracker()
    for step, rewards in enumerate(([0.0, 1.0, 2.0], [0.0, 1.5, 3.0])):
        batch = tracker.calculate(rewards, [False, True, True], group_size=3)
        tracker.update(batch, divergence=0.1, policy_adapter="sampled", step=step)
    state = json.loads(json.dumps(tracker.state_dict()))
    restored = make_tracker()
    restored.load_state_dict(state)
    batch = restored.calculate([1.4, 0.0, 3.1], [True, False, True], group_size=3)

    history = np.array(state["rewards"])
    weights = np.array(state["weights"])
    valid = np.array(state["valid"])
    mean = np.average(history[valid], weights=weights[valid])
    std = np.sqrt(np.average((history[valid] - mean) ** 2, weights=weights[valid]))
    assert batch.effective_rewards[1] == pytest.approx(1.0 - std)
    beta = batch.info["beta"]
    partition = np.average(np.exp(beta * history), weights=weights)
    np.testing.assert_allclose(batch.applied, np.exp(beta * batch.effective_rewards) / partition - 1)


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        return messages[0]["content"]

    def __call__(self, prompt, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[1]]) if return_tensors else [1])


@pytest.mark.parametrize("mode", INVARIANT_ADVANTAGE_MODES)
@pytest.mark.parametrize("no_train", [False, True])
def test_step_and_evaluated_recovery_use_same_labels(tmp_path, mode, no_train):
    cfg = SimpleNamespace(
        advantage_mode=mode, groups_per_step=2, group_size=2, no_train=no_train,
        hybrid_invariant_lambda=0.4, hybrid_invariant_target_rms=1.7,
        generation_backend="vllm", evaluation_gpu_id=None, kl_penalty_coef=0.1,
        reward_workers=2, sandbox_timeout_s=10, max_saved_construction=0,
        max_new_tokens=20, temperature=1.0, top_p=1.0, seed=42,
        coder_model_name="test-coder",
    )
    adapter = tmp_path / "adapter_initial"
    adapter.mkdir()
    model, tokenizer = SimpleNamespace(device="cpu"), Tokenizer()
    seeds = [SeedState(code="parent-a"), SeedState(code="parent-b")]
    sampler = make_sampler(num_seeds=2, seed_states=seeds)
    optimizer = Mock()
    optimizer.state_dict.return_value = {}
    tracker = make_tracker(adapter)
    orchestrator._save_training_checkpoint(tmp_path, 0, adapter, sampler, optimizer, tracker)

    def generate(**options):
        for group, rewards in enumerate(((1.0, 2.0), (0.0, 1.5))):
            responses = []
            for index, reward in enumerate(rewards):
                text = f"```python\n{reward}\n```" if reward else "malformed response"
                tokens = [10 + index]
                responses.append((text, tokens, [-0.4]) if options["return_logprobs"] else (text, tokens))
            yield group, responses

    def evaluate(text, parent, timeout, **options):
        code = text.splitlines()[1]
        return RewardResult(reward=float(code), raw_score=float(code), code=code,
                            valid=True, parsed=True, ran=True)

    problem = SimpleNamespace(
        maximize=True, metric_name="score", fail_score=0.0, compute_reward=evaluate,
        build_prompt=lambda parent: [{"role": "user", "content": parent.code}],
    )
    pool = SimpleNamespace(
        sequential=False, host_memory_admission=object(),
        iter_group_jobs=Mock(side_effect=generate),
        score_token_logprobs=Mock(side_effect=lambda pairs, **kwargs: [[-0.6] for _ in pairs]),
    )
    trainer = None if no_train else Mock()
    if trainer is not None:
        trainer.train_spo_rs.return_value = {}
    orchestrator.train_step(None, model, tokenizer, sampler, optimizer, 0, cfg, tmp_path, problem, pool,
                            parallel_trainer=trainer, spo_rs_tracker=tracker, sampling_adapter_path=adapter)
    assert pool.iter_group_jobs.call_args.kwargs["full_policy_sampling"] == (mode == HYBRID_INVARIANT_MODE)
    assert pool.score_token_logprobs.called == (mode == HYBRID_INVARIANT_MODE and not no_train)
    artifacts = orchestrator._recoverable_spo_rs_artifacts(tmp_path, 0, cfg)
    assert len(artifacts) == 4
    failure = artifacts[2]["meta"]
    assert failure["reward"] == 0.0
    assert orchestrator._literal_metadata_mapping(failure["spo_rs"])["effective_reward"] > 0.0
    assert len(orchestrator._recoverable_pending_spo_rs_artifacts(tmp_path, 0, cfg)) == 4
    orchestrator._prepare_recoverable_response_tokens(model, tokenizer, artifacts)

    checkpoint = orchestrator._load_training_checkpoint(tmp_path)
    recovered = make_tracker(adapter)
    recovered.load_state_dict(checkpoint["spo_rs_tracker"])
    sampler.load_state_dict(checkpoint["sampler"])
    orchestrator._resume_spo_rs_step_from_artifacts(
        model, tokenizer, sampler, 0, cfg, tmp_path, problem, trainer, recovered, adapter, artifacts)
    assert recovered.state_dict() == tracker.state_dict()
    if trainer is not None:
        assert not trainer.begin_spo_rs_overlap.called
        assert trainer.train_spo_rs.call_count == 2
        fresh, replay = (call.args[0] for call in trainer.train_spo_rs.call_args_list)
        assert [example["advantage"] for example in fresh] == [example["advantage"] for example in replay]

    # Interrupt metadata sealing after generation is complete.
    partial = dict(artifacts[-1]["meta"])
    partial.pop("advantage")
    artifacts[-1]["meta_path"].write_text(json.dumps(partial), encoding="utf-8")
    assert orchestrator._recoverable_spo_rs_artifacts(tmp_path, 0, cfg) is None
    pending = orchestrator._recoverable_pending_spo_rs_artifacts(tmp_path, 0, cfg)
    recovered.load_state_dict(checkpoint["spo_rs_tracker"])
    sampler.load_state_dict(checkpoint["sampler"])
    pool.iter_group_jobs.reset_mock()
    orchestrator.train_step(None, model, tokenizer, sampler, optimizer, 0, cfg, tmp_path, problem, pool,
                            parallel_trainer=trainer, spo_rs_tracker=recovered, sampling_adapter_path=adapter,
                            pending_artifacts=pending)
    assert not pool.iter_group_jobs.called
    assert recovered.state_dict() == tracker.state_dict()
