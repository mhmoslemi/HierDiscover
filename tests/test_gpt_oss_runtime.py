import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.gpt_oss_lora import (
    SharedExpertLoRA, _with_shared_projections, disable_shared_adapter,
    load_shared_adapter, save_shared_adapter,
)
from training import fast_distributed
from training.optimizer import HostMomentAdamW
from vllm_compat.gpt_oss_lora import _interleave_legacy_w13


def _grouped_mm(inputs, weights, offsets):
    return torch.cat([inputs[start:end] @ weight for start, end, weight in zip(
        [0, *offsets[:-1]], offsets, weights)], dim=0)


@pytest.mark.parametrize("interleaved", [True, False])
def test_shared_expert_projections_match_independent_dense_deltas(interleaved):
    torch.manual_seed(23)
    adapter = SharedExpertLoRA(3, 5, 7, 2, 3, 0, "cpu").double()
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.normal_()

    offsets = [3, 3, 8]
    inputs = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
    inter = torch.randn(8, 7, dtype=torch.float64, requires_grad=True)
    gate_up = torch.randn(8, 14, dtype=torch.float64, requires_grad=True)
    down = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)

    actual_gate_up = adapter.project(gate_up, inputs, offsets, _grouped_mm, "gate_up", interleaved)
    actual_down = adapter.project(down, inter, offsets, _grouped_mm, "down", interleaved)
    expert_ids = [0] * 3 + [2] * 5
    gate = torch.stack([adapter.gate_b[e] @ adapter.gate_a[0] @ row for e, row in zip(expert_ids, inputs)])
    up = torch.stack([adapter.up_b[e] @ adapter.up_a[0] @ row for e, row in zip(expert_ids, inputs)])
    down_delta = torch.stack([adapter.down_b[0] @ adapter.down_a[e] @ row for e, row in zip(expert_ids, inter)])
    gate_up_delta = torch.stack((gate, up), dim=-1).flatten(-2) if interleaved else torch.cat((gate, up), dim=-1)
    expected_gate_up = gate_up + adapter.scale * gate_up_delta
    expected_down = down + adapter.scale * down_delta
    torch.testing.assert_close(actual_gate_up, expected_gate_up)
    torch.testing.assert_close(actual_down, expected_down)
    leaves = (inputs, inter, gate_up, down, *adapter.parameters())
    gradients = torch.autograd.grad(actual_gate_up.square().sum() + actual_down.square().sum(), leaves)
    expected = torch.autograd.grad(expected_gate_up.square().sum() + expected_down.square().sum(), leaves)
    for actual, reference in zip(gradients, expected):
        torch.testing.assert_close(actual, reference)


def _mxfp4_expert_layout(source, projection, hidden_dim, model_type, experts_module):
    return source


def _mxfp4_base_grouped_mm(inputs, offsets, counts, weight_provider, recompute, layout):
    return _grouped_mm(inputs, layout, offsets)


_grouped_mm_with_backward_fix = _grouped_mm


def _gate_up_is_interleaved(experts_module):
    return True


def _base_forward(experts, inputs, offsets):
    layout = _mxfp4_expert_layout(experts.weight, "gate_up", 4, "gpt_oss", experts)
    return _mxfp4_base_grouped_mm(inputs, offsets, None, None, True, layout)


def test_binding_preserves_original_forward_and_disables_adapter():
    adapter = SharedExpertLoRA(2, 4, 3, 2, 2, 0, "cpu")
    experts = nn.Module()
    experts.ttt_expert_adapter = adapter
    experts.weight = torch.randn(2, 4, 6)

    @contextmanager
    def disable_peft():
        yield

    experts.disable_adapter = disable_peft
    inputs = torch.randn(5, 4)
    offsets = [2, 5]
    patched = _with_shared_projections(_base_forward)
    initial = _base_forward(experts, inputs, offsets)
    torch.testing.assert_close(patched(experts, inputs, offsets), initial)
    with torch.no_grad():
        adapter.gate_b.fill_(1.0)
    adapted = patched(experts, inputs, offsets)
    assert not torch.equal(adapted, initial)
    compiled = torch.compile(patched, backend="aot_eager", fullgraph=True)
    output = compiled(experts, inputs, offsets)
    torch.testing.assert_close(output, adapted)
    output.square().sum().backward()
    assert adapter.gate_b.grad.abs().sum() > 0
    with disable_shared_adapter(experts):
        torch.testing.assert_close(patched(experts, inputs, offsets), initial)
    torch.testing.assert_close(_base_forward(experts, inputs, offsets), initial)


def test_safetensors_roundtrip_preserves_shared_vllm_layout(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    model = nn.Module()
    model.experts = nn.Module()
    model.experts.ttt_expert_adapter = SharedExpertLoRA(3, 5, 7, 2, 2, 0, "cpu")
    model.head = nn.Parameter(torch.randn(11, 2))
    model.active_adapter = "default"

    def save_config(path):
        (path / "adapter_config.json").write_text(json.dumps({"r": 2}))

    def read_peft(model, *, save_embedding_layers):
        assert save_embedding_layers is False
        return {"lm_head.lora_B.weight": model.head}

    def load_peft(model, weights):
        model.head.copy_(weights["lm_head.lora_B.weight"])

    model.peft_config = {"default": SimpleNamespace(save_pretrained=save_config)}
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        get_peft_model_state_dict=read_peft, set_peft_model_state_dict=load_peft))
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    save_shared_adapter(model, tmp_path)
    saved = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert saved["experts.w1.lora_A.weight"].shape == (1, 2, 5)
    assert saved["experts.w2.lora_A.weight"].shape == (3, 2, 7)
    assert saved["experts.w2.lora_B.weight"].shape == (1, 5, 2)
    assert saved["experts.w3.lora_B.weight"].shape == (3, 7, 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(-42.0)
    load_shared_adapter(model, tmp_path)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name])


def test_bucketed_adamw_updates_each_parameter_once(monkeypatch):
    monkeypatch.setattr(fast_distributed, "_FP32_BUCKET_ELEMENTS", 4)
    main = [nn.Parameter(torch.randn(3)) for _ in range(4)]
    reference = [nn.Parameter(parameter.detach().clone()) for parameter in main]
    options = dict(lr=4e-5, betas=(0.9, 0.95), weight_decay=0.01)
    optimizer = HostMomentAdamW([{"params": main[:2]}, {"params": main[2:]}], hardware_profile="B200", **options)
    baseline = torch.optim.AdamW([{"params": reference[:2]}, {"params": reference[2:]}], **options)
    for _ in range(3):
        for actual, expected in zip(main[1:], reference[1:]):
            actual.grad = torch.randn_like(actual)
            expected.grad = actual.grad.clone()
        optimizer.step()
        baseline.step()
    for actual, expected in zip(main, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(state["step"].item() == 3 for state in optimizer.state.values())


def test_bucketed_collectives_preserve_order_and_missing_gradients(monkeypatch):
    monkeypatch.setattr(fast_distributed, "_FP32_BUCKET_ELEMENTS", 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    model = nn.ParameterList([nn.Parameter(torch.full((3,), float(i))) for i in range(3)])
    sent = []

    def broadcast(bucket, *, src):
        assert src == 0 and bucket.dtype == torch.float32 and bucket.numel() == 3
        sent.append(bucket.clone())

    def reduce(bucket, *, dst, op, group):
        assert dst == 0 and op == torch.distributed.ReduceOp.SUM
        bucket.add_(torch.ones_like(bucket))

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.distributed, "reduce", reduce)
    fast_distributed.broadcast_trainable_parameters(model)
    for actual, reference in zip(sent, model, strict=True):
        torch.testing.assert_close(actual, reference)
    model[1].grad = torch.full((3,), 3.0)
    fast_distributed.reduce_trainable_gradients(model)
    for parameter, expected in zip(model, (1.0, 4.0, 1.0)):
        torch.testing.assert_close(parameter.grad, torch.full((3,), expected))


@pytest.mark.parametrize("add_inputs", [True, False])
def test_vllm_split_deltas_reach_alternating_gate_and_up_channels(add_inputs):
    base = torch.full((2, 6), 10.0)
    split_delta = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    layer = SimpleNamespace(moe_config=SimpleNamespace(activation="swigluoai"))
    context = SimpleNamespace(w13_num_slices=2)

    def project(self, lora_context, *, y, add_inputs):
        assert add_inputs
        y.add_(split_delta)
        return "routing metadata"

    apply = _interleave_legacy_w13(project, "swigluoai")
    result = apply(layer, lora_context=context, y=base, add_inputs=add_inputs)
    expected = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    torch.testing.assert_close(base, (expected + (10.0 if add_inputs else 0.0)).expand(2, 6))
    assert result == "routing metadata"
