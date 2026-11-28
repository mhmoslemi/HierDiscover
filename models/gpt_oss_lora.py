"""Rank-32 shared-outer expert LoRA on Unsloth's packed GPT-OSS forward."""

from contextlib import contextmanager
from functools import cache
from pathlib import Path
import sys
from types import FunctionType, MethodType, ModuleType

import torch
from torch import nn
from torch.nn import functional as F


ATTENTION_AND_HEAD_TARGETS = r"(?:.*\.(?:q_proj|k_proj|v_proj|o_proj)|lm_head)"
PARAMETERS_PER_RANK = 41_074_624


def _gate_up_views(tensor, interleaved):
    if interleaved:
        return tensor[..., ::2], tensor[..., 1::2]
    return tensor.chunk(2, dim=-1)


class _AddGateUpDelta(torch.autograd.Function):
    """Add the two expert deltas without materializing the combined delta."""

    @staticmethod
    def forward(ctx, base, gate, up, scale, interleaved):
        ctx.scale = scale
        ctx.interleaved = interleaved
        result = torch.empty_like(base)
        base_gate, base_up = _gate_up_views(base, interleaved)
        result_gate, result_up = _gate_up_views(result, interleaved)
        torch.add(base_gate, gate, alpha=scale, out=result_gate)
        torch.add(base_up, up, alpha=scale, out=result_up)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = _gate_up_views(grad_output, ctx.interleaved)
        # Native grouped GEMM needs aligned gradients. In the interleaved
        # layout the up view starts one element into grad_output.
        gate = gate.contiguous()
        up = up.contiguous()
        if ctx.scale != 1.0:
            gate = gate * ctx.scale
            up = up * ctx.scale
        return grad_output, gate, up, None, None


class SharedExpertLoRA(nn.Module):
    def __init__(self, experts, hidden, intermediate, rank, alpha, dropout, device):
        super().__init__()
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(float(dropout))
        self.enabled = True

        def parameter(*shape, zero=False):
            value = torch.empty(shape, device=device, dtype=torch.float32)
            if zero:
                nn.init.zeros_(value)
            else:
                bound = shape[-1] ** -0.5
                nn.init.uniform_(value, -bound, bound)
            return nn.Parameter(value)

        self.gate_a = parameter(1, rank, hidden)
        self.gate_b = parameter(experts, intermediate, rank, zero=True)
        self.up_a = parameter(1, rank, hidden)
        self.up_b = parameter(experts, intermediate, rank, zero=True)
        self.down_a = parameter(experts, rank, intermediate)
        self.down_b = parameter(1, hidden, rank, zero=True)

    def project(self, base, inputs, offsets, grouped_mm, projection, interleaved):
        if not self.enabled:
            return base
        inputs = inputs.to(base.dtype)
        if projection == "down":
            low_rank = grouped_mm(
                self.dropout(inputs), self.down_a.to(inputs.dtype).transpose(-1, -2), offsets)
            return torch.addmm(base, low_rank, self.down_b[0].to(inputs.dtype).T, alpha=self.scale)

        if self.dropout.training and self.dropout.p:
            gate = F.linear(self.dropout(inputs), self.gate_a[0].to(inputs.dtype))
            up = F.linear(self.dropout(inputs), self.up_a[0].to(inputs.dtype))
        else:
            # One dense projection also avoids two full input gradients
            # that would otherwise be summed during backward.
            gate_up_a = torch.cat((self.gate_a[0], self.up_a[0])).to(inputs.dtype)
            gate, up = F.linear(inputs, gate_up_a).chunk(2, dim=-1)
        gate = grouped_mm(gate, self.gate_b.to(inputs.dtype).transpose(-1, -2), offsets)
        up = grouped_mm(up, self.up_b.to(inputs.dtype).transpose(-1, -2), offsets)
        return _AddGateUpDelta.apply(base, gate, up, self.scale, interleaved)

    def export_tensors(self):
        return {
            "w1.lora_A.weight": self.gate_a,
            "w1.lora_B.weight": self.gate_b,
            "w2.lora_A.weight": self.down_a,
            "w2.lora_B.weight": self.down_b,
            "w3.lora_A.weight": self.up_a,
            "w3.lora_B.weight": self.up_b,
        }


@cache
def _with_shared_projections(forward):
    # Bind only this model's expert instances to the two projection hooks.
    # The vendor's routing, dequantization, activation, and combine stay in its forward.
    name = f"{__name__}_{forward.__module__.replace('.', '_')}_{forward.__name__}"
    patched_module = ModuleType(name)
    globals_with_lora = vars(patched_module)
    globals_with_lora.update(forward.__globals__, __name__=name)
    try:
        layout_for = globals_with_lora["_mxfp4_expert_layout"]
        base_projection = globals_with_lora["_mxfp4_base_grouped_mm"]
        grouped_mm = globals_with_lora["_grouped_mm_with_backward_fix"]
        is_interleaved = globals_with_lora["_gate_up_is_interleaved"]
    except KeyError as error:
        raise RuntimeError("Unsloth Zoo lacks GPT-OSS's packed grouped-projection hooks") from error
    if not {"_mxfp4_expert_layout", "_mxfp4_base_grouped_mm"} <= set(forward.__code__.co_names):
        raise RuntimeError("Unsloth Zoo's grouped expert forward no longer calls its packed projection hooks")

    def layout_with_adapter(source, projection, hidden_dim, model_type, experts_module):
        layout = layout_for(source, projection, hidden_dim, model_type, experts_module)
        if layout is None:
            raise RuntimeError(f"GPT-OSS's {projection} no longer has a packed MXFP4 projection layout")
        return layout, experts_module.ttt_expert_adapter, projection, is_interleaved(experts_module)

    def project_with_adapter(inputs, offsets, counts, weight_provider, recompute, layout):
        packed_layout, adapter, projection, interleaved = layout
        base = base_projection(inputs, offsets, counts, weight_provider, recompute, packed_layout)
        return adapter.project(base, inputs, offsets, grouped_mm, projection, interleaved)

    globals_with_lora.update(
        _mxfp4_expert_layout=layout_with_adapter, _mxfp4_base_grouped_mm=project_with_adapter)
    # Dynamo resolves function globals by their importable module name.
    sys.modules[name] = patched_module
    patched = FunctionType(forward.__code__, globals_with_lora, forward.__name__,
                           forward.__defaults__, forward.__closure__)
    patched.__kwdefaults__ = forward.__kwdefaults__
    return patched


def _chunked_expert_forward(forward, chunk_tokens):
    from torch.utils.checkpoint import checkpoint

    def routed_forward(experts, hidden_states, router_indices=None, routing_weights=None):
        hidden_size = hidden_states.shape[-1]
        if hidden_states.numel() <= chunk_tokens * hidden_size:
            return forward(experts, hidden_states, router_indices, routing_weights)

        hidden = hidden_states.reshape(-1, hidden_size)
        indices = router_indices.reshape(len(hidden), -1)
        weights = routing_weights.reshape(len(hidden), -1)

        def run_chunk(states, routes, scores):
            return forward(experts, states, routes, scores)

        # Expert routing is per token. Recompute each chunk in backward so
        # the full layer's expanded expert activations are never live at once.
        grad_enabled = torch.is_grad_enabled()
        outputs = []
        for states, routes, scores in zip(
                hidden.split(chunk_tokens), indices.split(chunk_tokens), weights.split(chunk_tokens)):
            if grad_enabled:
                output = checkpoint(run_chunk, states, routes, scores, use_reentrant=False)
            else:
                output = run_chunk(states, routes, scores)
            outputs.append(output)
        return torch.cat(outputs, dim=0).reshape_as(hidden_states)

    return routed_forward


def install_shared_expert_lora(model, rank, alpha, dropout, *, expert_chunk_tokens=None):
    from unsloth_zoo.temporary_patches.moe_utils import forward_native_grouped_mm

    forward = _with_shared_projections(forward_native_grouped_mm)

    def routed_forward(experts, hidden_states, router_indices=None, routing_weights=None):
        return forward(experts, hidden_states, router_indices, routing_weights)

    if expert_chunk_tokens is not None:
        routed_forward = _chunked_expert_forward(forward, expert_chunk_tokens)
        print(f"[memory] GPT-OSS packed experts: {expert_chunk_tokens:,} tokens per chunk",
              flush=True)

    config = model.config
    expert_modules = [module for name, module in model.named_modules() if name.endswith(".experts")]
    if len(expert_modules) != config.num_hidden_layers:
        raise RuntimeError("GPT-OSS did not expose one packed expert module per decoder layer")
    for module in expert_modules:
        module.ttt_expert_adapter = SharedExpertLoRA(
            config.num_local_experts, config.hidden_size, config.intermediate_size,
            rank, alpha, dropout, module.gate_up_proj.device)
        module.forward = MethodType(routed_forward, module)

    expected = PARAMETERS_PER_RANK * rank
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if actual != expected:
        raise RuntimeError(f"GPT-OSS shared LoRA expected {expected:,} trainable weights, found {actual:,}")
    model._ttt_gpt_oss_shared_lora = True
    print(f"[backend=unsloth] GPT-OSS shared expert and head LoRA: {actual:,} trainable weights", flush=True)


@contextmanager
def disable_shared_adapter(model):
    adapters = [module for module in model.modules() if isinstance(module, SharedExpertLoRA)]
    enabled = [adapter.enabled for adapter in adapters]
    try:
        for adapter in adapters:
            adapter.enabled = False
        with model.disable_adapter():
            yield
    finally:
        for adapter, was_enabled in zip(adapters, enabled):
            adapter.enabled = was_enabled


def _expert_tensors(model):
    return {
        f"{name}.{suffix}": tensor
        for name, module in model.named_modules()
        if (adapter := getattr(module, "ttt_expert_adapter", None)) is not None
        for suffix, tensor in adapter.export_tensors().items()
    }


def save_shared_adapter(model, directory, *, base_policy=False):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = get_peft_model_state_dict(model, save_embedding_layers=False)
    weights.update(_expert_tensors(model))
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in weights.items()}
    target = directory / "adapter_model.safetensors"
    temp = target.with_suffix(".safetensors.tmp")
    metadata = {"format": "pt"}
    if base_policy:
        metadata["ttt_base_policy"] = "true"
    try:
        save_file(weights, str(temp), metadata=metadata)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    model.peft_config[model.active_adapter].save_pretrained(directory)


def is_base_policy_snapshot(directory):
    from safetensors import safe_open

    directory = Path(directory)
    path = directory / "adapter_model.safetensors"
    if not path.is_file():
        return False
    with safe_open(str(path), framework="pt", device="cpu") as saved:
        if (saved.metadata() or {}).get("ttt_base_policy") == "true":
            return True
        if directory.name != "adapter_initial":
            return False
        # Runs started before the marker was added can still skip LoRA if
        # their saved initial policy consists entirely of zero-output LoRAs.
        keys = set(saved.keys())
        b_keys = {key for key in keys if key.endswith(".lora_B.weight")}
        a_keys = {key.removesuffix(".lora_B.weight") + ".lora_A.weight" for key in b_keys}
        return bool(b_keys and keys == a_keys | b_keys and all(
            not torch.count_nonzero(saved.get_tensor(key)).item() for key in b_keys))


@torch.no_grad()
def load_shared_adapter(model, directory):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors import safe_open

    ordinary = get_peft_model_state_dict(model, save_embedding_layers=False)
    expert = _expert_tensors(model)
    expected = ordinary | expert
    with safe_open(str(Path(directory) / "adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        keys = set(saved.keys())
        if keys != set(expected) or any(
                saved.get_slice(key).get_shape() != list(expected[key].shape) for key in keys):
            raise ValueError(f"Incompatible GPT-OSS shared LoRA adapter in {directory}")
        set_peft_model_state_dict(model, {key: saved.get_tensor(key) for key in ordinary})
        for key, parameter in expert.items():
            parameter.copy_(saved.get_tensor(key))
