"""Shared expert and output-head LoRA support for the GPT-OSS vLLM worker."""

from functools import wraps


def _interleave_legacy_w13(original, interleaved_activation):
    import torch

    @wraps(original)
    def apply(self, lora_context, *, y, add_inputs=True, **kwargs):
        if (lora_context.w13_num_slices != 2
                or self.moe_config.activation != interleaved_activation):
            return original(self, lora_context, y=y, add_inputs=add_inputs, **kwargs)

        # Punica already skips its projection for a base-policy batch.
        # Keep its routing metadata without zeroing and adding a delta.
        punica = getattr(lora_context, "punica_wrapper", None)
        metadata = getattr(punica, "token_mapping_meta", None)
        no_lora = getattr(metadata, "no_lora_flag_cpu", None)
        if add_inputs and no_lora is not None and no_lora.item():
            return original(self, lora_context, y=y, add_inputs=True, **kwargs)

        # vLLM #60812: the split kernel emits [all gate; all up] while
        # GPT-OSS SwiGLU consumes alternating gate/up columns.
        delta = torch.zeros_like(y)
        result = original(self, lora_context, y=delta, add_inputs=True, **kwargs)
        destination = y.unflatten(-1, (-1, 2)).transpose(-1, -2)
        source = delta.unflatten(-1, (2, -1))
        if add_inputs:
            destination.add_(source)
        else:
            destination.copy_(source)
        return result

    apply._ttt_interleaves_w13 = True
    return apply


def install_gpt_oss_lora_support():
    from packaging.version import Version
    from vllm import __version__
    from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

    # The LoRA manager discovers output heads through this mapping.
    GptOssForCausalLM.embedding_modules = {
        **GptOssForCausalLM.embedding_modules, "lm_head": "output_embeddings"}

    if Version(__version__) < Version("0.32.0"):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import LoRAExpertsMixin

        original = LoRAExpertsMixin.apply_w13_lora
        if (not getattr(original, "_ttt_interleaves_w13", False)
                and "interleave_w13" not in original.__code__.co_varnames):
            LoRAExpertsMixin.apply_w13_lora = _interleave_legacy_w13(original, MoEActivation.SWIGLUOAI)
