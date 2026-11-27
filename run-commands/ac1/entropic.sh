#!/bin/sh
set -eu

# Resolve project paths from the repository root.
CDPATH= cd -- "$(dirname -- "$0")/../.."
config_path="${TTT_CONFIG:-configs/ac1.yaml}"
export AVAILABLE_GPUS=5
export TTT_LAUNCH_ADVANTAGE_DEFAULT=entropic
export TRIOUT=3  # Directory label only.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Keep persistent caches in the checkout and use shared memory for temporary work.
ttt_runtime_dir="$PWD/.runtime"
TTT_FAST_RUNTIME_DIR="$ttt_runtime_dir/fast"
if [ -d /dev/shm ] && [ -w /dev/shm ]; then
    TTT_FAST_RUNTIME_DIR=/dev/shm/ttt-small-runtime
fi
export TTT_FAST_RUNTIME_DIR
export TMPDIR="$TTT_FAST_RUNTIME_DIR/tmp"
export XDG_CACHE_HOME="$ttt_runtime_dir/cache"
export XDG_CONFIG_HOME="$ttt_runtime_dir/config"
export XDG_DATA_HOME="$ttt_runtime_dir/data"
export XDG_STATE_HOME="$ttt_runtime_dir/state"
export HF_HOME="$ttt_runtime_dir/cache/huggingface"
export VLLM_CACHE_ROOT="$TTT_FAST_RUNTIME_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TTT_FAST_RUNTIME_DIR/torchinductor-vllm"
export TRITON_CACHE_DIR="$TTT_FAST_RUNTIME_DIR/triton-vllm"
export TORCH_EXTENSIONS_DIR="$TTT_FAST_RUNTIME_DIR/torch_extensions"
export CUDA_CACHE_PATH="$TTT_FAST_RUNTIME_DIR/cuda"
export FLASHINFER_WORKSPACE_BASE="$TTT_FAST_RUNTIME_DIR/flashinfer"
export MPLCONFIGDIR="$ttt_runtime_dir/cache/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
    "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" \
    "$FLASHINFER_WORKSPACE_BASE" "$MPLCONFIGDIR"

exec python3 runtime/run_supervisor.py -- python3 orchestrator.py \
    --config "$config_path" --fused-long-attention "$@"
