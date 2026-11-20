#!/bin/sh
set -eu

# Always run the code/configs beside this launcher, even when invoked from
# another checkout. Updating files cannot update an already-running Python.
ttt_project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ttt_project_dir"

# Persistent files stay on NAS; compilation and IPC use RAM, never the 49 GB disk.
ttt_runtime_dir="$ttt_project_dir/.runtime"
if [ -d /dev/shm ] && [ -w /dev/shm ]; then
    ttt_fast_runtime_dir="/dev/shm/ttt-small-runtime"
else
    ttt_fast_runtime_dir="$ttt_runtime_dir/fast"
fi
export TTT_FAST_RUNTIME_DIR="$ttt_fast_runtime_dir"
export TMPDIR="$ttt_fast_runtime_dir/tmp"
export XDG_CACHE_HOME="$ttt_runtime_dir/cache"
export XDG_CONFIG_HOME="$ttt_runtime_dir/config"
export XDG_DATA_HOME="$ttt_runtime_dir/data"
export XDG_STATE_HOME="$ttt_runtime_dir/state"
export HF_HOME="$ttt_runtime_dir/cache/huggingface"
export VLLM_CACHE_ROOT="$ttt_fast_runtime_dir/vllm"
export TORCHINDUCTOR_CACHE_DIR="$ttt_fast_runtime_dir/torchinductor-vllm"
export TRITON_CACHE_DIR="$ttt_fast_runtime_dir/triton-vllm"
export TORCH_EXTENSIONS_DIR="$ttt_fast_runtime_dir/torch_extensions"
export CUDA_CACHE_PATH="$ttt_fast_runtime_dir/cuda"
export FLASHINFER_WORKSPACE_BASE="$ttt_fast_runtime_dir/flashinfer"
export MPLCONFIGDIR="$ttt_runtime_dir/cache/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
    "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" \
    "$FLASHINFER_WORKSPACE_BASE" "$MPLCONFIGDIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export AVAILABLE_GPUS="${AVAILABLE_GPUS:-6,7}"
# config_path="${TTT_CONFIG:-configs/circle_packing.yaml}"
config_path="${TTT_CONFIG:-configs/ac1.yaml}"
# config_path="${TTT_CONFIG:-configs/erdos-c4.yaml}"




# Runtime-only environment belongs here. Models, GPU roles,
# sampling, and optimization hyperparameters belong in the selected YAML.
# Authoritative ordered physical GPU inventory. Edit this one list (or export
# it before invoking the script); Python derives every role from it. Every
# rollout card hosts a parallel trainer replica and then rejoins generation.
# GPU-mode reserves only the last card for evaluation when a separate card
# exists. vLLM derives compatible TP groups/replicas that consume the remaining
# complete list without dropping a card.
# FlashInfer sampling can trigger runtime compilation and require nvcc. vLLM's
# native PyTorch/Triton sampler is the safe default; callers may explicitly opt
# back in with VLLM_USE_FLASHINFER_SAMPLER=1.
# Override without editing this launcher:
#   TTT_CONFIG=configs/gpu_mode_trimul.yaml sh run.sh
# Resume keeps saved experiment settings; AVAILABLE_GPUS still defines this
# launch's physical inventory:
#   sh run.sh --resume /path/to/run
# Training uses one process per GPU with adaptive batching.
# Use TTT's training loss with hybrid advantages:
#   sh run.sh --advantage-mode hybrid
# Set asymmetric SPO-RS clipping as distances below/above 1, for example:
#   sh run.sh --spo-rs-clip-epsilon-low 0.1 --spo-rs-clip-epsilon-high 0.3
# Isolate every candidate on one CPU. In this mode reward_workers from the
# selected YAML is the number of concurrent candidate processes per CPU:
#   sh run.sh --isolate-eval
# Skip adapter training completely while keeping rollout, evaluation, search,
# search and result/checkpoint persistence:
#   sh run.sh --no-train
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

TTT_LAUNCH_ADVANTAGE_DEFAULT=spo-rs \
exec python3 runtime/run_supervisor.py -- python3 orchestrator.py --config "$config_path" \
    --spo-rs-clip-epsilon-low 0.2 --spo-rs-clip-epsilon-high 0.38 \
    --isolate-eval --fused-long-attention "$@"
    # 
    # 
    # 





# TTT_LAUNCH_ADVANTAGE_DEFAULT=spo-rs \
# exec python3 runtime/run_supervisor.py -- python3 orchestrator.py --config "$config_path" \
#     --spo-rs-clip-epsilon-low 0.2 --spo-rs-clip-epsilon-high 0.38 \
#     --isolate-eval --fused-long-attention "$@"
#     # 
#     # 
#     # 

