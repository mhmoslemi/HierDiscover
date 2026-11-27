#!/bin/bash
set -euo pipefail

objective=$1
shift
export TTT_NIBI_MODEL_SOURCE=/home/mmoslem3/scratch/models/gpt-oss-120b
export TTT_NIBI_TRAINER_MODEL="${SLURM_TMPDIR:?}/gpt-oss-120b"
export TTT_VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# A retry on another node can reuse the compiled engine. Concurrent copies
# of the same job use separate caches while vLLM writes its compiler index.
cache_dir="$PWD/.runtime/nibi-h100/${SLURM_JOB_NAME:?}"
mkdir -p "$cache_dir"
exec 9>"$cache_dir/engine.lock"
if flock -n 9; then
    export TTT_NIBI_VLLM_CACHE="$cache_dir"
else
    export TTT_NIBI_VLLM_CACHE="$SLURM_TMPDIR/vllm-cache"
fi

# Compile the Hopper TP communication kernel while the checkpoint is being
# copied and the trainers start. FlashInfer normally builds it on first use.
# The build uses nvcc on the CPU and the same cache as the generation workers.
FLASHINFER_CUDA_ARCH_LIST=9.0a CUDA_VISIBLE_DEVICES="" \
    FLASHINFER_WORKSPACE_BASE="$TTT_NIBI_VLLM_CACHE/flashinfer" \
    python3 - <<'PY' &
from flashinfer.jit.comm import gen_trtllm_mnnvl_comm_module

kernel = gen_trtllm_mnnvl_comm_module()
if not kernel.is_compiled:
    kernel.build(need_lock=True)
PY

# Both trainer ranks mmap the same local copy. Run artifacts keep the
# permanent base-model path, which vLLM also uses for its compiler cache key.
echo "Staging GPT-OSS trainer weights on the node SSD"
mkdir -p "$TTT_NIBI_TRAINER_MODEL"
rsync -aL --whole-file "$TTT_NIBI_MODEL_SOURCE/" "$TTT_NIBI_TRAINER_MODEL/"
echo "Trainer weights ready after ${SECONDS}s"
exec sh ./run.sh --hardware-profile H100 --advantage-mode "$objective" \
    --coder-model-name "$TTT_NIBI_MODEL_SOURCE" "$@"
