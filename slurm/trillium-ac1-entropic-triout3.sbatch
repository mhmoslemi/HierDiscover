#!/bin/bash
#SBATCH --job-name=ac1-entropic-triout3
#SBATCH --account=def-boyuwang
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=h200:1
#SBATCH --chdir=/home/mmoslem3/links/scratch/HierDiscover
#SBATCH --output=/home/mmoslem3/links/scratch/HierDiscover/slurm-%x-%j.out

set -euo pipefail
source /home/mmoslem3/links/scratch/unsloth_env/bin/activate

export TTT_CONFIG=configs/ac1.yaml
export TRIOUT=3
export PYTHONUNBUFFERED=1

# CUDA exposes the H200 assigned by Slurm as device 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export AVAILABLE_GPUS="${CUDA_VISIBLE_DEVICES:?Slurm did not assign GPUs}"

model_path=/home/mmoslem3/links/scratch/models/gpt-oss-120b
exec sh ./run.sh --hardware-profile H200 --advantage-mode entropic \
    --coder-model-name "$model_path" "$@"
