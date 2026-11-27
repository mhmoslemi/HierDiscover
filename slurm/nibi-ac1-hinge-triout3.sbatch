#!/bin/bash
#SBATCH --job-name=ac1-hinge-triout3
#SBATCH --account=def-boyuwang
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=h100:2
#SBATCH --mem=512G
#SBATCH --chdir=/home/mmoslem3/scratch/HierDiscover
#SBATCH --output=/home/mmoslem3/scratch/HierDiscover/slurm-%x-%j.out

set -euo pipefail
source /home/mmoslem3/scratch/unsloth_env/bin/activate

export TTT_CONFIG=configs/ac1.yaml
export TRIOUT=3
export PYTHONUNBUFFERED=1

# CUDA numbers the two GPUs assigned by Slurm as 0 and 1 inside the job.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export AVAILABLE_GPUS="${CUDA_VISIBLE_DEVICES:?Slurm did not assign GPUs}"

model_path=/home/mmoslem3/scratch/models/gpt-oss-120b
exec sh ./run.sh --hardware-profile H100 --advantage-mode hinge \
    --coder-model-name "$model_path" "$@"
