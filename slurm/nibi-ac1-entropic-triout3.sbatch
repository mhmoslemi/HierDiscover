#!/bin/bash -l
#SBATCH --job-name=ac1-entropic-triout3
#SBATCH --account=def-bghaddar
#SBATCH --time=0-14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=75
#SBATCH --gpus-per-node=h100:2
#SBATCH --mem=512G
#SBATCH --chdir=/home/mmoslem3/scratch/HierDiscover
#SBATCH --output=/home/mmoslem3/scratch/HierDiscover/slurm-%x-%j.out

set -euo pipefail
# FlashInfer builds the tensor-parallel communication kernel with this toolkit.
module load cuda/13.2
source /home/mmoslem3/scratch/unsloth_env/bin/activate

export TTT_CONFIG=configs/ac1.yaml
export TRIOUT=3
export PYTHONUNBUFFERED=1

# CUDA numbers the two GPUs assigned by Slurm as 0 and 1 inside the job.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export AVAILABLE_GPUS="${CUDA_VISIBLE_DEVICES:?Slurm did not assign GPUs}"

exec bash ./slurm/nibi-ac1-start.sh entropic "$@"
