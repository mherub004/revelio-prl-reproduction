#!/bin/bash -l

#SBATCH --job-name=sd15_upft1_t100_cudafix
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/rlvl/rlvl173v/revelio/logs/sd15_upft1_t100_cudafix_%j.out
#SBATCH --error=/home/woody/rlvl/rlvl173v/revelio/logs/sd15_upft1_t100_cudafix_%j.err

set -e

echo "=================================================="
echo "Revelio Diff-C experiment"
echo "Model: Stable Diffusion 1.5"
echo "Dataset: Oxford-IIIT Pet"
echo "Layer: up_ft:1"
echo "Diffusion timestep: 100"
echo "Epochs: 30"
echo "Batch size: 4"
echo "Job ID: ${SLURM_JOB_ID:-not_available}"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

module load python/3.12-conda
conda activate revelio

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export WANDB_MODE=offline

# IMPORTANT:
# Do NOT use expandable_segments=True here.
# It caused the PyTorch CUDA allocator internal assert.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

mkdir -p /home/woody/rlvl/rlvl173v/revelio/logs

cd "$HOME/revelio/diffc_image_classification"

python train.py \
  --dataset_flag "timm/oxford-iiit-pet" \
  --output_dir "./outputs/sd15_upft1_t100_e30_cudafix" \
  --model_name "stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --diffusion_timestep 100 \
  --diffusion_layer "up_ft:1" \
  --learning_rate 1e-4 \
  --num_epochs 30 \
  --batch_size 4 \
  --num_classes 37 \
  --prompt_type "empty" \
  --pooling_strategy "GAP" \
  --dropout 0.0

echo "=================================================="
echo "Training completed"
echo "End time: $(date)"
echo "=================================================="
