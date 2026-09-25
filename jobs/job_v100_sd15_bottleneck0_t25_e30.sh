#!/bin/bash -l

#SBATCH --job-name=v100_sd15_bottleneck0_t25
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/rlvl/rlvl173v/revelio/logs/v100_sd15_bottleneck0_t25_%j.out
#SBATCH --error=/home/woody/rlvl/rlvl173v/revelio/logs/v100_sd15_bottleneck0_t25_%j.err

set -e

echo "=================================================="
echo "Revelio Diff-C experiment"
echo "Model: Stable Diffusion 1.5"
echo "Dataset: Oxford-IIIT Pet"
echo "Layer: bottleneck:0"
echo "Diffusion timestep: 25"
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

unset PYTORCH_CUDA_ALLOC_CONF
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

mkdir -p /home/woody/rlvl/rlvl173v/revelio/logs
cd "$HOME/revelio/diffc_image_classification"

rm -rf "./outputs/v100_sd15_bottleneck0_t25_${SLURM_JOB_ID}"

python train.py \
  --dataset_flag "timm/oxford-iiit-pet" \
  --output_dir "./outputs/v100_sd15_bottleneck0_t25_${SLURM_JOB_ID}" \
  --model_name "stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --diffusion_timestep 25 \
  --diffusion_layer "bottleneck:0" \
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
