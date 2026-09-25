#!/bin/bash -l
#SBATCH --job-name=sd15_bottleneck_t25
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/rlvl/rlvl173v/revelio/logs/sd15_bottleneck_t25_%j.out
#SBATCH --error=/home/woody/rlvl/rlvl173v/revelio/logs/sd15_bottleneck_t25_%j.err

module load python/3.12-conda
conda activate revelio

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /home/woody/rlvl/rlvl173v/revelio/diffc_image_classification

python train.py   --dataset_flag "timm/oxford-iiit-pet"   --output_dir "./outputs/sd15_bottleneck_t25"   --model_name "stable-diffusion-v1-5/stable-diffusion-v1-5"   --diffusion_timestep 25   --diffusion_layer "bottleneck"   --learning_rate 1e-4   --num_epochs 90   --batch_size 4   --num_classes 37   --prompt_type "empty"   --pooling_strategy "GAP"   --dropout 0.0
