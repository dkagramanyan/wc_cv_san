#!/bin/bash
#SBATCH --job-name=SAN-GAN-1024
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda

# nvidia a100 arch
export TORCH_CUDA_ARCH_LIST="8.0"
module load CUDA/12.4

module load gnu13/13.3
source activate san

python train.py --outdir=./runs/wc-cv \
        --cfg=stylegan3-r \
        --data=./datasets/imagenet_9to4_1024x1024_1024x1024.zip \
        --gpus=4 \
        --batch=192 \
        --mirror=0 \
        --snap 500 \
        --batch-gpu 96 \
        --kimg 20000 \
        --cond True \
        --syn_layers 6 \
        --superres \
        --up_factor 2 \
        --head_layers 7 \
        --path_stem ./runs/wc-cv/00018-stylegan3-r-imagenet_9to4_1024x1024_1024x1024-gpus4-batch20/best_model.pkl

