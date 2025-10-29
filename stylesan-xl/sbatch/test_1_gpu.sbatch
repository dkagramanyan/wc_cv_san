#!/bin/bash
#SBATCH --job-name=GAN
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-1:0
#SBATCH --partition=gpu-ef-quick
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate san

python train.py --outdir=./training-runs/ffhq \
        --cfg=stylegan3-r \
        --data=./data/ffhq16.zip \
        --gpus=1 \
        --batch=2048 \
        --mirror=1 \
        --snap 10 \
        --batch-gpu 8 \
        --kimg 300000 

