#!/bin/bash
#SBATCH --job-name=SAN-GAN-32
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate san

python train.py --outdir=./runs/wc-cv \
        --cfg=stylegan3-r \
        --data=./datasets/imagenet_9to4_1024x1024_32x32.zip \
        --gpus=4 \
        --batch=384 \
        --mirror=0 \
        --snap 100 \
        --batch-gpu 96 \
        --kimg 20000 \
        --cond True \
        --syn_layers 6 \
        --superres \
        --up_factor 2 \
        --head_layers 7 \
        --path_stem training-runs/wc-cv/00005-stylegan3-r-o_bc_left_4x_1536_1024x1024_rgb_16x16-gpus4-batch384/best_model.pkl
