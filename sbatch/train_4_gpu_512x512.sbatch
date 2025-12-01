#!/bin/bash
#SBATCH --job-name=SAN-GAN-512
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=4-0:0
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
        --data=./datasets/imagenet_9to4_1024x1024_512x512.zip \
        --gpus=4 \
        --batch=40 \
        --mirror=0 \
        --snap 100 \
        --batch-gpu 10 \
        --kimg 20000 \
        --cond True \
        --syn_layers 6 \
        --superres \
        --up_factor 2 \
        --head_layers 7 \
        --path_stem ./runs/wc-cv/00016-stylegan3-r-imagenet_9to4_1024x1024_512x512-gpus4-batch40/best_model.pkl

