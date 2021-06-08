#!/bin/bash
# SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sauravgupta3108@gmail.com

source ../../.venv/bin/activate
python3 train.py