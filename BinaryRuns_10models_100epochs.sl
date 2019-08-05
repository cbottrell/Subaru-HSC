#!/bin/bash 
#SBATCH --job-name=Binary_x10Models
#SBATCH --time=2:59:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=cbottrel@uvic.ca

source ~/venv/tf36-gpu/bin/activate
~/scratch/Subaru/HyperSuprime/BinaryRuns_10models_100epochs.py
