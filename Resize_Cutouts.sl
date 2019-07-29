#!/bin/bash 
#SBATCH --job-name=Resize_HSC_Cutouts
#SBATCH --time=2:59:00
#SBATCH --mem-per-cpu=256M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END
#SBATCH --mail-user=cbottrel@uvic.ca

source ~/venv/tf36-cpu/bin/activate
~/scratch/Subaru/HyperSuprime/Resize_Cutouts.py
