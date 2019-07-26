#!/bin/bash 
#SBATCH --job-name=Compute_SEx_Sizes
#SBATCH --time=9:59:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=cbottrel@uvic.ca

source ~/venv/tf36-cpu/bin/activate
/home/bottrell/projects/def-simardl/bottrell/Subaru/HyperSuprime/TidalCNN/SExtractor_Sizes.py
