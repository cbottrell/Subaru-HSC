#!/bin/bash 
#SBATCH --job-name=Generate_BinaryInput
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=16000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END
#SBATCH --mail-user=cbottrel@uvic.ca

source ~/venv/tf36-cpu/bin/activate
~/scratch/Subaru/HyperSuprime/Generate_BinaryInput.py
