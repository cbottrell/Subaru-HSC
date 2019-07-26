#!/bin/bash 
#SBATCH --job-name=Updates
#SBATCH --time=12:00:00
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=cbottrel@uvic.ca

tar -xzf Images.tar.gz
