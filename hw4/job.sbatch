#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=pingpong
#SBATCH --mail-type=END
#SBATCH --mail-user=jg6848@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load openmpi/gcc/3.1.4
mpirun ./pingpong 0 1
