#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=int_ring
#SBATCH --mail-type=END
#SBATCH --mail-user=jg6848@nyu.edu
#SBATCH --output=slurm_int_ring_%j.out

module purge
module load openmpi/gcc/3.1.4
mpirun ./int_ring 1 100000
