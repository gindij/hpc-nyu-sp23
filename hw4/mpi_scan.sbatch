#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=mpi_scan
#SBATCH --mail-type=END
#SBATCH --mail-user=jg6848@nyu.edu
#SBATCH --output=slurm_mpi_scan_%j.out

module purge
module load openmpi/gcc/3.1.4
mpirun ./mpi_scan
