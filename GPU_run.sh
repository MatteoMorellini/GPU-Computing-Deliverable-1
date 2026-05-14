#!/bin/bash
#SBATCH --job-name=GPU_deliverable-1
#SBATCH --output=my_output_%j.out
#SBATCH --error=my_error_%j.err
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load CUDA/11.8.0

./bin/cusparse
./bin/partial
./bin/vector
./bin/adaptive