#!/bin/bash
#SBATCH --job-name=adaptive_separate_updated
#SBATCH --output=my_output_%j.out
#SBATCH --error=my_error_%j.err
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load CUDA/11.8.0

EXECUTABLE=./bin/adaptive_paper
REPORT_NAME=$(basename "$EXECUTABLE")

nsys profile -o "$REPORT_NAME" "$EXECUTABLE"
