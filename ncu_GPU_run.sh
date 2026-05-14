#!/bin/bash
#SBATCH --job-name=GPU_deliverable-1
#SBATCH --output=my_output_%j.out
#SBATCH --error=my_error_%j.err
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load CUDA/11.8.0

mkdir -p "$HOME/tmp/ncu"
export TMPDIR="$HOME/tmp/ncu"
ncu \
  --kernel-name regex:csr_partial_overlap_kernel \
  --launch-skip 950 \
  --launch-count 1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  ./bin/partial 