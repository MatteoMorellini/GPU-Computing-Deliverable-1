#!/bin/bash
#SBATCH --job-name=
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
mkdir -p results/ncu

LAUNCH_SKIP=5
LAUNCH_COUNT=1

FORMATS=(vector adaptive partial cusparse adaptive_paper scalar)

for bin in "${FORMATS[@]}"; do
  [[ -x "./bin/${bin}" ]] || { echo "skip ${bin}: binary missing"; continue; }

  echo "=== profiling ${bin} ==="
  ncu \
    --launch-skip ${LAUNCH_SKIP} \
    --launch-count ${LAUNCH_COUNT} \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStats \
    --export "results/ncu/${bin}" \
    --force-overwrite \
    "./bin/${bin}"

  ncu --import "results/ncu/${bin}.ncu-rep" \
      > "results/ncu/${bin}.txt"
done
