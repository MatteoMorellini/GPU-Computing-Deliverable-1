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

# -----------------------------------------------------------------------------
# Reproducing the results reported in report.pdf
# -----------------------------------------------------------------------------
# Prerequisites (run once from the project root, on the login node):
#   1. Place the ten SuiteSparse matrices listed in matrices.txt into
#      ./matrices/ using exactly these filenames (already present in this
#      repository):
#        ASIC_680ks.mtx  FullChip.mtx  Rucci1.mtx  Si41Ge41H72.mtx
#        bone010.mtx     boyd2.mtx     eu-2005.mtx ldoor.mtx
#        rajat31.mtx     webbase-1M.mtx
#   2. Build every GPU binary used by this script:
#        module load CUDA/11.8.0
#        make gpu          # builds scalar, vector, adaptive, adaptive_paper,
#                          # partial, cusparse into ./bin/
#   3. Submit this script with sbatch on a node providing an NVIDIA A30:
#        sbatch GPU_run.sh
#
# Each binary iterates internally over every .mtx file in ./matrices/,
# performs 5 warmup runs + 100 timed runs per matrix using CUDA events,
# validates against the sequential CPU CSR baseline (tol 1e-3), and prints
# GFLOP/s = 2 * nnz / t. The numbers in report.pdf Table / geo-mean column
# are obtained by collecting the stdout produced below.
#
# Companion scripts:
#   - CPU_run.sh       : runs the single-core CPU baseline (./bin/program,
#                        built via `make cpu`) used for correctness checks.
#   - nsys_GPU_run.sh  : Nsight Systems timeline used for the overlap
#                        analysis of csr_partial_overlap.
#   - ncu_GPU_run.sh   : Nsight Compute section dump (SpeedOfLight,
#                        MemoryWorkloadAnalysis, WarpStateStats) used for
#                        the per-kernel roofline / stall discussion and for
#                        the csr_longrow_kernel ablation on FullChip/boyd2.
# -----------------------------------------------------------------------------

./bin/cusparse          # cuSPARSE baseline (Table: 96.4 GFLOP/s geo-mean)
./bin/scalar            # CSR-Scalar           (Table: 22.8 GFLOP/s geo-mean)
./bin/vector            # CSR-Vector           (Table: 22.8 GFLOP/s geo-mean)
./bin/adaptive_paper    # CSR-Adaptive         (Table: 59.4 GFLOP/s geo-mean)
./bin/partial           # CSR-Partial-Overlap  (Table: 40.0 GFLOP/s geo-mean)

# For the long-row ablation in Section "CSR-Adaptive Improvements"
# (18x speedup on FullChip, 8.4x on boyd2) build and run the improved
# variant that adds csr_longrow_kernel:
#   make adaptive
#   ./bin/adaptive

