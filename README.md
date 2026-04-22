# SpMV Investigation — GPU Computing 2025-2026

> **Deliverable 1** · Sparse Matrix-Vector Multiplication on NVIDIA Ampere GPUs

---

## Overview

This project implements and evaluates **Sparse Matrix-Vector Multiplication (SpMV)** across multiple sparse storage formats on a single NVIDIA Ampere GPU. The study goes beyond a simple format comparison: it investigates how storage format, parallelization strategy, and memory access patterns jointly determine performance across matrices with diverse sparsity structures.

A CPU reference implementation is included for correctness validation. Results are benchmarked against [cuSPARSE](https://developer.nvidia.com/cusparse) as an industry baseline.

---

## Goals

- Implement a correct, validated **CPU baseline** (sequential and/or OpenMP)
- Develop **at least two GPU kernels**:
  - A straightforward CSR-parallel kernel
  - An optimized kernel using shared memory and improved load balancing
- Compare performance across **≥ 10 SuiteSparse matrices** spanning diverse sparsity regimes
- Report **runtime** and **GFLOP/s** for all implementations
- Explain performance differences in terms of matrix structure, memory behavior, and algorithmic technique

---

## Repository Structure

```
.
├── cpu/
│   ├── spmv_cpu.c          # Sequential CPU SpMV reference
│   └── spmv_omp.c          # Optional OpenMP baseline
├── gpu/
│   ├── spmv_csr_scalar.cu  # Naive CSR kernel (1 thread per row)
│   ├── spmv_csr_vector.cu  # CSR-vector kernel (1 warp per row)
│   └── spmv_optimized.cu   # Shared memory + load-balanced kernel
├── formats/
│   ├── csr.h               # CSR format utilities
│   ├── coo.h               # COO format utilities
│   └── ell.h               # ELL / HYB format utilities (optional)
├── io/
│   └── mmio.c / mmio.h     # Matrix Market (.mtx) file reader
├── validation/
│   └── validate.py         # Numerical correctness checker
├── matrices/
│   └── README.md           # Dataset description and download instructions
├── results/
│   ├── timings.csv         # Raw benchmark data
│   └── plots/              # Generated figures
├── report/
│   └── report.pdf          # 4-page course report
├── Makefile
└── README.md
```

---

## Dataset

Ten matrices are selected from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) to stress different access patterns:

| # | Matrix | Rows | NNZ | Avg NNZ/row | Structure |
|---|--------|------|-----|-------------|-----------|
| 1 | *(TBD)* | | | | Structured / regular |
| 2 | *(TBD)* | | | | Unstructured / irregular |
| … | … | … | … | … | … |

> Matrices are chosen to cover a range of sizes, average and variance of nonzeros per row, and degree of structural regularity, following the dataset used in [Chu et al., HPDC '23].

Input vectors are randomly generated dense Float32 vectors with a **fixed random seed** for reproducibility.

---

## Build Instructions

### Requirements

- CUDA Toolkit ≥ 11.x (Ampere-compatible)
- GCC ≥ 9 (for CPU code)
- OpenMP (optional, for CPU parallel baseline)
- cuSPARSE (bundled with CUDA Toolkit)
- `make`

### Compile

```bash
# All targets
make all

# CPU only
make cpu

# GPU kernels
make gpu

# Clean
make clean
```

---

## Usage

```bash
# Run CPU reference
./spmv_cpu <matrix.mtx>

# Run GPU kernel (CSR scalar)
./spmv_gpu --format csr --kernel scalar --matrix <matrix.mtx>

# Run optimized GPU kernel
./spmv_gpu --format csr --kernel optimized --matrix <matrix.mtx>

# Run all kernels on all matrices and dump results
./run_benchmarks.sh
```

### Validation

```bash
python3 validation/validate.py --cpu output_cpu.txt --gpu output_gpu.txt --tol 1e-4
```

The tolerance `--tol` is configurable. Exact equality is checked for integer-like test cases; relative tolerance is used for Float32 results.

---

## Implementations

### CPU Baseline
A sequential SpMV over CSR format used solely for correctness validation. An optional OpenMP version is provided for reference.

### GPU Kernel 1 — CSR Scalar (Naive)
One thread per row. Simple and correct, but suffers from **load imbalance** and **non-coalesced memory access** for irregular matrices.

### GPU Kernel 2 — CSR Vector / Optimized
One warp per row (or a tile-based approach). Uses **shared memory** for partial reductions and improves **load balancing** to better utilize Streaming Multiprocessors (SMs).

> Further details and pseudocode are provided in the report.

---

## Measurements

| Metric | How reported |
|--------|-------------|
| Kernel time | CUDA events; excludes one-time setup costs |
| GFLOP/s | Based on `2 · NNZ` FP operations per SpMV |
| Repetitions | Median of N=100 runs (or best-of-N, justified in report) |
| Memory behavior | Optional: cache-miss profiling via `nvprof` / Nsight Compute |

---

## References

1. Gao et al., *A Systematic Literature Survey of Sparse Matrix-Vector Multiplication*, arXiv 2404.06047, 2024.
2. Chu et al., *Efficient Algorithm Design of Optimizing SpMV on GPU*, HPDC '23. DOI: 10.1145/3588195.3593002.
3. Bell & Garland, *Implementing SpMV on Throughput-Oriented Processors*, SC '09.
4. Greathouse & Daga, *Efficient SpMV on GPUs Using the CSR Storage Format*, SC14.
5. Merrill & Garland, *Merge-Based Parallel SpMV*, SC16.

---

## License

For academic use only — GPU Computing course 2025-2026.
