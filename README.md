# GPU Computing — Deliverable 1: Sparse Matrix-Vector Multiplication

CSR-based SpMV kernels benchmarked on an NVIDIA A30 across ten matrices from the SuiteSparse Matrix Collection.

**Author:** Matteo Morellini (268427) — University of Trento
**Contact:** matteo.morellini@studenti.unitn.it

## Overview

Sparse matrix-vector multiplication is bandwidth-bound and highly sensitive to row-length distribution, memory coalescing, and long-row handling. This work compares five CSR-based SpMV implementations on a single GPU, restricting the scope to kernels that operate directly on CSR (no format conversion overhead).

## Kernels

- **CSR-Scalar** — one thread per row. Weakest baseline; uncoalesced accesses and severe load imbalance on irregular rows.
- **CSR-Vector** — one warp per row. Fully coalesced reads, but underutilized warps when rows are shorter than 32 elements.
- **CSR-Adaptive** — dynamically groups contiguous rows into blocks and picks CSR-Stream or block-level CSR-Vector at runtime. Includes an improved variant with a dedicated `csr_longrow_kernel` for rows exceeding `NNZ_PER_BLOCK`.
- **CSR-Partial-Overlap** — two-stage pipeline using `memcpy_async` (Ampere) to overlap global-to-shared transfers with computation.
- **cuSPARSE** — NVIDIA's optimized baseline.

## Experimental Setup

- GPU: NVIDIA A30
- Precision: Float32 (validated against a sequential CPU CSR implementation with relative tolerance `1e-3`)
- Timing: CUDA events, 100 repetitions after 5 warmup runs, excluding file parsing, format conversion, and host-to-device transfer
- Throughput reported as `2 · nnz / t` GFLOP/s
- Symmetric matrices stored in expanded form

## Dataset

Ten matrices spanning diverse sparsity regimes:
`ASIC_680ks`, `FullChip`, `Rucci1`, `Si41Ge41H72`, `bone010`, `boyd2`, `eu-2005`, `ldoor`, `rajat31`, `webbase-1M`.

## Key Results (geometric mean across the portfolio)

| Kernel              | GFLOP/s |
|---------------------|--------:|
| cuSPARSE            |    96.4 |
| CSR-Adaptive        |    59.4 |
| CSR-Partial-Overlap |    40.0 |
| CSR-Scalar          |    22.8 |
| CSR-Vector          |    22.8 |

- **cuSPARSE** wins on 6 of 10 matrices and is the most robust (only 2.5× spread).
- **CSR-Adaptive** is the strongest custom kernel and beats cuSPARSE on `ASIC_680ks` and `webbase-1M`.
- **CSR-Scalar** wins on `Rucci1` and `rajat31` thanks to uniformly short rows that accidentally produce coalesced accesses.
- **Long-row matrices** (`FullChip`, `boyd2`) collapse every kernel that lacks a dedicated long-row handler.

## Ablation: CSR-Adaptive Improvements

Adding a `csr_longrow_kernel` that splits long rows into 1024-element chunks (each processed by an independent block with `atomicAdd` partial sums) yields:

- **18×** speedup on `FullChip` (5.7 → 102.3 GFLOP/s)
- **8.4×** speedup on `boyd2` (7.4 → 62.1 GFLOP/s)

Other matrices are unaffected or show minor regression from the block-size tradeoff.

## Takeaways

- SpMV performance is determined by memory access efficiency and load balance, not floating-point throughput.
- No single custom kernel is uniformly optimal — the best choice depends on row-length distribution.
- Splitting work across multiple blocks is essential for extreme long rows.
- The irregular gather `x[col_idx[j]]` remains the unaddressed bottleneck across all kernels.

## References

1. N. Bell, M. Garland. *Implementing sparse matrix-vector multiplication on throughput-oriented processors.* SC '09.
2. J. Gao et al. *A systematic literature survey of sparse matrix-vector multiplication.* arXiv:2404.06047, 2024.
3. G. Chu et al. *Efficient algorithm design of optimizing SpMV on GPU.* HPDC '23.
4. J. L. Greathouse, M. Daga. *Efficient sparse matrix-vector multiplication on GPUs using the CSR storage format.* SC '14.
5. G. Zeng, Y. Zou. *Leveraging memory copy overlap for efficient sparse matrix-vector multiplication on GPUs.* Electronics 12(17), 2023.