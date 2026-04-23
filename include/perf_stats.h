typedef struct {

    // -------------------------------------------------------
    // IDENTIFICATION METADATA
    // -------------------------------------------------------

    char name[256];            // Matrix filename (identifier of dataset)
    char format[32];           // Sparse storage format used (e.g., CSR, COO, ELL)
    char implementation[32];   // Kernel implementation (e.g., CPU single-core, OpenMP, CUDA)

    // -------------------------------------------------------
    // BASIC MATRIX DIMENSIONS
    // -------------------------------------------------------

    int rows;                  // Number of matrix rows (size of output vector y)
    int cols;                  // Number of matrix columns (size of input vector x)
    int nnz;                   // Total number of nonzero elements (dominant factor in SpMV cost)

    // -------------------------------------------------------
    // PERFORMANCE METRICS
    // These quantify runtime stability and computational throughput.
    // -------------------------------------------------------

    double avg_time_s;         // Average execution time of SpMV kernel (excluding preprocessing)

    double std_time_s;         // Runtime variability across repetitions; indicates stability of memory behavior
                               // and sensitivity to cache / OS scheduling effects

    // -------------------------------------------------------
    // COMPUTATIONAL THROUGHPUT
    // -------------------------------------------------------

    double gflops;             // Achieved floating-point throughput: 2 * nnz / execution_time
                               // standard performance metric for SpMV benchmarking

    // -------------------------------------------------------
    // VALIDATION METRICS
    // -------------------------------------------------------

    int valid;                 // Indicates whether kernel output matches reference implementation

    double max_abs_error;      // Maximum absolute difference vs reference result; detects correctness issues
                               // caused by numerical errors or implementation bugs

} PerfStats;