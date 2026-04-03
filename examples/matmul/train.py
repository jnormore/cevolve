"""
Matrix multiplication optimization.

Optimize tiling, loop ordering, and other parameters for
cache-efficient matrix multiplication.

Metric: gflops (higher is better)
"""

import random
import time

# =============================================================================
# PARAMETERS TO OPTIMIZE
# =============================================================================

# Matrix dimensions (square matrices)
MATRIX_SIZE = 512

# Tile/block size for cache optimization
# Should divide MATRIX_SIZE evenly for simplicity
TILE_SIZE = 32

# Loop order: "ijk", "ikj", "jik", "jki", "kij", "kji"
# Different orders have different cache access patterns
LOOP_ORDER = "ikj"

# Use tiled (blocked) matrix multiplication
USE_TILING = True

# Number of benchmark iterations
NUM_ITERATIONS = 3

# =============================================================================
# MATRIX MULTIPLICATION IMPLEMENTATIONS
# =============================================================================

def matmul_naive(A, B, C, n):
    """Naive O(n³) matrix multiplication with configurable loop order."""
    if LOOP_ORDER == "ijk":
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
    elif LOOP_ORDER == "ikj":
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    C[i][j] += A[i][k] * B[k][j]
    elif LOOP_ORDER == "jik":
        for j in range(n):
            for i in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
    elif LOOP_ORDER == "jki":
        for j in range(n):
            for k in range(n):
                for i in range(n):
                    C[i][j] += A[i][k] * B[k][j]
    elif LOOP_ORDER == "kij":
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    C[i][j] += A[i][k] * B[k][j]
    elif LOOP_ORDER == "kji":
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    C[i][j] += A[i][k] * B[k][j]


def matmul_tiled(A, B, C, n):
    """Tiled (blocked) matrix multiplication for better cache usage."""
    tile = TILE_SIZE
    
    for ii in range(0, n, tile):
        for kk in range(0, n, tile):
            for jj in range(0, n, tile):
                # Multiply tiles
                for i in range(ii, min(ii + tile, n)):
                    for k in range(kk, min(kk + tile, n)):
                        a_ik = A[i][k]
                        for j in range(jj, min(jj + tile, n)):
                            C[i][j] += a_ik * B[k][j]


def matmul(A, B, C, n):
    """Matrix multiplication dispatcher."""
    if USE_TILING:
        matmul_tiled(A, B, C, n)
    else:
        matmul_naive(A, B, C, n)


# =============================================================================
# BENCHMARK
# =============================================================================

def create_matrix(n, value=None):
    """Create n×n matrix, optionally filled with value."""
    if value is not None:
        return [[value for _ in range(n)] for _ in range(n)]
    return [[random.random() for _ in range(n)] for _ in range(n)]


def verify_result(A, B, C, n, samples=10):
    """Spot-check a few elements of the result."""
    for _ in range(samples):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        expected = sum(A[i][k] * B[k][j] for k in range(n))
        if abs(C[i][j] - expected) > 1e-6:
            return False
    return True


def benchmark():
    """Run matrix multiplication benchmark."""
    n = MATRIX_SIZE
    
    print(f"Matrix size: {n}x{n}")
    print(f"Configuration:")
    print(f"  TILE_SIZE: {TILE_SIZE}")
    print(f"  LOOP_ORDER: {LOOP_ORDER}")
    print(f"  USE_TILING: {USE_TILING}")
    print(f"  NUM_ITERATIONS: {NUM_ITERATIONS}")
    print()
    
    # Generate random matrices
    random.seed(42)
    A = create_matrix(n)
    B = create_matrix(n)
    
    # Warm up
    C_warmup = create_matrix(n // 4, 0.0)
    A_small = [row[:n//4] for row in A[:n//4]]
    B_small = [row[:n//4] for row in B[:n//4]]
    matmul(A_small, B_small, C_warmup, n // 4)
    
    # Benchmark
    times = []
    C = None
    
    for iteration in range(NUM_ITERATIONS):
        C = create_matrix(n, 0.0)
        
        start = time.perf_counter()
        matmul(A, B, C, n)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        print(f"  Iteration {iteration + 1}: {elapsed:.3f}s")
    
    # Use best time (least interference)
    best_time = min(times)
    avg_time = sum(times) / len(times)
    
    # Calculate GFLOPS
    # Matrix multiplication: 2*n³ floating point operations (n³ multiplies + n³ adds)
    flops = 2 * (n ** 3)
    gflops = (flops / best_time) / 1e9
    
    # Verify correctness
    random.seed(123)  # Different seed for verification
    correct = verify_result(A, B, C, n)
    
    # Output metrics
    print()
    print("---")
    print(f"gflops: {gflops:.4f}")
    print(f"best_time_s: {best_time:.4f}")
    print(f"avg_time_s: {avg_time:.4f}")
    print(f"correct: {1 if correct else 0}")
    
    if not correct:
        print("\nERROR: Result verification failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(benchmark())
