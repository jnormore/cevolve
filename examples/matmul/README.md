# Matrix Multiplication Optimization

Optimize cache-efficient matrix multiplication.

## Problem

Naive matrix multiplication has poor cache performance because it accesses memory in patterns that cause cache misses. Tiling (blocking) and loop reordering can dramatically improve performance by keeping data in cache.

## Parameters to Optimize

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TILE_SIZE` | 32 | Block size for tiled multiplication |
| `LOOP_ORDER` | "ikj" | Loop nesting order: "ijk", "ikj", "jik", "jki", "kij", "kji" |
| `USE_TILING` | True | Use tiled vs naive implementation |
| `MATRIX_SIZE` | 512 | Size of square matrices |

## Metric

- **gflops** (higher is better) — Billions of floating point operations per second

## Run

```bash
cd examples/matmul
uv run python train.py
```

## Optimize

```bash
uv run cli.py --target examples/matmul/train.py --metric gflops --direction higher --llm pi
```

## Ideas to Explore

- Tile sizes: 16, 32, 64, 128 (depends on cache size)
- Loop orders: "ikj" and "kij" typically best for row-major storage
- Register blocking (unroll inner loops)
- SIMD vectorization hints
- Strassen's algorithm for very large matrices
- Copy optimization (copy tiles to contiguous memory)

## Background

For row-major storage (Python lists):
- "ijk" order: B accessed column-wise (bad)
- "ikj" order: Both A and C accessed row-wise (good)
- Tiling keeps working set in L1/L2 cache
