# Sorting Optimization

Optimize a hybrid quicksort/insertion sort algorithm.

## Problem

Quicksort is fast for large arrays but has overhead that makes it slower than insertion sort for small arrays. A hybrid approach switches to insertion sort below a threshold. Finding the optimal threshold—and other parameters—depends on the data patterns and hardware.

## Parameters to Optimize

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INSERTION_THRESHOLD` | 16 | Switch to insertion sort below this size |
| `PIVOT_STRATEGY` | "median_of_three" | How to select pivot: "first", "middle", "median_of_three", "random" |
| `USE_ITERATIVE` | False | Use iterative (stack-based) vs recursive quicksort |

## Metric

- **time_ms** (lower is better) — Total time to sort 100 arrays of 10,000 elements each

## Run

```bash
cd examples/sorting
uv run python train.py
```

## Optimize

```bash
uv run cli.py --target examples/sorting/train.py --metric time_ms --llm pi
```

## Ideas to Explore

- Threshold values: 8, 16, 32, 64
- Pivot strategies for different data patterns
- Iterative vs recursive (stack depth vs function call overhead)
- Three-way partitioning for arrays with many duplicates
- Introsort (switch to heapsort if recursion too deep)
