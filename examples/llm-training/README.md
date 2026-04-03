# LLM Training Optimization

Optimize a GPT training script for lowest validation bits-per-byte (BPB).

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx).

## Problem

Small LLM training involves many hyperparameters and architectural choices that interact in non-obvious ways. Finding the best combination requires exploring a large search space.

## Setup

**Requirements:** Apple Silicon Mac (MLX), Python 3.10+

```bash
# Download data and train tokenizer (one-time, ~2 min)
cd examples/llm-training
uv run python prepare.py
```

## Parameters to Optimize

| Parameter          | Default | Description                                |
| ------------------ | ------- | ------------------------------------------ |
| `DEPTH`            | 8       | Number of transformer layers               |
| `TOTAL_BATCH_SIZE` | 2^17    | Total batch size for gradient accumulation |
| `MATRIX_LR`        | 0.04    | Learning rate for matrix parameters        |
| `EMBEDDING_LR`     | 0.6     | Learning rate for embeddings               |
| `WARMDOWN_RATIO`   | 0.5     | Fraction of training for LR cooldown       |
| `WEIGHT_DECAY`     | 0.2     | Weight decay for AdamW                     |
| `WINDOW_PATTERN`   | "SSSL"  | Sliding window attention pattern           |

## Metric

- **val_bpb** (lower is better) — Validation bits per byte

## Constraints

- Training runs for a **fixed 2-minute time budget**
- Only `train.py` can be modified
- No new packages or dependencies

## Run

```bash
# Test a single training run
cd examples/llm-training
uv run python train.py
```

## Optimize

```bash
uv run cevolve --target examples/llm-training/train.py --metric val_bpb --llm pi
```

## Ideas to Explore

- Model depth: 4, 6, 8, 10, 12 layers
- Batch size: 2^15, 2^16, 2^17, 2^18
- Learning rate schedules: warmup ratios, cooldown fractions
- Attention patterns: sliding window sizes
- Activation functions: relu_squared, gelu, swish
- Adam betas tuning
- Weight initialization strategies
