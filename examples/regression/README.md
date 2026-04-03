# Linear Regression Optimization

Optimize gradient descent hyperparameters for linear regression.

## Problem

Train a linear regression model using gradient descent. The optimal learning rate, batch size, and other hyperparameters depend on the data distribution and can significantly affect convergence speed and final accuracy.

## Parameters to Optimize

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.01 | Step size for gradient updates |
| `NUM_ITERATIONS` | 1000 | Training iterations |
| `BATCH_SIZE` | 32 | Samples per batch (0 = full batch) |
| `REGULARIZATION` | 0.001 | L2 regularization strength |
| `LR_SCHEDULE` | "constant" | Schedule: "constant", "linear_decay", "exponential_decay", "step_decay" |
| `MOMENTUM` | 0.0 | Momentum coefficient |
| `EARLY_STOPPING` | 0 | Stop if no improvement for N checks (0 = disabled) |

## Metric

- **mse** (lower is better) — Mean squared error on validation set

## Run

```bash
cd examples/regression
uv run python train.py
```

## Optimize

```bash
uv run cli.py --target examples/regression/train.py --metric mse --llm pi
```

## Ideas to Explore

- Learning rates: 0.001, 0.01, 0.1
- Batch sizes: 16, 32, 64, full batch
- Learning rate schedules with different decay rates
- Momentum values: 0.9, 0.99
- Adam optimizer (add beta1, beta2, epsilon parameters)
- Feature normalization/standardization
- Different regularization (L1, elastic net)
