# Examples

Sample optimization problems.

| Example                       | Metric    | Direction | Description                                      |
| ----------------------------- | --------- | --------- | ------------------------------------------------ |
| [agent-optimization](agent-optimization/) | pass_rate | higher | Tool-using agent prompt/config optimization |
| [llm-training](llm-training/) | val_bpb   | lower     | GPT pretraining optimization (MLX/Apple Silicon) |
| [sorting](sorting/)           | time_ms   | lower     | Hybrid quicksort/insertion sort optimization     |
| [matmul](matmul/)             | gflops    | higher    | Cache-efficient matrix multiplication            |
| [regression](regression/)     | mse       | lower     | Gradient descent hyperparameter tuning           |

## Running

```bash
# Test directly
uv run python examples/sorting/train.py
uv run python examples/agent-optimization/train.py

# Optimize with cevolve
uv run cevolve run --target examples/sorting/train.py --metric time_ms --llm pi
uv run cevolve run --target examples/agent-optimization/train.py --metric pass_rate --direction higher --llm pi
uv run cevolve --target examples/matmul/train.py --metric gflops --direction higher --llm pi
uv run cevolve --target examples/regression/train.py --metric mse --llm pi

# LLM training (requires Apple Silicon + MLX, run prepare.py first)
cd examples/llm-training && uv run python prepare.py
uv run cevolve --target examples/llm-training/train.py --metric val_bpb --llm pi
```

## Creating Your Own

Each example needs:

1. **train.py** — Script that outputs metrics in this format:

   ```
   ---
   metric_name: value
   other_metric: value
   ```

2. **Parameters** — Constants at the top of the file for the LLM to modify

3. **README.md** — Description, parameters, and ideas to explore

## Tips

- Keep examples self-contained (no external dependencies beyond stdlib)
- Use a fixed random seed for reproducibility
- Include parameter validation and error checking
- Output a verification/correctness metric alongside the primary metric
