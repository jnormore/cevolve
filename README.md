# cEvolve

_The LLM imagines. Evolution discovers._

Genetic algorithms for autonomous code and agent optimization. The LLM generates ideas, the GA explores which combinations work best together.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LLM generates ideas         GA evolves combinations         Best wins     │
│   ───────────────────         ────────────────────────         ─────────    │
│                                                                             │
│   "try caching"          ┌─► [cache, batch=32]  ────► 45ms                  │
│   "batch sizes 16,32,64" │                                                  │
│   "use SIMD"             ├─► [SIMD, cache]      ────► 38ms  ◄── winner!     │
│   "reduce allocations"   │                                                  │
│                          └─► [batch=64, alloc]  ────► 52ms                  │
│                                    │                                        │
│                                    ▼                                        │
│                          crossover + mutate                                 │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌─► [SIMD, cache, batch=32] ──► 35ms ◄── new best! │
│                          │                                                  │
│                          └─► ...                                            │
│                                    │                                        │
│                                    ▼                                        │
│                            ┌─────────────┐                                  │
│                            │   RETHINK   │  LLM analyzes what worked,       │
│                            │   (commit)  │  adds new ideas, removes duds    │
│                            └─────────────┘                                  │
│                                    │                                        │
│                                    ▼                                        │
│                             continue evolving...                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), `claude` or `pi` CLI.

```bash
# Install
uv sync

# Run evolution (primary CLI for humans)
uv run cevolve run --target examples/sorting/train.py --metric time_ms --llm pi
```

---

## How It Works

### Hill-Climbing vs Evolution

**Hill-climbing** (greedy) tries one change at a time:

```
baseline → try A → better? keep : discard → try B → ...
```

This struggles when:

- **Ideas interact**: A+B together beats A or B alone
- **Ideas conflict**: A helps, B helps, but A+B hurts
- **Local optima**: Greedy gets stuck
- **Time matters**: Each eval waits for the previous (no parallelization)

**Evolution** maintains a population and evolves combinations:

```
Generation 0:  [A], [B], [C], [A,B], [B,C]  → evaluate all (in parallel!)
                           ↓
              Selection: keep fittest
              Crossover: [A] × [B,C] → [A,C]
              Mutation:  [A,C] → [A,C,D]
                           ↓
Generation 1:  [A,B], [A,C], [A,C,D], ...   → evaluate all (in parallel!)
```

Crossover discovers combinations that greedy search might never find.

### Parallelization Advantage

```
Greedy:  eval1 → wait → eval2 → wait → eval3 → wait → ...  (sequential)
cEvolve: [eval1, eval2, eval3, eval4, eval5, eval6]       (parallel!)
             ↓
         [eval7, eval8, eval9, eval10, eval11, eval12]    (parallel!)

30 evals with 6 workers:
  Greedy: 30 × eval_time (sequential)
  cEvolve: 5 × eval_time  (6× faster wall-clock)
```

For expensive evaluations (LLM training, compilation, simulation), this can mean the difference of optimizations you can actually run vs those that aren't practical with hardware limitations.

### The Rethink Loop

Every N evaluations, the LLM analyzes results and suggests new ideas:

```
┌─────────────────────────────────────────────────────────────────┐
│    ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│    │  IDEAS   │ ──► │ EVOLVE   │ ──► │ RETHINK  │ ─┐           │
│    │  (genes) │     │ (N evals)│     │ (analyze)│  │           │
│    └──────────┘     └──────────┘     └──────────┘  │           │
│         ▲                                          │           │
│         │         Add new ideas, commit best       │           │
│         └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Accumulating Wins

On each rethink, the best configuration is kept and becomes the new baseline. Essentially a hill-climb optimization outer loop:

```
Era 1: Explore combinations → [reduce_depth] wins → commit
    ↓
Era 2: Optimize ON TOP of reduce_depth → find +5% more → commit
    ↓
Era 3: Continue stacking improvements...
```

This combines evolution's parallel exploration with sequential accumulation.

---

## Results

On an agent optimization benchmark (25 tasks, 5 tunable parameters):

| Metric                       | Greedy   | cEvolve              |
| ---------------------------- | -------- | -------------------- |
| Final performance            | 0.896    | 0.896                |
| Evals to reach good solution | 8.2      | **3.2** (61% faster) |
| Found max_iter+cot synergy   | 0/5 runs | **2/5 runs**         |

**Key finding:** cEvolve reaches good solutions **2.6× faster** and discovers gene combinations that greedy search never finds.

See [WHITEPAPER.md](WHITEPAPER.md) for full experimental details.

---

## Usage

### Primary CLI (`cevolve run`)

For humans using the tool directly. Built-in LLM discovers ideas and implements genes:

```bash
# Single file optimization
cevolve run --target train.py --metric val_bpb --llm claude

# Multi-file optimization
cevolve run --scope "src/**/*.py" --metric time_ms --llm pi

# Options
cevolve run --target train.py --metric time_ms --llm pi \
  --max-evals 50 \      # More evaluations
  --no-tui \            # Plain output (no TUI)
  --dry-run             # Mock LLM and training
```

### Composable Commands (for extensions and agents)

For tools like pi-evolve that bring their own LLM/agent logic, or direct use by agents:

```bash
# Initialize session with ideas
cevolve init --name my-opt \
  --idea "use_cache: Enable caching" \
  --idea "batch_size[16,32,64]: Batch size" \
  --bench "./bench.sh" \
  --metric time_ms

# Get next individual (genes to implement)
cevolve next --json
# {"individual_id": "ind-abc123", "genes": {"use_cache": "on", "batch_size": "32"}, ...}

# Extension implements genes using its own tools...

# Run benchmark, record result, revert changes
cevolve eval --id ind-abc123

# Or if extension runs benchmark itself:
cevolve record --id ind-abc123 --fitness 42.5 --metrics '{"memory_mb": 128}'
cevolve revert

# Analyze and modify ideas
cevolve rethink --add-idea "use_mmap: Memory-map files"

# Check status
cevolve status

# Finalize
cevolve stop
```

**Key difference:** `cevolve run` has built-in LLM. Composable commands are LLM-agnostic.

### `run` Options

| Flag          | Default    | Description                                   |
| ------------- | ---------- | --------------------------------------------- |
| `--target`    | `train.py` | File to optimize (single-file mode)           |
| `--scope`     | -          | Glob patterns (multi-file mode)               |
| `--metric`    | `val_bpb`  | Metric to optimize                            |
| `--direction` | `lower`    | Optimization direction (`lower` or `higher`)  |
| `--llm`       | `claude`   | LLM CLI to use (`claude` or `pi`)             |
| `--max-evals` | `20`       | Maximum evaluations                           |
| `--pop-size`  | `6`        | Population size                               |
| `--rethink`   | `5`        | Rethink every N evals (0 to disable)          |
| `--name`      | auto       | Session name (default: `run-YYYYMMDD-HHMMSS`) |
| `--no-tui`    | -          | Disable TUI, plain output                     |
| `--dry-run`   | -          | Mock LLM and training                         |

### `init` Options

| Flag       | Default   | Description                              |
| ---------- | --------- | ---------------------------------------- |
| `--name`   | required  | Session name                             |
| `--ideas`  | -         | JSON file or inline array of ideas       |
| `--idea`   | -         | Inline idea (repeatable)                 |
| `--bench`  | required  | Benchmark command                        |
| `--metric` | `time_ms` | Metric to optimize                       |
| `--revert` | `git`     | Revert strategy: `git`, `stash`, `cache` |

### TUI Controls

| Key            | Action                             |
| -------------- | ---------------------------------- |
| `d`            | Toggle details panel (shows ideas) |
| `↑/k`          | Scroll log up                      |
| `↓/j`          | Scroll log down                    |
| `Page Up/Down` | Scroll 20 lines                    |
| `Home/End`     | Jump to oldest/newest              |

---

## Configuration

| Parameter            | Default | Description                            |
| -------------------- | ------- | -------------------------------------- |
| `population_size`    | 6       | Individuals per generation             |
| `elitism`            | 2       | Top N kept each generation             |
| `mutation_rate`      | 0.2     | Gene mutation probability              |
| `crossover_rate`     | 0.7     | Crossover vs copy probability          |
| `convergence_evals`  | 16      | Stop after N evals without improvement |
| `rethink_interval`   | 5       | Analyze progress every N evals         |
| `experiment_timeout` | 600     | Timeout per experiment (seconds)       |

---

## Output

Results saved to `.cevolve/<session-name>/`:

```
.cevolve/run-20240404-123456/
├── config.json       # Run configuration
├── ideas.json        # Ideas explored
├── population.json   # Final population state
├── history.jsonl     # All evaluations (for analysis)
├── RESULTS.md        # Human-readable summary
├── convergence.png   # Fitness over time
├── idea_analysis.png # Idea effectiveness
└── synergy_matrix.png # Idea interactions
```

### history.jsonl

Each line is one evaluation:

```json
{"evaluation": 1, "generation": 0, "id": "ind-123", "genes": {"depth": null, ...}, "fitness": 2.493, "metrics": {...}}
```

### Charts

**convergence.png** — Fitness over evaluations with best-so-far line

**idea_analysis.png** — Average fitness with each idea ON vs OFF

**synergy_matrix.png** — Heatmap of idea combinations

---

## Documentation

| Doc                             | Description                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| **[Whitepaper](WHITEPAPER.md)** | How and why cEvolve works. Algorithm details, experimental results, when to use it. |
| **[Design](DESIGN.md)**         | Technical specification of the evolutionary algorithm.                              |

---

## License

MIT
