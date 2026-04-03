# Genetic Algorithm Design

This document describes the genetic algorithm design for evolutionary code optimization.

---

## Overview Diagrams

### Main Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                              INIT                                   │
│  Define ideas → Configure params → Create population (baseline + N) │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                ┌─────────────────────────────────────────┐
                │           GET NEXT INDIVIDUAL           │
                │  ┌─────────────────────────────────┐    │
                │  │ Stop? (max evals / converged)   │────┼──────► STOP
                │  └─────────────────────────────────┘    │
                │  ┌─────────────────────────────────┐    │
                │  │ Rethink due?                    │────┼──────► RETHINK ─┐
                │  └─────────────────────────────────┘    │                 │
                │  ┌─────────────────────────────────┐    │                 │
                │  │ All evaluated? → EVOLVE         │    │◄────────────────┘
                │  └─────────────────────────────────┘    │
                │  Return next unevaluated individual     │
                └─────────────────────┬───────────────────┘
                                      │
                                      ▼
                ┌─────────────────────────────────────────┐
                │              EVALUATE                   │
                │  Apply changes → Run experiment →       │
                │  Capture metrics → Revert changes →     │
                │  Record fitness → Update best           │
                └─────────────────────┬───────────────────┘
                                      │
                                      └──────────────┐
                                                     │
                                                     ▼
                                              (loop back to
                                             GET NEXT INDIVIDUAL)
```

### Generation Lifecycle

```
Generation N                           Generation N+1
┌─────────────────────┐               ┌─────────────────────┐
│ Individual 1 ✓ 1.23 │──┐            │ Elite 1 (copy)  ✓   │
│ Individual 2 ✓ 1.45 │  │  EVOLVE    │ Elite 2 (copy)  ✓   │
│ Individual 3 ✓ 1.12 │──┼──────────► │ Child 1 (new)       │
│ Individual 4 ✓ 1.67 │  │            │ Child 2 (new)       │
│ Individual 5 ✓ 1.34 │──┘            │ Child 3 (new)       │
└─────────────────────┘               └─────────────────────┘
     all evaluated                      elites keep fitness,
                                        children need evaluation

EVOLVE:
  1. Sort by fitness (best first)
  2. Copy top N as elites (preserve fitness + metrics)
  3. Fill rest via: select parents → crossover → mutate
```

### Individual Structure

```
Individual
├── id: "ind-a1b2c3"
├── genes: {                    ◄── one entry per idea
│     "depth": "6",                  (variant value)
│     "flash_attn": "on",            (binary ON)
│     "dropout": null                (OFF)
│   }
├── fitness: 1.234              ◄── primary metric (null if not evaluated)
├── metrics: { "time": 45.2 }   ◄── secondary metrics
├── generation: 2
└── parents: ["ind-x", "ind-y"] ◄── optional, for offspring
```

### Rethink Flow

```
On rethink (every N evaluations):
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Commit best │───►│  Update baseline│───►│  Reset pop,      │
│  if exists   │    │  code, era++    │    │  re-initialize   │
└──────────────┘    └─────────────────┘    └──────────────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                           │  Analyze stats,  │
                                           │  suggest ideas   │
                                           └──────────────────┘
```

---

## Core Concepts

### Ideas

An **Idea** represents a single optimization concept to explore:

```
Idea:
  name: string           # Short identifier (snake_case)
  description: string    # Human-readable description
  variants: string[]?    # Optional discrete values; null = binary (on/off)
```

**Binary ideas** have no variants - they're either applied or not (e.g., "use_flash_attention").

**Variant ideas** have multiple discrete values (e.g., "depth" with variants ["4", "6", "8"]).

### Individuals

An **Individual** is a candidate solution - a specific combination of idea settings:

```
Individual:
  id: string                          # Unique identifier (e.g., "ind-abc123")
  genes: map<string, string | null>   # idea_name → variant (null = off, "on" for binary, or variant value)
  fitness: number | null              # Primary metric value (null = not yet evaluated)
  metrics: map<string, number>        # Secondary metrics for monitoring
  generation: int                     # Which generation this individual was created in
  parents: [string, string]?          # IDs of parent individuals (for offspring)
```

**Gene representation**: Every idea has an entry in `genes`. Value is:

- `null` → idea is OFF
- `"on"` → binary idea is ON
- `"<variant>"` → variant idea is ON with that specific value

### Baseline

The **baseline** is an individual with all genes set to `null` (no optimizations applied). It establishes the reference fitness for comparison.

---

## Configuration

```
Config:
  name: string                    # Session name

  # Metrics
  metricName: string              # Primary metric to optimize (e.g., "val_bpb")
  metricDirection: "lower"|"higher"  # Whether lower or higher is better
  metricUnit: string              # Display unit (e.g., "ms", "MB", "")
  secondaryMetrics: []            # Additional metrics to track (not optimized)
    - name: string
    - unit: string
    - direction: "lower"|"higher"

  # GA parameters
  populationSize: int             # Number of individuals per generation (default: 6)
  elitism: int                    # Top N individuals to preserve (default: 2)
  mutationRate: float             # Probability of mutating each gene (default: 0.2)
  crossoverRate: float            # Probability of crossover vs copy (default: 0.7)

  # Stopping conditions
  maxEvaluations: int | null      # Maximum evaluations (null = unlimited)
  convergenceEvals: int           # Stop if no improvement for N evals (default: rethinkInterval * 3 + 1)

  # Execution
  trainCommand: string            # Command to run experiment
  experimentTimeout: int          # Timeout per experiment in seconds (default: 600)

  # Rethink
  rethinkInterval: int            # Trigger rethink every N evaluations (default: 5, 0 = disable)
```

---

## State

The algorithm maintains this state across evaluations:

```
State:
  config: Config
  ideas: map<string, Idea>        # Current idea pool
  population: Individual[]        # Current generation
  generation: int                 # Generation counter
  evaluations: int                # Total evaluations completed
  best: Individual | null         # Best individual found so far
  bestAtEval: int                 # Evaluation number when best was last improved
  era: int                        # Rethink counter
  lastRethink: int                # Evaluation count at last rethink
  history: HistoryEntry[]         # All evaluation results
  currentIndividual: string | null  # ID of individual being evaluated (for crash recovery)
```

---

## Algorithm Operations

### Initialization

1. Create initial population of `populationSize` individuals
2. First individual is always **baseline** (all genes = null)
3. Remaining individuals are **random**:
   - For each idea, 50% chance to be active
   - If active: binary → "on", variant → random choice from variants

### Selection (Tournament)

Select parent for reproduction:

1. Filter to evaluated individuals with finite fitness (exclude `null` and `Infinity`)
2. If none available, return random individual from population
3. Pick 3 random individuals from filtered set (tournament size = min(3, available))
4. Return the one with best fitness (respecting direction)

### Crossover (Uniform)

Create offspring from two parents:

1. Record both parent IDs on child (always, even if no crossover)
2. If random > crossoverRate: copy genes from random parent (no crossover)
3. Otherwise, for each idea in the idea pool:
   - Pick gene value from either parent (50/50)
   - Use `null` if parent doesn't have the gene

### Mutation

Modify an individual's genes (in place):

For each gene with probability `mutationRate`:

- **If variant idea**: pick a DIFFERENT value from [variants..., null], excluding current
- **If binary idea**: toggle (null ↔ "on")

### Gene Normalization

When ideas are added/removed, normalize all individuals:

1. For each idea in current pool: if missing from individual, set to `null`
2. For each gene in individual: if idea no longer exists, delete it

This ensures individuals always have exactly one entry per current idea.

### Evolution (Generation Transition)

1. Increment generation counter
2. Sort evaluated population by fitness (best first, Infinity last, respecting direction)
3. **Elitism**: Copy top N individuals to new population (preserving fitness AND metrics)
4. **Fill remaining**: Repeat until population full:
   - Select two parents via tournament
   - Create child via crossover
   - Apply mutation
   - Normalize genes
   - Add to new population
5. Replace population

---

## Evaluation Flow

### Getting Next Individual

1. Check stopping conditions (max evaluations, convergence) → stop if met
2. Check if rethink is required (`evaluations - lastRethink >= rethinkInterval`)
   - If so, trigger rethink before continuing
3. Find first individual in population with `fitness = null`
4. If none available, trigger evolution to next generation
5. Set `currentIndividual` to that ID
6. Return the individual for evaluation

### Recording Evaluation Result

1. Find individual by ID
2. Set `fitness` and `metrics`
3. Increment `evaluations`
4. If better than `best`:
   - Update `best`
   - Update `bestAtEval = evaluations`
5. Append to history
6. Clear `currentIndividual`
7. Save state

### Fitness Comparison

```
isBetter(a, b, direction):
  if direction == "lower": return a < b
  else: return a > b
```

---

## Stopping Conditions

Check after each evaluation:

1. **Max evaluations**: `evaluations >= maxEvaluations` (if set)
2. **Convergence**: `evaluations - bestAtEval >= convergenceEvals`

Either condition triggers stop.

---

## Metric Output Format

The train command must output metrics in one of these formats:

**Format 1: Colon-separated**

```
metricName: value
```

**Format 2: METRIC keyword with equals**

```
METRIC metricName=value
```

Example output:

```
val_bpb: 1.234
train_time: 45.2
METRIC memory_mb=2048
```

The primary metric (matching `metricName`) determines fitness. Others populate `metrics` map.

---

## Rethink Mechanism

Triggered when `evaluations - lastRethink >= rethinkInterval`:

1. Generate statistics for analysis:
   - Per-idea: eval count, success rate, appearances in top 5
   - Per-variant: count, average fitness, best fitness
   - Untested variants
   - Recommendations (bad genes to remove, promising genes)
2. Update `lastRethink = evaluations`
3. If ideas are added or removed:
   - If population is empty: create fresh population (baseline + random)
   - Otherwise: normalize all individuals' genes
   - Increment `era`

### Commit Best (on Rethink)

On each rethink, if a best configuration exists, it becomes the new baseline:

1. Write best individual's code to the target file
2. Update `original_code` to the new baseline
3. Increment `era`
4. Add history entry marking era transition
5. Reset `generation = 0`
6. Clear `best = null` and set `bestAtEval = evaluations`
7. Clear population and re-initialize (baseline + random)
8. Analyze results and potentially add new ideas

This allows incremental progress - locking in proven improvements while continuing to explore further optimizations on top of the new baseline.

---

## Summary Statistics

### Per-Idea Analysis

For each idea, compute:

- **Eval count**: Number of evaluations where idea was ON
- **Success rate**: Percentage of ON evaluations that improved over baseline
- **Top appearances**: How many times idea appeared in top 5 performers

### Per-Variant Analysis

For variant ideas, track per variant:

- **Count**: How many times this variant was tested
- **Average fitness**: Mean fitness when using this variant
- **Best fitness**: Best fitness achieved with this variant
- **Untested**: Variants that haven't been evaluated yet

### Dead Ends

Ideas with 3+ evaluations and 0% success rate.

### Improvement from Baseline

```
improvement = (best.fitness - baseline.fitness) / baseline.fitness * 100%
```

---

## Session Persistence

State persists to disk for resumability:

```
.cevolve/<session-name>/
  config.json       # Configuration
  ideas.json        # Current idea pool
  population.json   # Population, generation, evaluations, best, currentIndividual
  history.jsonl     # Append-only evaluation log
  RESULTS.md        # Human-readable summary (generated on stop)
```

### History Entry Format

```json
{
  "timestamp": 1234567890.123,
  "evaluation": 5,
  "generation": 1,
  "id": "ind-abc123",
  "genes": { "depth": "6", "flash_attn": "on", "dropout": null },
  "fitness": 1.234,
  "metrics": { "train_time": 45.2 }
}
```

---

## Workflow Summary

```
1. INIT
   - Define ideas (with optional variants)
   - Configure parameters
   - Create initial population (baseline + random)

2. EVALUATE LOOP
   - Get next unevaluated individual
   - Apply code changes for active genes (implementation-specific)
   - Run experiment, capture metrics
   - Revert code changes
   - Record fitness
   - Check stopping conditions → stop if met
   - Check rethink interval → trigger rethink if due
   - If generation complete: evolve to next generation

3. STOP
   - Generate summary (RESULTS.md)
   - Report best configuration
```
