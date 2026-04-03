# cEvolve: Genetic Algorithms for LLM-Guided Code Optimization

_The LLM imagines. Evolution discovers._

---

## Abstract

Large Language Models can suggest code optimizations, but current approaches test ideas one at a time-a greedy strategy that misses how optimizations interact. When hyperparameters synergize (A+B outperforms both A and B alone) or conflict (A+B worse than either), greedy search gets stuck.

**cEvolve** combines LLM hypothesis generation with genetic algorithm exploration. Instead of testing ideas sequentially, cEvolve maintains a population of idea _combinations_ and uses crossover to discover which ideas work together. A novel "rethink" mechanism periodically commits winning configurations and generates new hypotheses based on what's working.

On an agent optimization benchmark (25 tasks, 5 tunable parameters), cEvolve:

- Achieves **equivalent final performance** to greedy search
- Converges to good solutions **61% faster** (3.2 vs 8.2 evaluations)
- Discovers synergies that **greedy never finds** (2/5 vs 0/5 runs)

For expensive optimization problems-LLM training, compiler tuning, system configuration-where each evaluation costs minutes or hours, cEvolve's faster convergence translates directly to saved compute.

---

## 1. Introduction

### 1.1 The Problem: Optimization Ideas Interact

Consider optimizing an AI agent. You have ideas:

- **Increase max_iterations** (more attempts to solve task)
- **Add chain-of-thought reasoning** (think step-by-step)

A greedy optimizer tests each idea independently:

```
1. Try max_iterations=10  → no improvement → REJECT
2. Try chain_of_thought   → slight improvement → KEEP
3. Done. Final: just chain_of_thought.
```

But what if `max_iterations=10` + `chain_of_thought` _together_ is the best configuration? More iterations give the agent time to leverage its reasoning. **This synergy is invisible to greedy search.**

```
Synergy Example:

  max_iterations=10 alone:     0% improvement (no help)
  chain_of_thought alone:     +8% improvement (modest)

  COMBINED:                  +20% improvement! (synergy)
```

Greedy rejected `max_iterations=10` because it doesn't help _in isolation_. It never tested the combination.

### 1.2 The Solution: Population-Based Search

Genetic algorithms naturally discover combinations and enable **parallel evaluation**:

```
┌─────────────────────────────────────────────────────────────-┐
│  GREEDY: Inherently sequential                               │
│                                                              │
│  eval 1 → wait → eval 2 → wait → eval 3 → wait → ...         │
│                                                              │
│  Each step depends on the previous result.                   │
│  Cannot parallelize. Total time = N × eval_time              │
├─────────────────────────────────────────────────────────────-┤
│  EVOLVE: Naturally parallel                                  │
│                                                              │
│  Generation 1: [eval 1] [eval 2] [eval 3] [eval 4] ← parallel│
│                              ↓                               │
│  Generation 2: [eval 5] [eval 6] [eval 7] [eval 8] ← parallel│
│                                                              │
│  Population members are independent.                         │
│  Can parallelize. Total time = generations × eval_time       │
└─────────────────────────────────────────────────────────────-┘

With 8 parallel workers and 32 evaluations:
  Greedy: 32 sequential evals = 32 × eval_time
  cEvolve: 4 generations × 8 parallel = 4 × eval_time (8× faster)
```

Parallel fitness evaluation has been shown to reduce evolutionary optimization from years to hours, making previously intractable problems solvable [3].

```
┌─────────────────────────────────────────────────────────────┐
│  GREEDY: Test ideas one at a time                           │
│                                                             │
│  baseline → +A? → +B? → +C? → ...                           │
│             ↓      ↓                                        │
│            keep  reject (tested alone, not with A)          │
├─────────────────────────────────────────────────────────────┤
│  EVOLVE: Test combinations, discover what works together    │
│                                                             │
│  Population: [A], [B], [C], [A,B], [B,C], [A,C]             │
│                          ↓                                  │
│  Selection: keep fit individuals                            │
│  Crossover: [A] × [B,C] → [A,C] (new combination!)          │
│  Mutation:  [A,C] → [A,C,D]                                 │
│                          ↓                                  │
│  Next gen: [A,B], [A,C], [A,C,D], ...                       │
└─────────────────────────────────────────────────────────────┘
```

**Crossover is the key mechanism:** by recombining genes from different parents, the GA creates combinations that no sequential testing would produce.

### 1.3 Contributions

1. **cEvolve:** A system combining LLM hypothesis generation with GA exploration
2. **Rethink mechanism:** Adaptive search that commits wins and generates new hypotheses
3. **Empirical evidence:** Population search discovers synergies greedy misses
4. **Open source:** Working implementation at [github.com/...]

---

## 2. How cEvolve Works

### 2.1 System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         EVOLVE SYSTEM                            │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │
│  │    LLM      │     │     GA      │     │  Benchmark  │         │
│  │             │     │             │     │             │         │
│  │ "Try these  │────▶│ Population  │────▶│ Run code,   │         │
│  │  ideas..."  │     │ of combos   │     │ get fitness │         │
│  │             │◀────│             │◀────│             │         │
│  │ "These      │     │ Selection,  │     │             │         │
│  │  worked..." │     │ crossover,  │     │             │         │
│  └─────────────┘     │ mutation    │     └─────────────┘         │
│        ▲             └─────────────┘                             │
│        │                   │                                     │
│        └───────────────────┘                                     │
│              Rethink: analyze results,                           │
│              commit wins, new hypotheses                         │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Ideas and Genes

An **idea** is a discrete optimization choice:

- Binary: "use flash attention" (on/off)
- Variant: "model depth" (4, 6, 8, or 10 layers)

An **individual** is a combination of idea settings-a candidate configuration to evaluate.

```python
Individual:
  genes: {
    "depth": "6",           # variant idea
    "flash_attn": "on",     # binary idea
    "batch_size": "large",  # variant idea
    "dropout": null         # idea is OFF
  }
  fitness: 2.32             # measured performance
```

### 2.3 The GA Loop

```
Algorithm: cEvolve
─────────────────────────────────────────────────────
1. INITIALIZE population (baseline + random individuals)

2. REPEAT until converged:

   a. EVALUATE each individual:
      - Apply configuration to code
      - Run benchmark → get fitness
      - Revert changes

   b. If all evaluated, EVOLVE to next generation:
      - Sort by fitness
      - Keep top 2 (elitism)
      - Fill rest via selection + crossover + mutation

   c. If rethink interval reached:
      - Commit best configuration
      - Analyze what worked
      - Add/remove ideas
      - Reset population

3. RETURN best configuration found
```

### 2.4 Crossover: The Synergy Discovery Mechanism

```
How crossover discovers hidden synergies:

Generation 0: Random population
  ind-1: [max_iterations=10]                    pass_rate: 0.72
  ind-2: [chain_of_thought]                     pass_rate: 0.76
  ind-3: [max_iterations=10, history]           pass_rate: 0.74
  ind-4: [chain_of_thought, retry]              pass_rate: 0.78

Generation 1: Crossover combines traits

  Parent A: ind-1 [max_iterations=10]           (mediocre alone)
  Parent B: ind-2 [chain_of_thought]            (decent)

  Crossover: take max_iterations from A, chain_of_thought from B

  Child: [max_iterations=10, chain_of_thought]  pass_rate: 0.92 ← SYNERGY!

The child combines genes that were never tested together,
accidentally discovering the synergy.
```

### 2.5 Rethink: Adaptive Hypothesis Refinement

Every N evaluations, cEvolve "rethinks":

```
┌─────────────────────────────────────────────────────────────┐
│  RETHINK LOOP                                               │
│                                                             │
│  1. COMMIT best configuration as new baseline               │
│     "max_iterations=10 + cot works → lock it in"           │
│                                                             │
│  2. ANALYZE what worked and what didn't                     │
│     "chain_of_thought: 80% success rate"                   │
│     "retry_with_feedback: 10% success rate → remove?"      │
│                                                             │
│  3. GENERATE new hypotheses                                 │
│     "cot works well... try detailed tool instructions?"    │
│     → Add tool_instructions=detailed idea                   │
│                                                             │
│  4. RESET population and continue                           │
│     New era explores ON TOP of committed wins               │
└─────────────────────────────────────────────────────────────┘
```

This creates **era-based accumulation**:

```
Era 1: baseline=0.72 → best=0.80 → commit [chain_of_thought]
Era 2: baseline=0.80 → best=0.88 → commit [chain_of_thought, max_iterations=10]
Era 3: baseline=0.88 → best=0.92 → commit [chain_of_thought, max_iterations=10, history]

Total improvement: 0.72 → 0.92 = 28%
```

Each era locks in proven improvements and explores further.

---

## 3. Experimental Results

### 3.1 Benchmark Setup

We evaluate on a **real agent optimization task**: a tool-using agent solving 25 math and fact-lookup problems via Ollama (qwen2.5:1.5b).

| Property | Value |
| -------- | ----- |
| Tasks | 25 (math calculations, fact lookups) |
| Parameters | 5 tunable settings |
| Metric | pass_rate (higher = better) |
| Baseline | 0.72 (72% of tasks solved) |
| Evaluation | ~9 seconds per config |

**Search space (72 possible configurations):**

| Parameter | Values |
| --------- | ------ |
| max_iterations | 3, 5, 10 |
| include_history | true, false |
| retry_with_feedback | true, false |
| use_chain_of_thought | true, false |
| tool_instructions | minimal, standard, detailed |

**Potential synergies:**
- `max_iterations=10` + `use_chain_of_thought` (more iterations to leverage reasoning)
- `include_history` + `retry_with_feedback` (history enables useful error feedback)

### 3.2 Results: Final Performance

**Table 1: Best Fitness Achieved (20 evals × 5 runs, higher = better)**

| Method | Average | Best | Worst |
| ------ | ------- | ---- | ----- |
| Baseline | 0.720 | - | - |
| Greedy | 0.896 | 0.920 | 0.880 |
| **cEvolve** | 0.896 | 0.920 | 0.840 |

**Both methods achieve the same average performance.** Given enough evaluations, both find good configurations. The difference is in *how fast* they get there.

### 3.3 Results: Convergence Speed

**Table 2: Evaluations to Reach Threshold**

| Threshold | Greedy | cEvolve | Speedup |
| --------- | ------ | ------- | ------- |
| ≥0.62 (easy) | 1.2 evals (100%) | 1.0 evals (100%) | 17% |
| ≥0.72 (baseline) | 1.6 evals (100%) | 1.6 evals (100%) | 0% |
| ≥0.82 (good) | 8.2 evals (100%) | **3.2 evals (100%)** | **61%** |

**cEvolve reaches good solutions (≥0.82) in 2.6× fewer evaluations.**

For expensive evaluations, this speedup compounds:

| Eval Time | Greedy to ≥0.82 | cEvolve to ≥0.82 | Savings |
| --------- | --------------- | ---------------- | ------- |
| 9 sec (this benchmark) | 74 sec | 29 sec | 45 sec |
| 2 min (LLM training) | 16 min | 6 min | 10 min |
| 10 min (full training run) | 82 min | 32 min | 50 min |

### 3.4 Results: Synergy Discovery

**Table 3: Gene Combinations Found in Best Configs**

| Combination | Greedy | cEvolve |
| ----------- | ------ | ------- |
| include_history + retry_with_feedback | 2/5 | 2/5 |
| max_iterations=10 + use_chain_of_thought | **0/5** | **2/5** |

**cEvolve discovered the `max_iterations=10 + use_chain_of_thought` synergy in 2 out of 5 runs. Greedy never found it.**

This is the key result: crossover creates novel combinations that sequential testing misses.

### 3.5 Case Study: Why Greedy Misses the Synergy

```
GREEDY SEARCH TRACE (Run 0):
─────────────────────────────────────────────
Eval 0:  baseline                         → 0.720
Eval 1:  try max_iterations=5             → 0.760 ✓ keep
Eval 2:  try max_iterations=10            → 0.720 ✗ reject  ← TESTED ALONE
Eval 3:  try include_history=true         → 0.800 ✓ keep
Eval 4:  try use_chain_of_thought=true    → 0.880 ✓ keep
...
Final: {max_iterations=5, include_history, use_chain_of_thought}

Greedy tested max_iterations=10 in isolation (eval 2) and rejected it.
It never tested max_iterations=10 + use_chain_of_thought together.
```

```
EVOLVE SEARCH TRACE (Run 3):
─────────────────────────────────────────────
Gen 0: Random population
  ind-0: baseline                         → 0.640
  ind-1: [max_iterations=10, cot=true]    → 0.800
  ind-2: [include_history, retry]         → 0.720
  ...

Gen 1: Crossover combines genes
  Parent A: ind-1 [max_iterations=10, cot=true]
  Parent B: ind-2 [include_history, retry]
  Child:    [max_iterations=10, cot=true, history, retry] → 0.920 ✓

Crossover combined max_iterations=10 with chain_of_thought,
creating a combination that greedy never tested.
```

### 3.6 Best Configurations Found

**Greedy's best config:**
```yaml
max_iterations: 5
include_history: true
retry_with_feedback: true
use_chain_of_thought: true
tool_instructions: minimal  # unchanged from baseline
```

**cEvolve's best config:**
```yaml
max_iterations: 10          # different!
include_history: false      # different!
retry_with_feedback: false  # different!
use_chain_of_thought: true
tool_instructions: minimal
```

Both achieved 0.920 pass_rate, but via **different paths**. cEvolve discovered that `max_iterations=10 + chain_of_thought` works well even *without* history—a configuration greedy's sequential testing would never explore.

---

## 4. When to Use cEvolve

### 4.1 cEvolve Excels When:

✅ **Parameters interact** - synergies and antagonisms exist
✅ **Evaluations are expensive** - minutes or hours each
✅ **Search space is large** - many ideas and variants
✅ **"Good enough" matters** - faster convergence saves compute
✅ **Parallel compute available** - evaluate population concurrently

### 4.2 Greedy May Suffice When:

⚠️ **Parameters are independent** - no interactions
⚠️ **One dominant improvement** - greedy finds it fast
⚠️ **Very few evaluations** - not enough for GA to evolve

### 4.3 Target Applications

#### Known Parameter Spaces

When you have predefined knobs to tune:

| Domain             | Example Ideas                                      |
| ------------------ | -------------------------------------------------- |
| **LLM Training**   | depth, batch size, learning rates, warmup schedule |
| **Compiler Flags** | optimization levels, inlining, vectorization       |
| **Infrastructure** | instance types, replicas, caching strategies       |

#### Open-Ended Code Optimization

When the LLM discovers what to optimize:

```
Target: src/api/search.py
Metric: latency_p99

The LLM analyzes the code and proposes ideas:
  - "Add caching for database queries"
  - "Use batch fetching instead of N+1 queries"
  - "Parallelize independent API calls"
  - "Add an index on the search column"

cEvolve tests combinations-maybe caching + batching
conflict, but caching + parallelization synergize.
```

This is closer to the original autoresearch vision: the LLM isn't picking from a menu, it's **discovering what's possible** by reading the code.

#### Agent Optimization

Optimizing AI agents, prompts, tool configurations, and context management:

```
Target: agent/config.yaml + agent/prompts/*.md + agent/context.yaml
Metric: eval_pass_rate (higher is better)

Ideas the LLM might propose:

  Prompts:
    - "Add chain-of-thought to the planning prompt"
    - "Add few-shot examples to the system prompt"
    - "Add a self-critique step before final answer"

  Config:
    - "Increase max_iterations from 5 to 10"
    - "Use a different tool selection strategy"
    - "Reduce temperature for tool calls"

  Context Management:
    - "Increase context window from 8k to 16k"
    - "Summarize conversation history after 10 turns"
    - "Prioritize recent tool outputs over old messages"
    - "Include full file contents vs. snippets only"
    - "Add semantic retrieval for relevant past context"

cEvolve finds which combinations of prompts, config,
and context strategies maximize eval performance.
```

#### The Common Pattern

All applications share the same structure:

```
1. TARGET: Code, config, or prompts to optimize
2. IDEAS:  Either predefined OR discovered by LLM
3. METRIC: Measurable outcome (latency, accuracy, pass rate)
4. SEARCH: cEvolve finds which combinations work together
```

The LLM's role scales from "implement these predefined ideas" to "discover and implement ideas from scratch."

---

## 5. Limitations

### What We're Claiming

- cEvolve converges faster to good solutions (61% fewer evals in our benchmark)
- Crossover discovers synergies that greedy misses (2/5 vs 0/5 runs)
- Final performance is equivalent given enough evaluations

### What We're NOT Claiming

- cEvolve always beats greedy (it doesn't for simple, independent parameters)
- These exact numbers generalize to all problems (results from one agent optimization task)
- Evolution is the only way to find synergies (exhaustive search works too, just slower)

### Experiment Limitations

- Small scale: 5 runs × 20 evals per method
- Single domain: agent optimization only
- LLM variance: Ollama outputs vary between runs, adding noise
- Limited synergies: only 2 potential synergies in search space

---

## 6. Conclusion

Greedy optimization tests ideas in isolation, missing how they interact. For complex optimization problems—agent tuning, LLM training, system configuration—this leaves performance on the table.

cEvolve combines LLM hypothesis generation with genetic algorithm exploration:

- **Population-based search** discovers synergies through crossover
- **Rethink mechanism** adapts the search based on results
- **Era-based accumulation** locks in wins and builds on them

On a real agent optimization benchmark, cEvolve:

- Achieves **equivalent final performance** to greedy
- Converges **61% faster** to good solutions (3.2 vs 8.2 evals)
- Discovers synergies **greedy never finds** (2/5 vs 0/5 runs)

For expensive optimization where evaluations cost real compute, **cEvolve's faster convergence translates directly to saved time and money.**

And with parallel evaluation, cEvolve can achieve further wall-clock speedups—population members can be evaluated concurrently, while greedy is inherently sequential.

---

## Implementation

Code available at [https://github.com/jnormore/cevolve](https://github.com/jnormore/cevolve)

---

## References

1. Karpathy, A. "Autoresearch." GitHub, 2024. https://github.com/karpathy/autoresearch
2. Zhang, Y. et al. "A Systematic Survey on Large Language Models for Evolutionary Optimization." arXiv:2509.08269, 2025.
3. Normore, J., Harding, S., & Banzhaf, W. "Computational Fluid Dynamics on GPUs for Genetic Programming Fitness Evaluation." GECCO, 2010.
