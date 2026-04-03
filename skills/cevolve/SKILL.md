---
name: cevolve
description: Evolutionary code optimization using genetic algorithms. Use when asked to "run cevolve", "optimize code", "find best configuration", or "explore hyperparameter combinations".
---

# cevolve Skill

Evolutionary optimization via composable CLI commands. The GA engine handles population management, selection, and evolution. You handle idea generation and gene implementation.

## Commands

| Command | Purpose |
|---------|---------|
| `cevolve init` | Create session with ideas |
| `cevolve next` | Get next individual (genes to implement) |
| `cevolve eval` | Run benchmark, record result, revert |
| `cevolve record` | Record result (you ran benchmark) |
| `cevolve revert` | Revert file changes |
| `cevolve rethink` | Analyze results, modify ideas |
| `cevolve status` | Get session state |
| `cevolve stop` | Finalize session |

All commands support `--json` for machine-readable output.

## Setup Checklist

1. **Understand the goal** — What to optimize, which metric, files in scope
2. **Create a branch** — `git checkout -b cevolve/<goal>`
3. **Read source files deeply** — Understand architecture, data flow, allocations
4. **Profile** — See WHERE time/memory is spent. Don't guess.
5. **Form a hypothesis** — "This code is slow because..."
6. **Create `bench.sh`** — Must output `METRIC name=value`
7. **Create `SESSION.md`** — Document hypothesis and learnings
8. **Initialize session** — `cevolve init` with ideas that address hypothesis
9. **Loop** — `next` → implement → `eval` → repeat

## Workflow

### 1. Understand the Goal

Ask or infer:
- What to optimize (speed, memory, accuracy)
- Which metric to measure
- Which files are in scope

### 2. Profile First

Before proposing ideas, understand WHERE time/resources are spent:

```bash
# Example: profile Python code
python -m cProfile -s cumtime script.py
```

Form a hypothesis: "This code is slow because..."

### 3. Create Benchmark Script

Create `bench.sh` that outputs metrics:

```bash
#!/bin/bash
set -euo pipefail

# Fast pre-check (catch errors before full benchmark)
python -c "import mymodule"  # syntax check

# Run benchmark
python train.py

# Must output: METRIC name=value
echo "METRIC time_ms=1234.5"
echo "METRIC memory_mb=512"
```

**Design principles:**
- Fast pre-checks — catch syntax errors in <1s
- Stable measurements — multiple iterations, report median
- Keep it fast — every second is multiplied by dozens of evaluations

### 4. Create Session Document

Create `SESSION.md` to track learnings:

```markdown
# cEvolve: <goal>

## Objective
What we're optimizing and why.

## Hypothesis
"This code is slow because..."

## Profiler Findings
Where is time actually spent? What's the bottleneck?

## Metrics
- **Primary**: time_ms (lower is better)
- **Secondary**: memory_mb

## How to Run
`./bench.sh` — outputs METRIC lines

## Files in Scope
- `src/parser.py` — main parsing logic
- `src/tokenizer.py` — tokenization

## Off Limits
What must NOT be touched (APIs, tests, etc.)

## What's Working
Which ideas help? Why?

## Dead Ends
What didn't work? WHY didn't it work?
```

Update this after each rethink to capture learnings.

### 5. Initialize Session

```bash
cevolve init \
  --name "optimize-parser" \
  --idea "use_bytesio: Replace string concat with BytesIO buffer" \
  --idea "batch_size[16,32,64,128]: Number of items per batch" \
  --idea "precompile_regex: Compile regex patterns at module level" \
  --bench "./bench.sh" \
  --metric time_ms \
  --direction lower
```

**Idea formats:**
- Binary: `"name: description"` — on/off
- Variant: `"name[v1,v2,v3]: description"` — discrete choices

### 6. Evolution Loop

```bash
# Get next individual
cevolve next --json
```

Returns:
```json
{
  "status": "ready",
  "individual_id": "ind-abc123",
  "genes": {"use_bytesio": "on", "batch_size": "64", "precompile_regex": null},
  "active": [
    {"name": "use_bytesio", "value": "on", "description": "Replace string concat..."},
    {"name": "batch_size", "value": "64", "description": "Number of items..."}
  ],
  "is_baseline": false
}
```

**If `is_baseline: true`:** Don't modify any code. Just run eval.

**If `status: "converged"` or `status: "max_evals"`:** Evolution complete.

### 7. Implement Genes

**Rules:**
1. Implement ONLY active genes
2. For variant genes (`batch_size = 64`), implement that specific value
3. For binary genes (`use_bytesio = on`), implement the described optimization
4. Do NOT implement inactive genes

**Example:** If `batch_size = 64`:
```python
# Before
BATCH_SIZE = 32

# After  
BATCH_SIZE = 64
```

**Example:** If `use_bytesio = on`:
```python
# Before
result = ""
for item in items:
    result += process(item)

# After
from io import BytesIO
buffer = BytesIO()
for item in items:
    buffer.write(process(item))
result = buffer.getvalue()
```

**Baseline:** When `is_baseline: true`, do NOT modify any code. Just run eval.

### 8. Evaluate

**Option A:** Let cevolve run the benchmark:
```bash
cevolve eval --id ind-abc123
```
This runs the benchmark, records the result, and reverts all changes.

**Option B:** Run benchmark yourself, then record:
```bash
# Run your benchmark
./bench.sh
# Output: METRIC time_ms=42.5

# Record the result
cevolve record --id ind-abc123 --fitness 42.5

# Revert changes
cevolve revert
```

### 9. Repeat

Continue the loop:
```bash
cevolve next --json
# implement genes...
cevolve eval --id <id>
```

Until `status: "converged"` or `status: "max_evals"`.

### 10. Rethink (Periodic)

Every few evaluations, analyze what's working:

```bash
cevolve rethink
```

#### If Ideas Aren't Working

Don't just add more ideas at the same level. Ask:

1. **Is my hypothesis wrong?**
   - Re-read the source files
   - Re-run the profiler
   - What did I miss?

2. **Am I optimizing the wrong thing?**
   - Is the benchmark representative of production?
   - Am I measuring what matters?

3. **What would I need to change structurally to get 2x improvement?**
   - If small changes aren't helping, the architecture may be the bottleneck
   - Try something structurally different

#### Commit Best & Accumulate Wins

When one idea clearly dominates:

```bash
cevolve rethink --commit-best
```

This:
1. Makes the winning config the new baseline
2. Future evals build ON TOP of this improvement
3. You can now propose ideas that depend on that change existing

**Use when:**
- One idea gives big improvement (e.g., +50%)
- You want to explore optimizations that build on it
- You're moving from one optimization level to the next

#### Don't Thrash

Repeatedly trying variations of the same idea? Stop. Think harder about root cause.

#### After Rethink

Update `SESSION.md`:
- Revise hypothesis if needed
- Document WHY things didn't work
- Record insights for future agents

```bash
cevolve rethink \
  --add-idea "use_mmap: Memory-map large files" \
  --remove-idea "precompile_regex"
```

### 11. Finalize

Evolution stops when:
- No improvement for N evaluations (convergence)
- `max_evaluations` reached

When converged, `cevolve next` returns `status: "converged"` with the winning config.

```bash
cevolve stop
```

Then:
1. Re-implement the winning genes permanently
2. Run tests to verify nothing broke
3. Commit with a descriptive message

```bash
git add -A
git commit -m "perf: apply evolved optimizations (use_bytesio, batch_size=64)"
```

## Proposing Ideas

**The best ideas come from deep understanding, not from trying random variations.**

Before proposing ideas:
1. Read the source files
2. Study the profiling data
3. Reason about what the CPU is actually doing
4. Write a hypothesis: "This code is slow because..."

Your ideas should **address that hypothesis**. Don't list "possible optimizations" — propose changes that fix the root cause you identified.

### What Makes a Good Idea

- **Addresses your hypothesis** — not a random "this might help"
- **Specific** — points to WHERE and WHAT to change
- **Testable** — you'll know if it worked or not

```bash
--idea "use_bytesio: Replace string concat with BytesIO (addressing: O(n²) string building)"
--idea "cache_compiled[lru,dict,none]: Cache strategy for compiled patterns"
```

### What to Avoid

- Listing micro-optimizations because they're easy to think of
- Adding ideas "just in case" without understanding why they'd help
- Copying optimizations from other codebases without understanding if they apply

```bash
# BAD - no hypothesis
--idea "try_numpy: Maybe numpy is faster?"
--idea "optimize: Make it faster"
--idea "use_rust: Rewrite in Rust"  # Doesn't address specific bottleneck
```

## Example Session

```bash
# 1. Profile
python -m cProfile -s cumtime parser.py 2>&1 | head -20
# Shows: tokenize() takes 80% of time

# 2. Hypothesis
# "Tokenization is slow because we create new regex objects on each call"

# 3. Benchmark
cat > bench.sh << 'EOF'
#!/bin/bash
set -euo pipefail
python -c "
import time
from parser import parse
start = time.time()
for _ in range(1000):
    parse(test_input)
elapsed = (time.time() - start) * 1000
print(f'METRIC time_ms={elapsed:.1f}')
"
EOF
chmod +x bench.sh

# 4. Initialize
cevolve init \
  --name "optimize-tokenizer" \
  --idea "precompile_regex: Compile regex at module level" \
  --idea "use_re2: Use re2 instead of stdlib re" \
  --idea "token_cache[none,lru128,lru1024]: Cache tokenization results" \
  --bench "./bench.sh" \
  --metric time_ms

# 5. Loop
while true; do
  result=$(cevolve next --json)
  status=$(echo "$result" | jq -r '.status')
  
  if [ "$status" = "converged" ] || [ "$status" = "max_evals" ]; then
    echo "Done!"
    break
  fi
  
  id=$(echo "$result" | jq -r '.individual_id')
  
  # Implement genes based on $result...
  
  cevolve eval --id "$id"
done

# 6. Finalize
cevolve stop
```

## JSON Output Reference

### `cevolve next --json`

```json
{
  "status": "ready",
  "individual_id": "ind-abc123",
  "generation": 2,
  "genes": {
    "precompile_regex": "on",
    "use_re2": null,
    "token_cache": "lru128"
  },
  "active": [
    {"name": "precompile_regex", "value": "on", "description": "..."},
    {"name": "token_cache", "value": "lru128", "description": "..."}
  ],
  "inactive": ["use_re2"],
  "is_baseline": false
}
```

### `cevolve eval --id X --json`

```json
{
  "individual_id": "ind-abc123",
  "fitness": 42.5,
  "metrics": {"memory_mb": 128},
  "is_best": true,
  "improvement": "-15.3%",
  "evaluations": 12,
  "status": "continue"
}
```

### `cevolve status --json`

```json
{
  "session": "optimize-tokenizer",
  "evaluations": 12,
  "generation": 2,
  "era": 0,
  "best": {
    "id": "ind-abc123",
    "fitness": 42.5,
    "genes": {"precompile_regex": "on", "token_cache": "lru128"},
    "improvement": "-15.3%"
  },
  "baseline_fitness": 50.2,
  "converged": false,
  "evals_since_improvement": 3
}
```

## When NOT to Use cEvolve

cEvolve explores combinations of predefined ideas. It's NOT ideal for:

- **Pure discovery** — when you don't know what's wrong yet (profile first)
- **Sequential rewrites** — where change B only makes sense after A is committed
- **Architectural changes** — that require coordinated changes across many files

For those cases, do manual exploration first, THEN use cevolve to optimize.

## Tips

1. **Start with profiling** — don't guess at optimizations
2. **Baseline first** — the first evaluation is always baseline (no changes)
3. **Small, focused ideas** — each idea should be one specific change
4. **Variants for parameters** — use `[v1,v2,v3]` syntax for numeric/enum choices
5. **Always revert** — `eval` auto-reverts; with `record`, call `revert` explicitly
6. **JSON for scripting** — use `--json` flag for reliable parsing
7. **Update SESSION.md** — document what works and what doesn't
8. **Don't thrash** — if variations of the same idea keep failing, rethink the hypothesis
