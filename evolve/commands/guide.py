"""
cevolve guide - Print workflow guide for coding agents.
"""

GUIDE = """
# cevolve - Evolutionary Code Optimization

## Quick Start (for coding agents)

1. Create benchmark script that outputs: METRIC name=value
2. Initialize session with optimization ideas
3. Loop: get next → implement genes → eval → git revert → repeat
4. Periodically: commit best genes, call rethink to start new era

**Important:** The CLI does NOT do git operations. Agent handles git.

## Workflow

### 1. Setup

```bash
# Create bench.sh that outputs metrics
echo 'python train.py && echo "METRIC time_ms=123"' > bench.sh
chmod +x bench.sh

# Initialize session
cevolve init \\
  --name my-optimization \\
  --idea "use_cache: Set USE_CACHE = True in config.py" \\
  --idea "batch_size[16,32,64]: Set BATCH_SIZE to this value" \\
  --bench "./bench.sh" \\
  --metric time_ms
```

**Important:** Idea descriptions should say exactly WHAT to change and HOW.

### 2. Evolution Loop

```bash
cevolve next --json
```

Returns one of:
- `status: "ready"` → implement genes, eval, then revert
- `status: "rethink_required"` → must commit best genes and call rethink first
- `status: "converged"` or `"max_evals"` → done!

**If ready:**
```json
{
  "status": "ready",
  "individual_id": "ind-abc123",
  "is_baseline": false,
  "active": [
    {"name": "use_cache", "value": "on", "description": "Set USE_CACHE = True in config.py"},
    {"name": "batch_size", "value": "64", "description": "Set BATCH_SIZE to this value"}
  ]
}
```

- If `is_baseline: true` → don't change any code
- Otherwise → implement each gene in `active` based on description and value
- Run: `cevolve eval --id ind-abc123`
- **Then revert:** `git checkout .`

### 3. Git Operations (Agent's Responsibility)

The CLI does NOT run git commands. Agent must:

**After each eval:**
```bash
git checkout .   # Revert changes before next individual
```

**When `next` returns `rethink_required`:**
```bash
# next() blocks until you commit best and rethink
cevolve next --json
# Returns: {"status": "rethink_required", "best": {"genes_to_implement": [...]}}

# 1. Implement genes from best.genes_to_implement
# 2. git add -A && git commit -m "cevolve: apply best"
# 3. cevolve rethink --commit-best
# 4. Continue with: cevolve next
```

### 4. Gene Implementation

For each gene in `active`:
- `value: "on"` (binary) → apply the optimization described
- `value: "64"` (variant) → use that specific value

Example:
```python
# Gene: {"name": "batch_size", "value": "64", "description": "Set BATCH_SIZE to this value"}
# Before:
BATCH_SIZE = 32
# After:
BATCH_SIZE = 64
```

### 5. Commands Reference

| Command | Purpose |
|---------|---------|
| `cevolve init` | Create session with ideas |
| `cevolve next --json` | Get next individual |
| `cevolve eval --id X` | Run benchmark and record result |
| `cevolve record --id X --fitness Y` | Record result (you ran benchmark) |
| `cevolve status --json` | Get current state |
| `cevolve rethink` | Start new era (after you commit best) |
| `cevolve stop` | Finalize and generate report |

### 6. Tips

- Use `--json` for machine-readable output
- Benchmark should be fast (every second × dozens of evals)
- Write idea descriptions with implementation details
- Always `git checkout .` after eval before next individual
- Commit best genes periodically to accumulate wins
""".strip()


def handle(args) -> dict:
    """Handle guide command."""
    # For --json, return structured guide
    if getattr(args, 'json', False):
        return {
            "guide": GUIDE,
        }
    
    # For human output, just print
    print(GUIDE)
    return {"printed": True}
