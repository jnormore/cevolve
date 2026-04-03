# Agent Optimization

Optimize a simple tool-using agent to maximize eval pass rate.

**This is an open-ended optimization example** — the LLM can modify prompts directly, not just pick from predefined options.

## Requirements

This example uses [Ollama](https://ollama.ai) for local LLM inference.

### Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Start the Ollama server
ollama serve

# Pull a small model (in another terminal)
ollama pull qwen2.5:0.5b
```

### Available Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `smollm2:135m` | ~100MB | Very fast | Too weak (0%) |
| `qwen2.5:0.5b` | ~350MB | Fast | Weak (~56% baseline) |
| `qwen2.5:1.5b` | ~1GB | Medium | Decent |
| `llama3.2:1b` | ~1.3GB | Medium | Decent |

We recommend `qwen2.5:0.5b` — weak enough to benefit from optimization but can follow basic instructions.

## What Gets Optimized

### 1. System Prompt (open-ended)

The LLM can directly edit `prompts/system.md`:

```markdown
# prompts/system.md - LLM modifies this file

Answer the task. Use tools if needed.

{tools}

Respond with:
- TOOL: tool_name(argument) to use a tool  
- ANSWER: your final answer
```

The LLM might try:
- Adding chain-of-thought instructions
- Adding few-shot examples directly in the prompt
- Restructuring the format instructions
- Adding step-by-step guidance
- Being more explicit about when to use tools

### 2. Config Parameters

`config.yaml` has tunable parameters:

| Parameter | Description |
|-----------|-------------|
| `num_examples` | Few-shot examples (0, 2, 5) |
| `max_iterations` | Tool use attempts (3, 5, 10) |
| `include_history` | Include conversation history |

## The Agent

A simple tool-using agent:
1. Receives a task (math or fact question)
2. Can use tools: `calculator(expr)` and `lookup(key)`
3. Iterates until answer or max iterations
4. Returns final answer

## The Eval Suite

25 tasks:
- Math word problems (needs calculator)
- Fact lookups (needs lookup tool)
- Multi-step reasoning

## Run

```bash
# Make sure Ollama is running
ollama serve &

# Test single run
cd examples/agent-optimization
python train.py

# Optimize with cEvolve (LLM modifies prompts/system.md)
uv run cevolve run --target examples/agent-optimization \
  --metric pass_rate --direction higher --llm pi
```

## Baseline vs Optimized

**Baseline** (weak prompt, no examples):
```
pass_rate: 0.560
passed: 14
failed: 11
```

**After optimization** (LLM-improved prompt):
```
pass_rate: ???
```

## Files

```
examples/agent-optimization/
├── prompts/
│   └── system.md      ← LLM edits this directly
├── config.yaml        ← Tunable parameters
├── agent.py           ← Agent implementation
├── tools.py           ← Calculator + lookup
├── evals/
│   └── tasks.json     ← 25 eval tasks
├── train.py           ← Benchmark script
└── README.md
```

## Why Open-Ended?

Instead of picking from predefined prompts:
```yaml
system_prompt: cot  # [minimal, default, detailed, cot]
```

The LLM discovers what works:
```markdown
# prompts/system.md - LLM writes whatever helps

You are a precise assistant. Think step-by-step.

First, identify if this is a MATH or FACT question.
- For MATH: Always use calculator(expression)
- For FACT: Always use lookup(query)

Never guess. Always use a tool first, then answer.
...
```

This is closer to real agent optimization where you're not constrained to predefined options.
