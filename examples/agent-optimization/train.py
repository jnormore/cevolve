#!/usr/bin/env python3
"""
Benchmark script for agent optimization.

Runs the agent on eval tasks and outputs pass_rate metric.

Usage:
    python train.py
    
Output:
    ---
    pass_rate: 0.720
    passed: 18
    failed: 7
    total: 25
    avg_iterations: 2.4
"""

import json
import yaml
import sys
from pathlib import Path

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from agent import run_agent


import unicodedata

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    # Convert to lowercase, strip whitespace
    normalized = answer.lower().strip()
    
    # Remove accents (Brasília -> Brasilia)
    normalized = unicodedata.normalize('NFD', normalized)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Remove common suffixes/prefixes
    normalized = normalized.replace("$", "").replace("dollars", "")
    normalized = normalized.replace("degrees", "").replace("celsius", "")
    normalized = normalized.replace("mph", "").replace("miles per hour", "")
    
    # Handle numeric formatting
    try:
        # Try to parse as number and normalize
        num = float(normalized.replace(",", ""))
        if num == int(num):
            return str(int(num))
        return f"{num:.2f}".rstrip("0").rstrip(".")
    except ValueError:
        pass
    
    return normalized


def check_answer(got: str | None, expected: str) -> bool:
    """Check if the answer is correct."""
    if got is None:
        return False
    
    got_norm = normalize_answer(got)
    expected_norm = normalize_answer(expected)
    
    # Exact match
    if got_norm == expected_norm:
        return True
    
    # Check if expected is contained in got (for text answers)
    if expected_norm in got_norm:
        return True
    
    # Check numeric equality with tolerance
    try:
        got_num = float(got_norm.replace(",", ""))
        expected_num = float(expected_norm.replace(",", ""))
        if abs(got_num - expected_num) < 0.01:
            return True
    except ValueError:
        pass
    
    return False


def run_benchmark(config: dict, tasks: list, verbose: bool = True) -> dict:
    """
    Run benchmark on all tasks.
    
    Returns:
        dict with pass_rate, passed, failed, total, avg_iterations
    """
    passed = 0
    failed = 0
    total_iterations = 0
    
    for task in tasks:
        task_id = task["id"]
        task_text = task["task"]
        expected = task["answer"]
        
        if verbose:
            print(f"  Task {task_id}: {task_text[:50]}...", end=" ", flush=True)
        
        answer, iterations = run_agent(task_text, config)
        total_iterations += iterations
        
        if check_answer(answer, expected):
            passed += 1
            if verbose:
                print(f"✓ ({answer})")
        else:
            failed += 1
            if verbose:
                print(f"✗ (got: {answer}, expected: {expected})")
    
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0
    avg_iterations = total_iterations / total if total > 0 else 0
    
    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "failed": failed,
        "total": total,
        "avg_iterations": avg_iterations
    }


def main():
    # Load config
    config_path = SCRIPT_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    
    # Load tasks
    tasks_path = SCRIPT_DIR / "evals" / "tasks.json"
    tasks = json.loads(tasks_path.read_text())
    
    print(f"Agent Optimization Benchmark")
    print(f"=" * 50)
    print(f"Config:")
    print(f"  model: {config.get('model', 'qwen2.5:0.5b')}")
    print(f"  prompt: prompts/system.md")
    print(f"  num_examples: {config.get('num_examples', 0)}")
    print(f"  max_iterations: {config.get('max_iterations', 3)}")
    print(f"  include_history: {config.get('include_history', False)}")
    print(f"  temperature: {config.get('temperature', 0.7)}")
    print(f"  retry_with_feedback: {config.get('retry_with_feedback', False)}")
    print(f"  use_chain_of_thought: {config.get('use_chain_of_thought', False)}")
    print(f"=" * 50)
    print(f"Running {len(tasks)} tasks...")
    print()
    
    results = run_benchmark(config, tasks, verbose=True)
    
    # Output in cevolve metric format
    print()
    print("---")
    print(f"pass_rate: {results['pass_rate']:.3f}")
    print(f"passed: {results['passed']}")
    print(f"failed: {results['failed']}")
    print(f"total: {results['total']}")
    print(f"avg_iterations: {results['avg_iterations']:.1f}")


if __name__ == "__main__":
    main()
