"""
Benchmark execution and metric parsing.
"""

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    fitness: Optional[float]  # Primary metric value (None if failed)
    metrics: dict[str, float]  # All parsed metrics
    duration: float  # Seconds
    output: str  # Raw output
    error: Optional[str]  # Error message if failed


def parse_metrics(output: str) -> dict[str, float]:
    """
    Parse metrics from benchmark output.
    
    Supports two formats:
    1. "name: value" (e.g., "val_bpb: 1.234")
    2. "METRIC name=value" (e.g., "METRIC time_ms=45.2")
    """
    metrics = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Format 1: "name: value"
        match = re.match(r'^([\w.]+):\s*([\d.\-+eE]+)', line)
        if match:
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
            continue
        
        # Format 2: "METRIC name=value"
        match = re.match(r'^METRIC\s+([\w.]+)=([\d.\-+eE]+)', line)
        if match:
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    
    return metrics


def run_benchmark(
    command: str,
    work_dir: Path,
    metric_name: str,
    timeout: int = 600,
    log_callback: Callable[[str], None] = None,
) -> BenchmarkResult:
    """
    Run a benchmark command and extract metrics.
    
    Args:
        command: Shell command to run
        work_dir: Working directory
        metric_name: Primary metric to extract as fitness
        timeout: Timeout in seconds
        log_callback: Optional callback for progress logging
    
    Returns:
        BenchmarkResult with fitness, metrics, duration, output, and error
    """
    start = time.time()
    output_lines = []
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        last_progress = 0
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output_lines.append(line)
            
            elapsed = time.time() - start
            
            # Progress logging every 30 seconds
            if log_callback and elapsed - last_progress >= 30:
                log_callback(f"... {int(elapsed)}s elapsed")
                last_progress = elapsed
            
            # Timeout check
            if elapsed > timeout:
                process.kill()
                process.wait()
                return BenchmarkResult(
                    fitness=None,
                    metrics={},
                    duration=elapsed,
                    output=''.join(output_lines),
                    error=f"Timeout after {timeout}s",
                )
        
        output = ''.join(output_lines)
        duration = time.time() - start
        
        if process.returncode != 0:
            return BenchmarkResult(
                fitness=None,
                metrics={},
                duration=duration,
                output=output,
                error=f"Exit code {process.returncode}",
            )
        
        # Parse metrics
        metrics = parse_metrics(output)
        fitness = metrics.pop(metric_name, None)
        
        if fitness is None:
            return BenchmarkResult(
                fitness=None,
                metrics=metrics,
                duration=duration,
                output=output,
                error=f"Metric '{metric_name}' not found in output",
            )
        
        return BenchmarkResult(
            fitness=fitness,
            metrics=metrics,
            duration=duration,
            output=output,
            error=None,
        )
        
    except Exception as e:
        return BenchmarkResult(
            fitness=None,
            metrics={},
            duration=time.time() - start,
            output=''.join(output_lines),
            error=str(e),
        )
