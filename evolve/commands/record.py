"""
cevolve record - Record evaluation result without running benchmark.

Use when the agent runs the benchmark itself.
Call `cevolve revert` separately if needed.
"""

import json
from ..session import Session


def handle(args) -> dict:
    """Handle record command."""
    
    session = Session.load(
        name=getattr(args, 'session', None),
        work_dir=getattr(args, 'work_dir', '.'),
    )
    
    # Parse metrics
    metrics = {}
    if hasattr(args, 'metrics') and args.metrics:
        if isinstance(args.metrics, str):
            metrics = json.loads(args.metrics)
        else:
            metrics = args.metrics
    
    # Handle failure case
    if getattr(args, 'failed', False):
        fitness = None
        error = getattr(args, 'error', "Evaluation failed")
    else:
        if args.fitness is None:
            raise ValueError("--fitness required (or use --failed)")
        fitness = args.fitness
        error = None
    
    result = session.record(
        individual_id=args.id,
        fitness=fitness,
        metrics=metrics,
        error=error,
    )
    
    return {
        "individual_id": result.individual_id,
        "fitness": result.fitness,
        "metrics": result.metrics,
        "is_best": result.is_best,
        "improvement": result.improvement,
        "evaluations": result.evaluations,
        "status": result.status,
        "error": result.error,
        "note": "Changes NOT reverted. Call 'cevolve revert' if needed.",
    }
