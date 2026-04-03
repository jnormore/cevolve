"""
cevolve eval - Run benchmark and record result.

For composable CLI, this does NOT auto-revert. Agent handles git operations.
After eval, run `git checkout .` to revert changes before next individual.
"""

from ..session import Session


def handle(args) -> dict:
    """Handle eval command."""
    
    session = Session.load(
        name=getattr(args, 'session', None),
        work_dir=getattr(args, 'work_dir', '.'),
    )
    
    # Composable CLI: don't auto-revert, agent handles git
    result = session.eval(
        individual_id=args.id,
        timeout=getattr(args, 'timeout', None),
        revert=False,
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
        "note": "Run 'git checkout .' to revert changes before next individual",
    }
