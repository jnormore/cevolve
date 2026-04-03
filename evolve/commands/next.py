"""
cevolve next - Get the next individual to evaluate.
"""

from ..session import Session


def handle(args) -> dict:
    """Handle next command."""
    
    session = Session.load(
        name=getattr(args, 'session', None),
        work_dir=getattr(args, 'work_dir', '.'),
    )
    
    result = session.next()
    
    if result.status in ("converged", "max_evals"):
        return {
            "status": result.status,
            "best": result.best,
            "message": result.message,
        }
    
    if result.status == "rethink_required":
        return {
            "status": result.status,
            "best": result.best,
            "message": result.message,
            "instructions": (
                "1. Implement the genes from best.genes_to_implement\n"
                "2. git add -A && git commit -m 'cevolve: best config'\n"
                "3. cevolve rethink --commit-best"
            ),
        }
    
    return {
        "status": result.status,
        "individual_id": result.individual_id,
        "generation": result.generation,
        "genes": result.genes,
        "active": result.active,
        "inactive": result.inactive,
        "is_baseline": result.is_baseline,
    }
