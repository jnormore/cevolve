"""
cevolve status - Get session state.
"""

from ..session import Session


def handle(args) -> dict:
    """Handle status command."""
    
    session = Session.load(
        name=getattr(args, 'session', None),
        work_dir=getattr(args, 'work_dir', '.'),
    )
    
    result = session.status()
    
    # Add verbose details if requested
    if getattr(args, 'verbose', False):
        result["population"] = [
            {
                "id": ind.id,
                "genes": {k: v for k, v in ind.genes.items() if v is not None},
                "fitness": ind.fitness,
            }
            for ind in session.population
        ]
    
    return result
