"""
cevolve stop - Finalize session.
"""

import shutil
from ..session import Session


def handle(args) -> dict:
    """Handle stop command."""
    
    session = Session.load(
        name=getattr(args, 'session', None),
        work_dir=getattr(args, 'work_dir', '.'),
    )
    
    result = session.stop()
    
    # Cleanup if requested
    if getattr(args, 'cleanup', False):
        shutil.rmtree(session.session_dir)
        result["cleaned_up"] = True
    
    return result
