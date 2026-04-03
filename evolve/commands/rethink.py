"""
cevolve rethink - Analyze results and modify ideas.
"""

import re
from ..core import Idea
from ..session import Session


def handle(args) -> dict:
    """Handle rethink command."""
    
    session = Session.load(
        name=getattr(args, 'session', None),
        work_dir=getattr(args, 'work_dir', '.'),
    )
    
    # Parse new ideas
    add_ideas = []
    if hasattr(args, 'add_ideas') and args.add_ideas:
        for idea_str in args.add_ideas:
            idea = _parse_inline_idea(idea_str)
            if idea:
                add_ideas.append(idea)
    
    # Get ideas to remove
    remove_ideas = getattr(args, 'remove_ideas', None) or []
    
    # Commit best flag
    commit_best = getattr(args, 'commit_best', False)
    
    result = session.rethink(
        add_ideas=add_ideas if add_ideas else None,
        remove_ideas=remove_ideas if remove_ideas else None,
        commit_best=commit_best,
    )
    
    return result


def _parse_inline_idea(idea_str: str) -> Idea:
    """Parse inline idea string."""
    # Try variant format
    match = re.match(r'^(\w+)\[([^\]]+)\]:\s*(.+)$', idea_str)
    if match:
        return Idea(
            name=match.group(1),
            description=match.group(3),
            variants=[v.strip() for v in match.group(2).split(',')],
        )
    
    # Try binary format
    match = re.match(r'^[\-\*]?\s*(\w+):\s*(.+)$', idea_str)
    if match:
        return Idea(
            name=match.group(1),
            description=match.group(2),
            variants=[],
        )
    
    return None
