"""
cevolve sessions - List or switch sessions.
"""

import json
from pathlib import Path


def handle(args) -> dict:
    """Handle sessions command."""
    
    work_dir = Path(getattr(args, 'work_dir', '.'))
    cevolve_dir = work_dir / ".cevolve"
    
    if not cevolve_dir.exists():
        return {"sessions": [], "message": "No sessions found"}
    
    # Switch if requested
    if hasattr(args, 'switch') and args.switch:
        session_dir = cevolve_dir / args.switch
        if not session_dir.exists():
            raise ValueError(f"Session '{args.switch}' not found")
        
        current_file = cevolve_dir / ".current"
        current_file.write_text(args.switch)
        
        return {"switched": args.switch}
    
    # List sessions
    current = None
    current_file = cevolve_dir / ".current"
    if current_file.exists():
        current = current_file.read_text().strip()
    
    sessions = []
    for d in sorted(cevolve_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue
        
        state_file = d / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            sessions.append({
                "name": d.name,
                "evaluations": state.get("evaluations", 0),
                "generation": state.get("generation", 0),
                "current": d.name == current,
            })
        else:
            sessions.append({
                "name": d.name,
                "evaluations": 0,
                "generation": 0,
                "current": d.name == current,
            })
    
    return {"sessions": sessions, "current": current}
