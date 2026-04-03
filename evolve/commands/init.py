"""
cevolve init - Create a new session with ideas.
"""

import json
import re
from pathlib import Path

from ..core import Idea
from ..session import Session


def handle(args) -> dict:
    """Handle init command."""
    
    # Parse ideas
    ideas = []
    
    if args.ideas:
        # From file or JSON string
        if Path(args.ideas).exists():
            with open(args.ideas) as f:
                ideas_data = json.load(f)
        else:
            ideas_data = json.loads(args.ideas)
        
        for d in ideas_data:
            ideas.append(Idea(
                name=d["name"],
                description=d["description"],
                variants=d.get("variants") or [],
            ))
    
    if hasattr(args, 'inline_ideas') and args.inline_ideas:
        for idea_str in args.inline_ideas:
            idea = _parse_inline_idea(idea_str)
            if idea:
                ideas.append(idea)
    
    if not ideas:
        raise ValueError("No ideas provided. Use --ideas or --idea")
    
    # Parse secondary metrics
    secondary_metrics = []
    if hasattr(args, 'secondary_metrics') and args.secondary_metrics:
        for m in args.secondary_metrics:
            parts = m.split(':')
            metric = {"name": parts[0]}
            if len(parts) > 1:
                metric["unit"] = parts[1]
            if len(parts) > 2:
                metric["direction"] = parts[2]
            secondary_metrics.append(metric)
    
    # Create session
    session = Session.create(
        name=args.name,
        ideas=ideas,
        bench_command=args.bench,
        metric=args.metric,
        direction=args.direction,
        scope=getattr(args, 'scope', None),
        exclude=getattr(args, 'exclude', None),
        population_size=getattr(args, 'pop_size', 6),
        elitism=getattr(args, 'elitism', 2),
        mutation_rate=getattr(args, 'mutation_rate', 0.2),
        crossover_rate=getattr(args, 'crossover_rate', 0.7),
        max_evaluations=getattr(args, 'max_evals', None),
        convergence_evals=getattr(args, 'convergence_evals', None),
        rethink_interval=getattr(args, 'rethink_interval', 5),
        experiment_timeout=getattr(args, 'timeout', 600),
        revert_strategy=getattr(args, 'revert', 'git'),
        target_file=getattr(args, 'target', None),
        work_dir=getattr(args, 'work_dir', '.'),
        secondary_metrics=secondary_metrics if secondary_metrics else None,
    )
    
    # Calculate search space
    search_space = 1
    for idea in ideas:
        if idea.variants:
            search_space *= (len(idea.variants) + 1)
        else:
            search_space *= 2
    
    return {
        "session": session.config.name,
        "status": "initialized",
        "population_size": session.config.population_size,
        "ideas": len(ideas),
        "search_space": search_space,
        "session_dir": str(session.session_dir),
    }


def _parse_inline_idea(idea_str: str) -> Idea:
    """
    Parse inline idea string.
    
    Formats:
        "name: description"
        "name[v1,v2,v3]: description"
    """
    # Try variant format
    match = re.match(r'^(\w+)\[([^\]]+)\]:\s*(.+)$', idea_str)
    if match:
        name = match.group(1)
        variants = [v.strip() for v in match.group(2).split(',')]
        description = match.group(3)
        return Idea(name=name, description=description, variants=variants)
    
    # Try binary format
    match = re.match(r'^(\w+):\s*(.+)$', idea_str)
    if match:
        name = match.group(1)
        description = match.group(2)
        return Idea(name=name, description=description, variants=[])
    
    return None
