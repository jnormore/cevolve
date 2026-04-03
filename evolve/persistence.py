"""
Session persistence - save/load evolve state to/from disk.

Structure:
.cevolve/<session-name>/
  config.json       # Configuration
  ideas.json        # Current idea pool
  population.json   # Population state
  history.jsonl     # Append-only evaluation log
  RESULTS.md        # Summary (generated on stop)
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import GARunner, Idea, Individual, Config


def get_session_dir(work_dir: Path, name: str) -> Path:
    """Get session directory path."""
    return work_dir / ".cevolve" / name


def save_config(runner: "GARunner") -> None:
    """Save configuration to disk."""
    session_dir = get_session_dir(runner.config.work_dir, runner.config.name)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    config_data = {
        "name": runner.config.name,
        "population_size": runner.config.population_size,
        "max_evaluations": runner.config.max_evaluations,
        "elitism": runner.config.elitism,
        "mutation_rate": runner.config.mutation_rate,
        "crossover_rate": runner.config.crossover_rate,
        "experiment_timeout": runner.config.experiment_timeout,
        "train_command": runner.config.train_command,
        "work_dir": str(runner.config.work_dir),
        "target_file": runner.config.target_file,
        "metric_name": runner.config.metric_name,
        "metric_direction": runner.config.metric_direction,
        "metric_unit": runner.config.metric_unit,
        "secondary_metrics": [
            {"name": m.name, "unit": m.unit, "direction": m.direction}
            for m in runner.config.secondary_metrics
        ],
        "rethink_interval": runner.config.rethink_interval,
        "convergence_evals": runner.config.convergence_evals,
        "num_ideas": runner.config.num_ideas,
    }
    
    (session_dir / "config.json").write_text(json.dumps(config_data, indent=2))


def save_ideas(runner: "GARunner") -> None:
    """Save ideas to disk."""
    session_dir = get_session_dir(runner.config.work_dir, runner.config.name)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    ideas_data = {
        name: {
            "name": idea.name,
            "description": idea.description,
            "variants": idea.variants,
        }
        for name, idea in runner.ideas.items()
    }
    
    (session_dir / "ideas.json").write_text(json.dumps(ideas_data, indent=2))


def save_population(runner: "GARunner") -> None:
    """Save population state to disk."""
    session_dir = get_session_dir(runner.config.work_dir, runner.config.name)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    def serialize_individual(ind: "Individual") -> dict:
        return {
            "id": ind.id,
            "genes": ind.genes,
            "fitness": ind.fitness if ind.fitness != float('inf') else "inf",
            "metrics": ind.metrics,
            "generation": ind.generation,
            "parents": list(ind.parents) if ind.parents else None,
        }
    
    pop_data = {
        "generation": runner.generation,
        "evaluations": runner.evaluations,
        "era": runner.era,
        "last_rethink": runner.last_rethink,
        "best_at_eval": runner.best_at_eval,
        "best": serialize_individual(runner.best) if runner.best else None,
        "absolute_best": serialize_individual(runner.absolute_best) if runner.absolute_best else None,
        "current_individual": runner.current_individual,
        "population": [serialize_individual(ind) for ind in runner.population],
    }
    
    (session_dir / "population.json").write_text(json.dumps(pop_data, indent=2))


def save_state(runner: "GARunner") -> None:
    """Save full state to disk."""
    save_config(runner)
    save_ideas(runner)
    save_population(runner)
    save_history(runner)


def save_history(runner: "GARunner") -> None:
    """Save full history to disk (overwrites existing)."""
    session_dir = get_session_dir(runner.config.work_dir, runner.config.name)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    history_path = session_dir / "history.jsonl"
    with open(history_path, "w") as f:
        for entry in runner.history:
            # Handle infinity
            if "fitness" in entry and entry.get("fitness") == float('inf'):
                entry = entry.copy()
                entry["fitness"] = None
            f.write(json.dumps(entry) + "\n")


def append_history(runner: "GARunner", entry: dict) -> None:
    """Append entry to history file."""
    session_dir = get_session_dir(runner.config.work_dir, runner.config.name)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle infinity
    if "fitness" in entry and entry["fitness"] == float('inf'):
        entry = entry.copy()
        entry["fitness"] = None
    
    history_path = session_dir / "history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_summary(runner: "GARunner") -> Path:
    """Save summary to RESULTS.md."""
    session_dir = get_session_dir(runner.config.work_dir, runner.config.name)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    summary = runner.generate_summary()
    summary_path = session_dir / "RESULTS.md"
    summary_path.write_text(summary)
    
    return summary_path


def load_state(work_dir: Path, name: str) -> dict | None:
    """Load state from disk. Returns dict with config, ideas, population data."""
    session_dir = get_session_dir(work_dir, name)
    
    if not session_dir.exists():
        return None
    
    try:
        config_path = session_dir / "config.json"
        ideas_path = session_dir / "ideas.json"
        pop_path = session_dir / "population.json"
        history_path = session_dir / "history.jsonl"
        
        if not config_path.exists() or not ideas_path.exists() or not pop_path.exists():
            return None
        
        config_data = json.loads(config_path.read_text())
        ideas_data = json.loads(ideas_path.read_text())
        pop_data = json.loads(pop_path.read_text())
        
        # Load history
        history = []
        if history_path.exists():
            for line in history_path.read_text().strip().split("\n"):
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        # Convert infinity back
        def restore_infinity(d: dict) -> dict:
            if d and d.get("fitness") == "inf":
                d = d.copy()
                d["fitness"] = float('inf')
            return d
        
        if pop_data.get("best"):
            pop_data["best"] = restore_infinity(pop_data["best"])
        if pop_data.get("absolute_best"):
            pop_data["absolute_best"] = restore_infinity(pop_data["absolute_best"])
        pop_data["population"] = [restore_infinity(p) for p in pop_data["population"]]
        
        return {
            "config": config_data,
            "ideas": ideas_data,
            "population": pop_data,
            "history": history,
        }
        
    except Exception:
        return None


def list_sessions(work_dir: Path) -> list[str]:
    """List available session names."""
    base_dir = work_dir / ".cevolve"
    if not base_dir.exists():
        return []
    
    sessions = []
    for path in base_dir.iterdir():
        if path.is_dir() and (path / "config.json").exists():
            sessions.append(path.name)
    
    return sorted(sessions)


def delete_session(work_dir: Path, name: str) -> bool:
    """Delete a session directory."""
    import shutil
    
    session_dir = get_session_dir(work_dir, name)
    if session_dir.exists():
        shutil.rmtree(session_dir)
        return True
    return False
