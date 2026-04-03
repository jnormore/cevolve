"""
Generate analysis charts for evolve sessions.
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def generate_charts(runner: "GARunner") -> list[Path]:
    """Generate all analysis charts. Returns list of created file paths."""
    from evolve.persistence import get_session_dir
    
    session_dir = get_session_dir(runner.config.work_dir, runner.config.name)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    charts = []
    
    # Only generate if we have data
    eval_entries = [e for e in runner.history if "fitness" in e and e.get("fitness") is not None]
    if not eval_entries:
        return charts
    
    try:
        charts.append(generate_convergence_chart(runner, session_dir))
    except Exception as e:
        print(f"Warning: Could not generate convergence chart: {e}")
    
    try:
        charts.append(generate_idea_analysis_chart(runner, session_dir))
    except Exception as e:
        print(f"Warning: Could not generate idea analysis chart: {e}")
    
    try:
        charts.append(generate_synergy_matrix(runner, session_dir))
    except Exception as e:
        print(f"Warning: Could not generate synergy matrix: {e}")
    
    return [c for c in charts if c is not None]


def generate_convergence_chart(runner: "GARunner", session_dir: Path) -> Path:
    """Generate fitness convergence chart."""
    # Extract evaluation data
    evals = []
    fitnesses = []
    best_so_far = []
    generations = []
    
    current_best = float('inf') if runner.config.metric_direction == "lower" else float('-inf')
    is_lower_better = runner.config.metric_direction == "lower"
    
    for entry in runner.history:
        if "fitness" not in entry or entry.get("fitness") is None:
            continue
        
        fitness = entry["fitness"]
        evals.append(entry["evaluation"])
        fitnesses.append(fitness)
        generations.append(entry.get("generation", 0))
        
        if is_lower_better:
            current_best = min(current_best, fitness)
        else:
            current_best = max(current_best, fitness)
        best_so_far.append(current_best)
    
    if not evals:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all fitness values
    ax.scatter(evals, fitnesses, alpha=0.5, c=generations, cmap='viridis', 
               label='Individual fitness', s=50)
    
    # Plot best so far line
    ax.plot(evals, best_so_far, 'r-', linewidth=2, label='Best so far')
    
    # Mark baseline
    if fitnesses:
        ax.axhline(y=fitnesses[0], color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({fitnesses[0]:.4f})')
    
    ax.set_xlabel('Evaluation', fontsize=12)
    ax.set_ylabel(f'{runner.config.metric_name}', fontsize=12)
    ax.set_title('Fitness Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for generations
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Generation')
    
    chart_path = session_dir / "convergence.png"
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return chart_path


def generate_idea_analysis_chart(runner: "GARunner", session_dir: Path) -> Path:
    """Generate idea effectiveness analysis chart."""
    # Collect stats per idea
    idea_stats = defaultdict(lambda: {"on": [], "off": []})
    
    for entry in runner.history:
        if "fitness" not in entry or entry.get("fitness") is None:
            continue
        if "genes" not in entry:
            continue
        
        fitness = entry["fitness"]
        genes = entry["genes"]
        
        for idea_name in runner.ideas:
            value = genes.get(idea_name)
            if value is not None:
                idea_stats[idea_name]["on"].append(fitness)
            else:
                idea_stats[idea_name]["off"].append(fitness)
    
    if not idea_stats:
        return None
    
    # Calculate averages
    ideas = list(idea_stats.keys())
    on_avgs = []
    off_avgs = []
    on_counts = []
    off_counts = []
    
    for idea in ideas:
        on_fits = idea_stats[idea]["on"]
        off_fits = idea_stats[idea]["off"]
        on_avgs.append(np.mean(on_fits) if on_fits else 0)
        off_avgs.append(np.mean(off_fits) if off_fits else 0)
        on_counts.append(len(on_fits))
        off_counts.append(len(off_fits))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ideas))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, off_avgs, width, label='Idea OFF', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, on_avgs, width, label='Idea ON', color='lightgreen', alpha=0.8)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars1, off_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'n={count}', 
                ha='center', va='bottom', fontsize=8)
    for i, (bar, count) in enumerate(zip(bars2, on_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'n={count}', 
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Idea', fontsize=12)
    ax.set_ylabel(f'Average {runner.config.metric_name}', fontsize=12)
    ax.set_title('Idea Effectiveness Analysis', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ideas, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add indicator for which is better
    is_lower_better = runner.config.metric_direction == "lower"
    for i, (on, off) in enumerate(zip(on_avgs, off_avgs)):
        if on and off:
            if (is_lower_better and on < off) or (not is_lower_better and on > off):
                ax.annotate('✓', (i + width/2, on), ha='center', fontsize=12, color='green')
    
    fig.tight_layout()
    
    chart_path = session_dir / "idea_analysis.png"
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return chart_path


def generate_synergy_matrix(runner: "GARunner", session_dir: Path) -> Path:
    """Generate idea synergy/interaction matrix."""
    ideas = list(runner.ideas.keys())
    n = len(ideas)
    
    if n < 2:
        return None
    
    # Build co-occurrence fitness matrix
    # synergy[i][j] = average fitness when both idea i and j are ON
    synergy = np.zeros((n, n))
    counts = np.zeros((n, n))
    
    for entry in runner.history:
        if "fitness" not in entry or entry.get("fitness") is None:
            continue
        if "genes" not in entry:
            continue
        
        fitness = entry["fitness"]
        genes = entry["genes"]
        
        # Find which ideas are active
        active = [i for i, idea in enumerate(ideas) if genes.get(idea) is not None]
        
        # Update synergy matrix for all pairs
        for i in active:
            for j in active:
                synergy[i][j] += fitness
                counts[i][j] += 1
    
    # Calculate averages
    with np.errstate(divide='ignore', invalid='ignore'):
        synergy = np.where(counts > 0, synergy / counts, np.nan)
    
    # Also calculate baseline (nothing active) for comparison
    baseline_fits = []
    for entry in runner.history:
        if "fitness" not in entry or entry.get("fitness") is None:
            continue
        if "genes" not in entry:
            continue
        genes = entry["genes"]
        if all(v is None for v in genes.values()):
            baseline_fits.append(entry["fitness"])
    
    baseline = np.mean(baseline_fits) if baseline_fits else np.nan
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    is_lower_better = runner.config.metric_direction == "lower"
    cmap = 'RdYlGn' if is_lower_better else 'RdYlGn_r'
    
    im = ax.imshow(synergy, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(f'Average {runner.config.metric_name}')
    
    # Add labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(ideas, rotation=45, ha='right')
    ax.set_yticklabels(ideas)
    
    # Add value annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(synergy[i, j]):
                text = ax.text(j, i, f'{synergy[i, j]:.3f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if abs(synergy[i, j] - np.nanmean(synergy)) > np.nanstd(synergy) else 'black')
    
    ax.set_title(f'Idea Synergy Matrix\n(Baseline: {baseline:.4f})', fontsize=14)
    ax.set_xlabel('Idea', fontsize=12)
    ax.set_ylabel('Idea', fontsize=12)
    
    fig.tight_layout()
    
    chart_path = session_dir / "synergy_matrix.png"
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return chart_path
