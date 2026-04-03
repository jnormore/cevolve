"""
cevolve run - Primary CLI for humans.

Full evolution workflow with built-in LLM for:
- Discovering optimization ideas from code
- Implementing genes (code changes) for each individual
- Analyzing results and suggesting new ideas (rethink)

Extensions like pi-evolve use the composable commands instead,
bringing their own LLM/agent logic.
"""

import subprocess
import sys
import time
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..core import Idea
from ..session import Session


def handle(args) -> dict:
    """Handle run command."""
    from rich.console import Console
    console = Console()
    
    work_dir = Path(getattr(args, 'work_dir', '.'))
    
    # Determine target files (supports multiple)
    targets = getattr(args, 'targets', None) or ["train.py"]
    target_files = []
    target_contents = {}
    
    for target in targets:
        target_path = work_dir / target
        if not target_path.exists():
            console.print(f"[red]Error: {target_path} not found[/red]")
            sys.exit(1)
        target_files.append(target)
        target_contents[target] = target_path.read_text()
    
    # Combined code for idea discovery
    code = "\n\n".join(f"# === {f} ===\n{c}" for f, c in target_contents.items())
    
    # Setup LLM
    dry_run = getattr(args, 'dry_run', False)
    if dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")
        llm_call = _mock_llm
        ideas = _mock_ideas()
    else:
        llm_call = _make_llm_call(args.llm)
        console.print(f"[cyan]Discovering ideas (using {args.llm})...[/cyan]")
        ideas = discover_ideas(code, llm_call)
    
    # Display ideas
    console.print(f"\n[bold]Ideas ({len(ideas)}):[/bold]")
    for idea in ideas:
        if idea.variants:
            console.print(f"  • {idea.name}: {idea.description}")
            console.print(f"    [dim]variants: {', '.join(idea.variants)}[/dim]")
        else:
            console.print(f"  • {idea.name}: {idea.description} [dim](binary)[/dim]")
    console.print()
    
    # Session name
    name = getattr(args, 'name', None) or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    
    # Bench command (default to first target if it's a .py file)
    default_bench = f"uv run python {target_files[0]}" if target_files[0].endswith('.py') else "./bench.sh"
    bench = getattr(args, 'bench', None) or default_bench
    
    # Create session (single or multi-file mode)
    if len(target_files) > 1:
        revert_strategy = "multi"
        revert_kwargs = {"target_files": target_files}
    else:
        revert_strategy = "single"
        revert_kwargs = {"target_file": target_files[0]}
    
    session = Session.create(
        name=name,
        ideas=ideas,
        bench_command=bench,
        metric=args.metric,
        direction=args.direction,
        population_size=getattr(args, 'pop_size', 6),
        max_evaluations=getattr(args, 'max_evals', 20),
        rethink_interval=getattr(args, 'rethink', 5),
        revert_strategy=revert_strategy,
        work_dir=str(work_dir),
        **revert_kwargs,
    )
    
    # Store LLM context and flags (multi-file support)
    session._llm_call = llm_call
    session._target_files = target_files
    session._target_contents = target_contents  # Original contents for all files
    session._original_code = code  # Combined for idea discovery
    session._work_dir = work_dir
    session._dry_run = dry_run
    
    # Run
    no_tui = getattr(args, 'no_tui', False)
    if no_tui:
        result = _run_plain(session, console, dry_run)
    else:
        result = _run_with_tui(session, console, dry_run)
    
    return result


def _run_plain(session: Session, console, dry_run: bool = False) -> dict:
    """Run evolution loop in plain mode."""
    
    def log(msg: str):
        elapsed = time.time() - session.start_time
        mins, secs = divmod(int(elapsed), 60)
        console.print(f"[{mins:02d}:{secs:02d}] {msg}")
    
    session.log_callback = log
    
    log(f"Starting: {len(session.ideas)} ideas, pop={session.config.population_size}")
    
    while True:
        # Get next
        result = session.next()
        
        if result.status in ("converged", "max_evals"):
            log(f"🎯 {result.status.upper()}: {result.message}")
            if result.best:
                log(f"   Best: {result.best['fitness']} ({result.best['improvement']})")
            break
        
        # Handle rethink_required from next() - do rethink and continue
        if result.status == "rethink_required":
            log("\n=== RETHINK ===")
            _do_rethink(session, log)
            continue
        
        log(f"Evaluating {result.individual_id}: {_describe_genes(result.genes)}")
        
        # Implement genes (LLM)
        if not result.is_baseline and not dry_run:
            _implement_genes(session, result, log)
        
        # Eval
        eval_result = session.eval(result.individual_id)
        
        if eval_result.error:
            log(f"  [red]Error: {eval_result.error}[/red]")
        elif eval_result.is_best:
            log(f"  [green]NEW BEST: {eval_result.fitness}[/green]")
        else:
            log(f"  Fitness: {eval_result.fitness}")
        
        # Note: rethink is triggered by next() returning rethink_required status
    
    # Stop and print summary
    final = session.stop()
    _print_final_summary(session, final, console)
    
    return final


def _run_with_tui(session: Session, console, dry_run: bool = False) -> dict:
    """Run with TUI."""
    import threading
    
    try:
        from ..tui import EvolveTUI
    except ImportError:
        console.print("[yellow]TUI not available, using plain mode[/yellow]")
        return _run_plain(session, console, dry_run)
    
    app = EvolveTUI(list(session.ideas.values()))
    
    def run_loop():
        time.sleep(0.5)
        
        def log(msg: str):
            elapsed = time.time() - session.start_time
            mins, secs = divmod(int(elapsed), 60)
            formatted = f"[{mins:02d}:{secs:02d}] {msg}"
            app.call_from_thread(app.add_log, formatted)
            app.call_from_thread(app.update_session, session)
        
        session.log_callback = log
        
        try:
            while True:
                result = session.next()
                
                if result.status in ("converged", "max_evals"):
                    log(f"🎯 {result.status.upper()}")
                    break
                
                # Handle rethink_required from next()
                if result.status == "rethink_required":
                    log("\n=== RETHINK ===")
                    _do_rethink(session, log)
                    continue
                
                log(f"Evaluating: {_describe_genes(result.genes)}")
                
                if not result.is_baseline and not dry_run:
                    _implement_genes(session, result, log)
                
                eval_result = session.eval(result.individual_id)
                
                if eval_result.is_best:
                    log(f"[green]NEW BEST: {eval_result.fitness}[/green]")
                else:
                    log(f"Fitness: {eval_result.fitness}")
                
                # Note: rethink triggered by next() returning rethink_required
            
            final = session.stop()
            session._final_result = final
            
        except Exception as e:
            log(f"[red]Error: {e}[/red]")
        finally:
            app.call_from_thread(app.exit)
    
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    app.run()
    thread.join(timeout=2.0)
    
    # Print summary after TUI exits
    final = getattr(session, '_final_result', session.stop())
    _print_final_summary(session, final, console)
    
    return final


# =============================================================================
# LLM Integration
# =============================================================================

def _make_llm_call(cli: str) -> Callable[[str], str]:
    """Create LLM call function."""
    def llm_call(prompt: str) -> str:
        cmd = ["claude", "-p"] if cli == "claude" else ["pi"]
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"LLM failed: {result.stderr}")
        return result.stdout.strip()
    return llm_call


def _mock_llm(prompt: str) -> str:
    return "# No changes"


def _mock_ideas() -> list[Idea]:
    return [
        Idea("depth", "Number of layers", variants=["4", "6", "8"]),
        Idea("batch_size", "Batch size", variants=["16", "32", "64"]),
        Idea("use_cache", "Enable caching", variants=[]),
    ]


def discover_ideas(code: str, llm_call: Callable) -> list[Idea]:
    """Have LLM analyze code and suggest optimization ideas."""
    
    # Check if this looks like multi-file content
    is_multifile = "# ===" in code and ".md ===" in code
    
    if is_multifile:
        file_instruction = """You can modify ANY of these files to optimize the metric:
- Markdown/prompt files: Rewrite prompts, add instructions, change formatting
- YAML config: Adjust tunable settings
- Python code: Change parameters, logic, algorithms

For prompt files, suggest ideas like:
- prompt_style: How the prompt is structured (minimal, detailed, step_by_step)
- add_examples: Include few-shot examples in the prompt
- tool_instructions: How tools are described (brief, detailed, with_examples)
"""
    else:
        file_instruction = ""
    
    prompt = f"""Analyze these files and suggest optimization ideas to explore.
{file_instruction}
For each idea, specify the possible variants (values to try).

Files:
```
{code[:8000]}
```

IMPORTANT: Output ONLY the ideas in the exact format below. No markdown, no headers, no explanations, no summaries.

Output format (one idea per block):
idea_name: description of what this controls
  variants: value1, value2, value3, value4

For binary ideas (just on/off), omit the variants line.

Example output:
depth: Number of transformer layers
  variants: 4, 6, 8, 10, 12

activation: MLP activation function
  variants: relu_squared, gelu, swish

use_gradient_clipping: Add gradient clipping for stability

Rules:
- idea_name must be a valid Python identifier (letters, numbers, underscores only, no leading numbers)
- No markdown formatting (no **, no `, no headers)
- No explanatory text before or after the ideas
- 4-10 ideas maximum

Output only the ideas:"""
    
    response = llm_call(prompt)
    return _parse_ideas_response(response)


def _parse_ideas_response(response: str) -> list[Idea]:
    """Parse ideas from LLM response."""
    ideas = []
    lines = response.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Skip markdown headers and formatting
        if line.startswith('**') or line.startswith('##') or line.startswith('- ') or line.startswith('* '):
            i += 1
            continue
        
        # Skip lines that look like explanatory text
        skip_patterns = ['example', 'note:', 'important:', 'output', 'format', 'rules', 'summary', 'tradeoff']
        if any(p in line.lower() for p in skip_patterns):
            i += 1
            continue
        
        if ':' in line and not line.lower().startswith('variants'):
            parts = line.split(':', 1)
            # Clean name: remove markdown artifacts, backticks, leading underscores
            name = parts[0].strip()
            name = re.sub(r'[`*\[\]\(\)]', '', name)  # Remove markdown chars
            name = name.replace(' ', '_').replace('-', '_').lower()
            name = name.lstrip('_')  # Remove leading underscores
            name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
            
            desc = parts[1].strip()
            # Skip if description is just markdown or very short
            desc_clean = re.sub(r'[`*]', '', desc).strip()
            
            # Validate: name must be a valid Python identifier
            if not name or not desc_clean or len(name) > 30:
                i += 1
                continue
            
            # Check if name is a valid Python identifier
            if not re.match(r'^[a-z][a-z0-9_]*$', name):
                i += 1
                continue
            
            # Skip names that are clearly not optimization ideas
            bad_names = ['binary', 'add', 'variants', 'rationale', 'summary', 'example', 'output', 'format']
            if name in bad_names:
                i += 1
                continue
            
            variants = []
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip().lower()
                if next_line.startswith('variants:'):
                    variant_str = lines[i + 1].strip().split(':', 1)[1].strip()
                    variants = [v.strip() for v in variant_str.split(',') if v.strip()]
                    i += 1
            
            ideas.append(Idea(name=name, description=desc_clean, variants=variants))
        
        i += 1
    
    return ideas


def _implement_genes(session: Session, result, log: Callable) -> None:
    """Use LLM to implement genes across multiple target files."""
    if not hasattr(session, '_llm_call') or not session._llm_call:
        return
    
    changes = []
    for gene in result.active:
        if gene['value'] == 'on':
            changes.append(f"- {gene['name']}: {gene['description']}")
        else:
            changes.append(f"- {gene['name']} = {gene['value']}: {gene['description']}")
    
    if not changes:
        return
    
    # Build file listing for multi-file support
    target_files = getattr(session, '_target_files', [])
    target_contents = getattr(session, '_target_contents', {})
    
    if len(target_files) > 1:
        # Multi-file mode
        files_section = "\n\n".join(
            f"### FILE: {f}\n```\n{target_contents[f]}\n```"
            for f in target_files
        )
        file_format = """For each edit, specify the file:

### FILE: path/to/file
<<<<<<< SEARCH
exact code to find
=======
replacement code
>>>>>>> REPLACE"""
    else:
        # Single-file mode (backwards compatible)
        files_section = f"```\n{session._original_code}\n```"
        file_format = """<<<<<<< SEARCH
exact code to find
=======
replacement code
>>>>>>> REPLACE"""
    
    prompt = f"""Apply these changes to the code:

{chr(10).join(changes)}

Output your changes as SEARCH/REPLACE blocks. Each block:
- Must match the original code EXACTLY (including whitespace)
- Will replace that exact text with your new version

Format:
{file_format}

Only output the blocks, no other text.

Files:
{files_section}"""
    
    log(f"  Calling LLM for edits...")
    response = session._llm_call(prompt)
    
    edits = _parse_edits_multifile(response, target_files)
    log(f"  Applying {len(edits)} edit(s)")
    
    # Apply edits to each file
    modified_contents = dict(target_contents)  # Copy
    for file_path, search, replace in edits:
        if file_path in modified_contents:
            content = modified_contents[file_path]
            if search in content:
                modified_contents[file_path] = content.replace(search, replace, 1)
            else:
                log(f"  [yellow]Warning: Could not find search text in {file_path}[/yellow]")
        else:
            log(f"  [yellow]Warning: Unknown file {file_path}[/yellow]")
    
    # Write all modified files
    work_dir = getattr(session, '_work_dir', Path('.'))
    if isinstance(work_dir, str):
        work_dir = Path(work_dir)
    
    for file_path, content in modified_contents.items():
        (work_dir / file_path).write_text(content)


def _parse_edits(response: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks (single-file, backwards compatible)."""
    edits = []
    parts = response.split("<<<<<<< SEARCH")
    
    for part in parts[1:]:
        if "=======" not in part or ">>>>>>> REPLACE" not in part:
            continue
        try:
            search_rest = part.split("=======", 1)
            search = search_rest[0].strip()
            replace_rest = search_rest[1].split(">>>>>>> REPLACE", 1)
            replace = replace_rest[0].strip()
            if search:
                edits.append((search, replace))
        except (IndexError, ValueError):
            continue
    
    return edits


def _parse_edits_multifile(response: str, target_files: list[str]) -> list[tuple[str, str, str]]:
    """Parse SEARCH/REPLACE blocks with file specifiers.
    
    Returns list of (file_path, search, replace) tuples.
    """
    edits = []
    
    # Default to first target file if only one
    default_file = target_files[0] if target_files else "train.py"
    current_file = default_file
    
    # Split by FILE markers or SEARCH markers
    lines = response.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for file marker: ### FILE: path/to/file
        if line.startswith('### FILE:') or line.startswith('FILE:'):
            file_spec = line.split(':', 1)[1].strip()
            # Match against known target files
            for tf in target_files:
                if tf in file_spec or file_spec in tf or file_spec == Path(tf).name:
                    current_file = tf
                    break
            i += 1
            continue
        
        # Check for SEARCH block start
        if '<<<<<<< SEARCH' in line:
            # Find the content between SEARCH and =======
            search_lines = []
            i += 1
            while i < len(lines) and '=======' not in lines[i]:
                search_lines.append(lines[i])
                i += 1
            
            search = '\n'.join(search_lines).strip()
            
            # Find content between ======= and REPLACE
            replace_lines = []
            i += 1  # Skip =======
            while i < len(lines) and '>>>>>>> REPLACE' not in lines[i]:
                replace_lines.append(lines[i])
                i += 1
            
            replace = '\n'.join(replace_lines).strip()
            
            if search:
                edits.append((current_file, search, replace))
        
        i += 1
    
    return edits


def _do_rethink(session: Session, log: Callable) -> None:
    """
    Rethink with LLM suggestions.
    
    Per DESIGN.md, on each rethink:
    1. Commit best config as new baseline (if exists)
    2. Analyze results
    3. Suggest new ideas
    
    All changes are collected and passed to a single rethink() call
    to ensure era only increments once per rethink event.
    """
    should_commit = False
    new_ideas = []
    
    # Check if we should commit best
    if session.best:
        log(f"Committing best: {session.best.fitness}")
        # Re-apply the best individual's genes before committing
        # (they were reverted after eval, so we need to re-implement)
        if not getattr(session, '_dry_run', False):
            _reimplement_best(session, log)
        should_commit = True
    
    # Get LLM suggestions for new ideas
    if hasattr(session, '_llm_call') and session._llm_call:
        analysis = session._get_analysis()
        
        summary_lines = [
            f"Evaluations: {analysis['evaluations']}",
            f"Best: {analysis['best_fitness']} ({analysis['improvement']})",
            "",
            "Idea effectiveness:",
        ]
        for name, stats in analysis.get('ideas', {}).items():
            summary_lines.append(f"  {name}: {stats['eval_count']} evals, {stats['success_rate']:.0%} success")
        
        prompt = f"""Based on these optimization results, suggest 2-3 NEW ideas to try.

{chr(10).join(summary_lines)}

Original code:
```
{session._original_code[:3000]}...
```

Suggest NEW ideas (different from what we've tried).
Format:
idea_name: description
  variants: v1, v2, v3

Or for binary: idea_name: description"""
        
        try:
            response = session._llm_call(prompt)
            new_ideas = _parse_ideas_response(response)[:3]
        except Exception as e:
            log(f"  [yellow]Rethink LLM failed: {e}[/yellow]")
    
    # Single rethink() call with all changes
    if should_commit or new_ideas:
        session.rethink(commit_best=should_commit, add_ideas=new_ideas if new_ideas else None)
        for idea in new_ideas:
            log(f"  + {idea.name}: {idea.description}")
    else:
        # Just update last_rethink even if no changes
        session.last_rethink = session.evaluations
        session._save()


def _describe_genes(genes: dict) -> str:
    """Describe genes in human-readable format."""
    active = [f"{k}={v}" if v != "on" else k for k, v in genes.items() if v is not None]
    return ", ".join(active) if active else "baseline"


def _print_final_summary(session: Session, final: dict, console) -> None:
    """Print a comprehensive final summary."""
    from rich.panel import Panel
    
    elapsed = time.time() - session.start_time
    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        time_str = f"{hours}h {mins}m {secs}s"
    elif mins:
        time_str = f"{mins}m {secs}s"
    else:
        time_str = f"{secs}s"
    
    best = final.get('best')
    
    lines = [""]
    
    if best:
        direction = session.config.metric_direction
        metric = session.config.metric_name
        unit = session.config.metric_unit or ""
        
        # Use initial baseline for cumulative improvement across eras
        initial_baseline = session.initial_baseline_fitness
        current_baseline = session.baseline_fitness
        baseline = initial_baseline or current_baseline
        
        if baseline:
            # Calculate improvement from original baseline
            pct = (best['fitness'] - baseline) / baseline * 100
            if direction == "lower":
                arrow = "↓" if pct < 0 else "↑"
                improved = pct < 0
            else:
                arrow = "↑" if pct > 0 else "↓"
                improved = pct > 0
            
            improvement_str = f"{pct:+.1f}%"
            
            # Show original baseline prominently
            lines.append(f"[bold]📊 Baseline:[/] {baseline:.4f}{unit}")
            
            # Show best with improvement
            if improved:
                lines.append(f"[bold green]🏆 Best {metric}:[/] {best['fitness']:.4f}{unit} [bold green]({improvement_str} {arrow})[/]")
            else:
                lines.append(f"[bold yellow]🏆 Best {metric}:[/] {best['fitness']:.4f}{unit} [bold yellow]({improvement_str} {arrow})[/]")
        else:
            lines.append(f"[bold green]🏆 Best {metric}:[/] {best['fitness']:.4f}{unit}")
        
        # Collect all committed genes from era transitions + current best
        cumulative_genes = {}
        for entry in session.history:
            if entry.get('event') == 'era_transition':
                era_genes = entry.get('best_genes', {})
                if era_genes:
                    for name, value in era_genes.items():
                        if value is not None:
                            cumulative_genes[name] = value
        
        # Add current best's genes
        current_genes = best.get('genes', {})
        for name, value in current_genes.items():
            if value is not None:
                cumulative_genes[name] = value
        
        if cumulative_genes:
            lines.append("")
            lines.append("[bold cyan]Winning configuration:[/]")
            for name, value in cumulative_genes.items():
                # Get description from ideas if available
                idea = session.ideas.get(name)
                desc = idea.description if idea else ""
                # Truncate long descriptions
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                
                if value == 'on':
                    lines.append(f"  [green]• {name}[/]")
                    if desc:
                        lines.append(f"    [dim]{desc}[/]")
                else:
                    lines.append(f"  [green]• {name} = {value}[/]")
                    if desc:
                        lines.append(f"    [dim]{desc}[/]")
        else:
            lines.append("")
            lines.append("[bold cyan]Winning configuration:[/] baseline (no changes)")
    else:
        lines.append("[yellow]No successful evaluations[/]")
    
    lines.append("")
    lines.append("[bold]Stats:[/]")
    lines.append(f"  Evaluations: {session.evaluations}")
    lines.append(f"  Generations: {session.generation}")
    lines.append(f"  Eras: {session.era}")
    lines.append(f"  Ideas explored: {len(session.ideas)}")
    lines.append(f"  Time: {time_str}")
    
    lines.append("")
    lines.append(f"[dim]Results: {final['results_dir']}[/]")
    
    console.print(Panel(
        "\n".join(lines),
        title="[bold]Evolution Complete[/]",
        border_style="green",
    ))


def _reimplement_best(session: Session, log: Callable) -> None:
    """
    Re-implement the best individual's genes.
    
    Called before commit_best to ensure the winning code is applied
    (since it was reverted after evaluation).
    """
    if not session.best:
        return
    
    # Build active genes list
    active = []
    for name, value in session.best.genes.items():
        if value is not None:
            active.append({
                "name": name,
                "value": value,
                "description": session.ideas[name].description,
            })
    
    if not active:
        return  # Baseline is best, nothing to implement
    
    # Create a fake NextResult for _implement_genes
    class FakeResult:
        def __init__(self, active_genes):
            self.active = active_genes
    
    log(f"Re-implementing best genes...")
    _implement_genes(session, FakeResult(active), log)


