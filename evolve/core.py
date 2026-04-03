"""
Core GA logic with variant support.

An Idea has a name, description, and optional variants.
An Individual is a selection of variants for each idea.
The LLM generates edit instructions to implement those choices.
"""

import json
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Idea:
    """An optimization idea with optional variants."""
    name: str              # Short identifier (e.g., "depth")
    description: str       # What this idea controls
    variants: list[str] = field(default_factory=list)  # Possible values, empty = binary (on/off)
    
    def __hash__(self):
        return hash(self.name)
    
    def is_binary(self) -> bool:
        return len(self.variants) == 0


@dataclass
class SecondaryMetric:
    """A secondary metric to track (not optimized)."""
    name: str
    unit: str = ""
    direction: str = "lower"  # "lower" or "higher"


@dataclass
class Individual:
    """A selection of variants for each idea."""
    id: str
    genes: dict[str, str | None]   # idea_name -> variant (None = off, "on" for binary, or variant value)
    fitness: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)  # Secondary metrics
    generation: int = 0
    parents: tuple[str, str] | None = None  # Parent IDs for offspring
    code: str | None = None  # Generated code (implementation detail)
    
    def describe(self) -> str:
        """Describe active genes."""
        active = []
        for k, v in sorted(self.genes.items()):
            if v is not None:
                if v == "on":
                    active.append(k)
                else:
                    active.append(f"{k}={v}")
        return ", ".join(active) if active else "baseline"
    
    def active_count(self) -> int:
        """Count active genes."""
        return sum(1 for v in self.genes.values() if v is not None)


@dataclass 
class Config:
    """GA configuration."""
    name: str = "default"
    population_size: int = 6
    max_evaluations: int | None = 20
    elitism: int = 2
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    experiment_timeout: int = 600
    train_command: str = "uv run train.py"
    work_dir: Path = field(default_factory=lambda: Path("."))
    target_file: str = "train.py"
    metric_name: str = "val_bpb"
    metric_direction: str = "lower"  # "lower" or "higher"
    metric_unit: str = ""
    secondary_metrics: list[SecondaryMetric] = field(default_factory=list)
    rethink_interval: int = 5
    convergence_evals: int | None = None  # Default: rethink_interval * 3 + 1
    num_ideas: int = 8
    
    def __post_init__(self):
        if self.convergence_evals is None:
            self.convergence_evals = self.rethink_interval * 3 + 1
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)


class GARunner:
    """
    GA runner with variant support.
    
    1. Start with ideas (each with variants)
    2. Create population of variant combinations
    3. For each individual, ask LLM to generate edit instructions
    4. Apply edits, evaluate fitness
    5. Evolve population
    """
    
    def __init__(
        self,
        ideas: list[Idea],
        config: Config,
        llm_call=None,  # Function: (prompt: str) -> str
        log_callback=None,  # Function: (msg: str, runner: GARunner) -> None
    ):
        self.ideas = {idea.name: idea for idea in ideas}
        self.config = config
        self.llm_call = llm_call
        self.log_callback = log_callback
        
        self.target_path = config.work_dir / config.target_file
        if self.target_path.exists():
            self.original_code = self.target_path.read_text()
            self.backup_path = config.work_dir / f"{config.target_file}.backup"
            self.backup_path.write_text(self.original_code)
        else:
            self.original_code = ""
            self.backup_path = None
        
        self.population: list[Individual] = []
        self.evaluations = 0
        self.generation = 0
        self.best: Individual | None = None  # Best in current era
        self.best_at_eval: int = 0
        self.absolute_best: Individual | None = None  # Best across all eras (never reset)
        
        # Era tracking (for rethink)
        self.era = 0
        self.last_rethink = 0
        
        # Current individual being evaluated (for crash recovery)
        self.current_individual: str | None = None
        
        # Timing
        self.start_time = time.time()
        
        # History
        self.history: list[dict] = []
    
    def log(self, msg: str):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        formatted = f"[{mins:02d}:{secs:02d}] {msg}"
        if self.log_callback:
            self.log_callback(formatted, self)
        else:
            print(formatted)
    
    def is_better(self, a: float, b: float) -> bool:
        """Is fitness a better than b?"""
        if self.config.metric_direction == "lower":
            return a < b
        return a > b
    
    # -------------------------------------------------------------------------
    # Population
    # -------------------------------------------------------------------------
    
    def create_individual(self, genes: dict[str, str | None], generation: int = 0) -> Individual:
        """Create an individual with full gene map."""
        # Ensure all ideas are present
        full_genes = {}
        for name in self.ideas:
            full_genes[name] = genes.get(name)
        return Individual(
            id=f"ind-{random.randint(0, 999999):06d}",
            genes=full_genes,
            generation=generation,
        )
    
    def create_baseline_individual(self, generation: int = 0) -> Individual:
        """Create baseline individual (all genes off)."""
        genes = {name: None for name in self.ideas}
        return self.create_individual(genes, generation)
    
    def create_random_individual(self, generation: int | None = None) -> Individual:
        """Create individual with random variant selections."""
        if generation is None:
            generation = self.generation
        genes = {}
        for name, idea in self.ideas.items():
            # 50% chance to include each idea
            if random.random() < 0.5:
                genes[name] = None
            elif idea.is_binary():
                genes[name] = "on"
            else:
                genes[name] = random.choice(idea.variants)
        return self.create_individual(genes, generation)
    
    def initialize_population(self):
        """Create initial population."""
        # Always include baseline (no changes)
        self.population = [self.create_baseline_individual(0)]
        
        # Fill with random individuals
        while len(self.population) < self.config.population_size:
            self.population.append(self.create_random_individual(0))
        
        self.log(f"Created population of {len(self.population)}")
    
    # -------------------------------------------------------------------------
    # Gene Normalization
    # -------------------------------------------------------------------------
    
    def normalize_genes(self, individual: Individual) -> None:
        """Ensure individual has all current ideas (None for missing)."""
        # Add missing ideas as None
        for name in self.ideas:
            if name not in individual.genes:
                individual.genes[name] = None
        # Remove genes for ideas that no longer exist
        for name in list(individual.genes.keys()):
            if name not in self.ideas:
                del individual.genes[name]
    
    # -------------------------------------------------------------------------
    # Code Generation (LLM with edits)
    # -------------------------------------------------------------------------
    
    def generate_code(self, individual: Individual) -> str:
        """Ask LLM to generate edits implementing the individual's genes."""
        if individual.active_count() == 0:
            return self.original_code
        
        if self.llm_call is None:
            return self.original_code
        
        # Build description of changes
        changes = []
        for name, variant in sorted(individual.genes.items()):
            if variant is None:
                continue
            idea = self.ideas[name]
            if idea.is_binary():
                changes.append(f"- {name}: {idea.description}")
            else:
                changes.append(f"- {name} = {variant}: {idea.description}")
        
        changes_text = "\n".join(changes)
        
        prompt = f"""Apply these changes to the code below:

{changes_text}

Output your changes as SEARCH/REPLACE blocks. Each block:
- Must match the original code EXACTLY (including whitespace)
- Will replace that exact text with your new version

Format:
<<<<<<< SEARCH
exact code to find
=======
replacement code
>>>>>>> REPLACE

You can output multiple SEARCH/REPLACE blocks.
Only output the blocks, no other text.

Original code:
```
{self.original_code}
```

Your SEARCH/REPLACE blocks:"""
        
        self.log(f"  Calling LLM for edits: {individual.describe()}")
        response = self.llm_call(prompt)
        
        # Parse and apply edits
        edits = self._parse_edits(response)
        self.log(f"  Applying {len(edits)} edit(s)")
        
        code = self.original_code
        for search, replace in edits:
            if search in code:
                code = code.replace(search, replace, 1)
            else:
                self.log(f"  Warning: Could not find search text, skipping edit")
        
        return code
    
    def _parse_edits(self, response: str) -> list[tuple[str, str]]:
        """Parse SEARCH/REPLACE blocks from LLM response."""
        edits = []
        parts = response.split("<<<<<<< SEARCH")
        
        for part in parts[1:]:
            if "=======" not in part or ">>>>>>> REPLACE" not in part:
                continue
            
            try:
                search_and_rest = part.split("=======", 1)
                search = search_and_rest[0].strip()
                
                replace_and_rest = search_and_rest[1].split(">>>>>>> REPLACE", 1)
                replace = replace_and_rest[0].strip()
                
                if search:
                    edits.append((search, replace))
            except (IndexError, ValueError):
                continue
        
        return edits
    
    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    
    def _parse_metrics(self, output: str) -> dict[str, float]:
        """Parse all metrics from output."""
        metrics = {}
        for line in output.split('\n'):
            # Format 1: "name: value"
            match = re.match(r'^([\w.]+):\s*([\d.\-+eE]+)', line)
            if match:
                try:
                    metrics[match.group(1)] = float(match.group(2))
                except ValueError:
                    pass
            # Format 2: "METRIC name=value"
            match = re.match(r'^METRIC\s+([\w.]+)=([\d.\-+eE]+)', line)
            if match:
                try:
                    metrics[match.group(1)] = float(match.group(2))
                except ValueError:
                    pass
        return metrics
    
    def evaluate(self, individual: Individual) -> float:
        """Evaluate an individual's fitness."""
        self.log(f"Evaluating {individual.id}: {individual.describe()}")
        self.current_individual = individual.id
        
        # Generate code if needed
        if individual.code is None:
            try:
                individual.code = self.generate_code(individual)
            except Exception as e:
                self.log(f"  Code generation failed: {e}")
                individual.fitness = float('inf')
                self.evaluations += 1
                self._log_history(individual)
                self.current_individual = None
                return float('inf')
        
        # Write code to file
        try:
            self.target_path.write_text(individual.code)
        except Exception as e:
            self.log(f"  Failed to write code: {e}")
            individual.fitness = float('inf')
            self.evaluations += 1
            self._log_history(individual)
            self.current_individual = None
            return float('inf')
        
        # Run training
        try:
            fitness, metrics = self._run_training()
            individual.fitness = fitness
            individual.metrics = metrics
            self.evaluations += 1
            
            # Track era best (for convergence detection)
            if self.best is None or (fitness != float('inf') and self.is_better(fitness, self.best.fitness)):
                self.best = individual
                self.best_at_eval = self.evaluations
                self.log(f"  NEW BEST (era): {fitness}")
            else:
                self.log(f"  Fitness: {fitness}")
            
            # Track absolute best (never reset across eras)
            if self.absolute_best is None or (fitness != float('inf') and self.is_better(fitness, self.absolute_best.fitness)):
                self.absolute_best = individual
                self.log(f"  NEW ABSOLUTE BEST: {fitness}")
            
            self._log_history(individual)
            return fitness
            
        finally:
            self.target_path.write_text(self.original_code)
            self.current_individual = None
    
    def _run_training(self) -> tuple[float, dict[str, float]]:
        """Run training and extract metrics."""
        self.log(f"  Running training (timeout: {self.config.experiment_timeout}s)...")
        start = time.time()
        
        try:
            process = subprocess.Popen(
                self.config.train_command,
                shell=True,
                cwd=self.config.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            output_lines = []
            last_progress = 0
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output_lines.append(line)
                
                elapsed = time.time() - start
                if elapsed - last_progress >= 30:
                    self.log(f"  ... {int(elapsed)}s elapsed")
                    last_progress = elapsed
                
                if elapsed > self.config.experiment_timeout:
                    process.kill()
                    raise subprocess.TimeoutExpired(self.config.train_command, self.config.experiment_timeout)
            
            output = ''.join(output_lines)
            returncode = process.returncode
            
            elapsed = time.time() - start
            self.log(f"  Completed in {int(elapsed)}s")
            
            if returncode != 0:
                self.log(f"  Training failed (exit code {returncode})")
                (self.config.work_dir / "run.log").write_text(output)
                return float('inf'), {}
            
            # Parse all metrics
            all_metrics = self._parse_metrics(output)
            
            # Extract primary fitness
            fitness = all_metrics.pop(self.config.metric_name, float('inf'))
            
            if fitness == float('inf'):
                self.log(f"  Could not find {self.config.metric_name} in output")
            
            return fitness, all_metrics
            
        except subprocess.TimeoutExpired:
            self.log(f"  Training timed out")
            return float('inf'), {}
    
    def _log_history(self, individual: Individual):
        """Log an evaluation to history."""
        from evolve import persistence
        
        entry = {
            "timestamp": time.time(),
            "evaluation": self.evaluations,
            "generation": self.generation,
            "id": individual.id,
            "genes": individual.genes,
            "fitness": individual.fitness if individual.fitness != float('inf') else None,
            "metrics": individual.metrics,
        }
        self.history.append(entry)
        
        # Write immediately to disk for crash recovery
        persistence.append_history(self, entry)
    
    def _log_ideas(self, event: str, ideas: list[Idea] = None, new_ideas: list[Idea] = None):
        """Log ideas event to history."""
        entry = {
            "event": event,
            "timestamp": time.time(),
            "era": self.era,
        }
        
        if event == "init":
            entry["ideas"] = [
                {"name": idea.name, "description": idea.description, "variants": idea.variants}
                for idea in (ideas or self.ideas.values())
            ]
        elif event == "rethink":
            entry["new_ideas"] = [
                {"name": idea.name, "description": idea.description, "variants": idea.variants}
                for idea in (new_ideas or [])
            ]
            entry["total_ideas"] = len(self.ideas)
        
        self.history.append(entry)
    
    # -------------------------------------------------------------------------
    # Convergence
    # -------------------------------------------------------------------------
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if self.best is None:
            return False
        evals_since_best = self.evaluations - self.best_at_eval
        return evals_since_best >= self.config.convergence_evals
    
    # -------------------------------------------------------------------------
    # Rethink (Era transitions)
    # -------------------------------------------------------------------------
    
    def summarize_results(self) -> str:
        """Summarize what we've learned so far."""
        evaluated = [i for i in self.population if i.fitness is not None and i.fitness != float('inf')]
        if not evaluated:
            return "No successful results yet."
        
        evaluated.sort(key=lambda i: i.fitness, reverse=(self.config.metric_direction == "higher"))
        
        lines = []
        lines.append(f"Evaluations so far: {self.evaluations}")
        lines.append(f"Best fitness: {self.best.fitness if self.best else 'N/A'}")
        lines.append("")
        lines.append("Top 5 results:")
        for i, ind in enumerate(evaluated[:5]):
            lines.append(f"  {i+1}. fitness={ind.fitness:.4f} genes={ind.describe()}")
        
        # Gene frequency in top performers
        lines.append("")
        lines.append("Common genes in top 5:")
        gene_counts = {}
        for ind in evaluated[:5]:
            for gene, variant in ind.genes.items():
                if variant is not None:
                    key = f"{gene}={variant}" if variant != "on" else gene
                    gene_counts[key] = gene_counts.get(key, 0) + 1
        for gene, count in sorted(gene_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {gene}: {count}/5")
        
        return "\n".join(lines)
    
    def get_rethink_statistics(self) -> dict:
        """Get statistics for rethink analysis."""
        stats = {
            "evaluations": self.evaluations,
            "best_fitness": self.best.fitness if self.best else None,
            "baseline_fitness": None,
            "ideas": {},
        }
        
        # Find baseline fitness
        for entry in self.history:
            if "genes" in entry and all(v is None for v in entry["genes"].values()):
                stats["baseline_fitness"] = entry.get("fitness")
                break
        
        # Per-idea statistics
        for name in self.ideas:
            idea_stats = {
                "eval_count": 0,
                "success_count": 0,
                "in_top5": 0,
                "variants": {},
            }
            
            # Collect evaluations where this idea was ON
            for entry in self.history:
                if "genes" not in entry or entry.get("fitness") is None:
                    continue
                
                value = entry["genes"].get(name)
                if value is not None:
                    idea_stats["eval_count"] += 1
                    
                    # Check if it improved over baseline
                    if stats["baseline_fitness"] is not None:
                        if self.config.metric_direction == "lower":
                            if entry["fitness"] < stats["baseline_fitness"]:
                                idea_stats["success_count"] += 1
                        else:
                            if entry["fitness"] > stats["baseline_fitness"]:
                                idea_stats["success_count"] += 1
                    
                    # Per-variant stats
                    if value not in idea_stats["variants"]:
                        idea_stats["variants"][value] = {"count": 0, "fitnesses": []}
                    idea_stats["variants"][value]["count"] += 1
                    idea_stats["variants"][value]["fitnesses"].append(entry["fitness"])
            
            # Calculate variant averages and bests
            for variant, vs in idea_stats["variants"].items():
                if vs["fitnesses"]:
                    vs["avg_fitness"] = sum(vs["fitnesses"]) / len(vs["fitnesses"])
                    if self.config.metric_direction == "lower":
                        vs["best_fitness"] = min(vs["fitnesses"])
                    else:
                        vs["best_fitness"] = max(vs["fitnesses"])
                del vs["fitnesses"]  # Don't need raw list
            
            stats["ideas"][name] = idea_stats
        
        return stats
    
    def add_ideas(self, new_ideas: list[Idea]) -> list[str]:
        """Add new ideas and normalize all individuals."""
        added = []
        for idea in new_ideas:
            if idea.name not in self.ideas:
                self.ideas[idea.name] = idea
                added.append(idea.name)
        
        if added:
            # Normalize all individuals
            for ind in self.population:
                self.normalize_genes(ind)
            self.era += 1
        
        return added
    
    def remove_ideas(self, idea_names: list[str]) -> list[str]:
        """Remove ideas and normalize all individuals."""
        removed = []
        for name in idea_names:
            if name in self.ideas:
                del self.ideas[name]
                removed.append(name)
        
        if removed:
            # Normalize all individuals
            for ind in self.population:
                self.normalize_genes(ind)
            self.era += 1
        
        return removed
    
    def rethink(self):
        """Have LLM analyze results and suggest new ideas/variants."""
        self.log(f"\n=== RETHINK (Era {self.era + 1}) ===")
        
        self.last_rethink = self.evaluations
        
        if self.llm_call is None:
            self.log("  No LLM available for rethink")
            return
        
        summary = self.summarize_results()
        
        current_ideas = "\n".join(
            f"- {name}: {idea.description}" + 
            (f" (variants: {', '.join(idea.variants)})" if idea.variants else " (binary)")
            for name, idea in self.ideas.items()
        )
        
        num_new = self.config.num_ideas // 2
        
        prompt = f"""Based on these optimization results, suggest {num_new} NEW ideas to try.

Current ideas:
{current_ideas}

Results so far:
{summary}

Original code being optimized:
```python
{self.original_code[:3000]}...
```

Suggest NEW ideas (different from current) with variants where applicable.

Format:
idea_name: description
  variants: value1, value2, value3

Or for binary ideas (on/off):
idea_name: description

Your {num_new} new ideas:"""
        
        try:
            response = self.llm_call(prompt)
            new_ideas = self._parse_ideas(response)
            
            added = self.add_ideas(new_ideas[:num_new])
            
            for name in added:
                idea = self.ideas[name]
                variant_str = f" (variants: {', '.join(idea.variants)})" if idea.variants else ""
                self.log(f"  + {name}: {idea.description}{variant_str}")
            
            if added:
                self.log(f"Added {len(added)} new ideas (total: {len(self.ideas)})")
                self._log_ideas("rethink", new_ideas=[self.ideas[n] for n in added])
            else:
                self.log("No new ideas suggested")
                
        except Exception as e:
            self.log(f"Rethink failed: {e}")
    
    def _parse_ideas(self, response: str) -> list[Idea]:
        """Parse ideas with variants from LLM response."""
        ideas = []
        lines = response.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Look for idea definition (name: description)
            if ':' in line and not line.startswith('variants'):
                parts = line.split(':', 1)
                name = parts[0].strip().replace(' ', '_').replace('-', '_').replace('*', '').lower()
                desc = parts[1].strip()
                
                if not name or not desc or len(name) > 30:
                    i += 1
                    continue
                
                # Check for variants on next line
                variants = []
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('variants:'):
                        variant_str = next_line.split(':', 1)[1].strip()
                        variants = [v.strip() for v in variant_str.split(',') if v.strip()]
                        i += 1
                
                ideas.append(Idea(name=name, description=desc, variants=variants))
            
            i += 1
        
        return ideas
    
    # -------------------------------------------------------------------------
    # Commit Best
    # -------------------------------------------------------------------------
    
    def commit_best(self) -> bool:
        """Apply best config permanently and reset for next era."""
        if not self.best or self.best.fitness is None:
            return False
        
        # 1. Apply best individual's code permanently
        if self.best.code:
            self.target_path.write_text(self.best.code)
        
        # 2. Git commit
        try:
            subprocess.run(["git", "add", "-A"], cwd=self.config.work_dir, check=True)
            improvement = ""
            baseline_fitness = None
            for entry in self.history:
                if "genes" in entry and all(v is None for v in entry["genes"].values()):
                    baseline_fitness = entry.get("fitness")
                    break
            if baseline_fitness and self.best.fitness:
                pct = (self.best.fitness - baseline_fitness) / baseline_fitness * 100
                sign = "+" if pct >= 0 else ""
                improvement = f" ({sign}{pct:.1f}%)"
            
            msg = f"evolve: {self.best.describe()}{improvement}"
            subprocess.run(["git", "commit", "-m", msg], cwd=self.config.work_dir, check=True)
        except subprocess.CalledProcessError as e:
            self.log(f"Git commit failed: {e}")
        
        # 3. Update original_code to new baseline
        self.original_code = self.target_path.read_text()
        
        # 4. Log era transition
        self.history.append({
            "event": "era_transition",
            "timestamp": time.time(),
            "era": self.era,
            "id": f"era-{self.era}-baseline",
            "genes": self.best.genes,
            "fitness": self.best.fitness,
        })
        
        # 5. Reset state (but absolute_best is preserved)
        self.era += 1
        self.last_rethink = self.evaluations
        self.generation = 0
        self.best = None
        self.best_at_eval = self.evaluations
        self.population = []  # Will be re-initialized with new ideas
        
        return True
    
    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    
    def select(self) -> Individual:
        """Tournament selection."""
        valid = [i for i in self.population if i.fitness is not None and i.fitness != float('inf')]
        if not valid:
            return random.choice(self.population)
        
        tournament_size = min(3, len(valid))
        tournament = random.sample(valid, tournament_size)
        if self.config.metric_direction == "lower":
            return min(tournament, key=lambda i: i.fitness)
        return max(tournament, key=lambda i: i.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover: for each gene, pick from either parent."""
        child_genes = {}
        
        if random.random() > self.config.crossover_rate:
            # No crossover - copy random parent
            parent = random.choice([parent1, parent2])
            child_genes = {k: v for k, v in parent.genes.items()}
        else:
            # Uniform crossover: each gene from either parent
            for name in self.ideas:
                child_genes[name] = random.choice([
                    parent1.genes.get(name),
                    parent2.genes.get(name)
                ])
        
        child = self.create_individual(child_genes, self.generation)
        child.parents = (parent1.id, parent2.id)
        return child
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate by changing variants or toggling genes."""
        for name, idea in self.ideas.items():
            if random.random() < self.config.mutation_rate:
                current = individual.genes.get(name)
                
                if idea.is_binary():
                    # Toggle
                    individual.genes[name] = None if current == "on" else "on"
                else:
                    # Pick different value from [variants + None]
                    options = list(idea.variants) + [None]
                    options = [v for v in options if v != current]
                    if options:
                        individual.genes[name] = random.choice(options)
        
        individual.code = None  # Invalidate cached code
        return individual
    
    def evolve(self):
        """Create next generation."""
        self.generation += 1
        self.log(f"\n=== Generation {self.generation} ===")
        
        # Sort by fitness (handle inf)
        evaluated = [i for i in self.population if i.fitness is not None]
        evaluated.sort(
            key=lambda i: i.fitness if i.fitness != float('inf') else float('inf'),
            reverse=(self.config.metric_direction == "higher")
        )
        
        # Elitism - copy best individuals
        new_population = []
        for i in range(min(self.config.elitism, len(evaluated))):
            if evaluated[i].fitness != float('inf'):
                elite = self.create_individual(evaluated[i].genes.copy(), self.generation)
                elite.code = evaluated[i].code
                elite.fitness = evaluated[i].fitness
                elite.metrics = evaluated[i].metrics.copy()
                new_population.append(elite)
        
        # Fill with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.select()
            parent2 = self.select()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            self.normalize_genes(child)
            new_population.append(child)
        
        self.population = new_population
    
    # -------------------------------------------------------------------------
    # Summary Generation
    # -------------------------------------------------------------------------
    
    def generate_summary(self) -> str:
        """Generate RESULTS.md content."""
        lines = [f"# cEvolve Results: {self.config.name}", ""]
        
        # Use absolute_best (never reset across eras)
        best = self.absolute_best or self.best
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Evaluations**: {self.evaluations}")
        if self.generation > 0:
            lines.append(f"- **Generations**: {self.generation + 1}")
        lines.append(f"- **Eras**: {self.era + 1}")
        
        # Find baseline
        baseline_fitness = None
        for entry in self.history:
            if "genes" in entry and all(v is None for v in entry["genes"].values()):
                baseline_fitness = entry.get("fitness")
                break
        
        if baseline_fitness is not None:
            lines.append(f"- **Baseline**: {baseline_fitness} {self.config.metric_unit}")
        
        if best:
            improvement = ""
            if baseline_fitness and best.fitness:
                pct = (best.fitness - baseline_fitness) / baseline_fitness * 100
                sign = "+" if pct >= 0 else ""
                improvement = f" ({sign}{pct:.1f}% from baseline)"
            lines.append(f"- **Best**: {best.fitness} {self.config.metric_unit}{improvement}")
        
        lines.append("")
        
        # Winning configuration
        lines.append("## Winning Configuration")
        lines.append("")
        
        if best:
            active = [(n, v) for n, v in best.genes.items() if v is not None]
            if not active:
                lines.append("Baseline (no changes)")
            else:
                for name, value in active:
                    idea = self.ideas.get(name)
                    if idea and idea.variants:
                        lines.append(f"- **{name}** = `{value}`")
                    else:
                        lines.append(f"- **{name}**")
                    if idea:
                        lines.append(f"  _{idea.description}_")
        else:
            lines.append("No successful evaluation yet.")
        
        lines.append("")
        
        # Ideas effectiveness
        lines.append("## Ideas Effectiveness")
        lines.append("")
        lines.append("| Idea | Evals | Success Rate | In Top 5 |")
        lines.append("|------|-------|--------------|----------|")
        
        stats = self.get_rethink_statistics()
        for name, idea_stats in stats["ideas"].items():
            evals = idea_stats["eval_count"]
            if evals == 0:
                continue
            success_rate = idea_stats["success_count"] / evals * 100 if evals > 0 else 0
            lines.append(f"| {name} | {evals} | {success_rate:.0f}% | {idea_stats['in_top5']} |")
        
        lines.append("")
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    
    def run(self) -> Individual | None:
        """Run the GA."""
        if self.config.max_evaluations:
            evals_after_first = self.config.max_evaluations - self.config.population_size
            evals_per_gen = self.config.population_size - self.config.elitism
            estimated_gens = 1 + max(0, evals_after_first // evals_per_gen) if evals_per_gen > 0 else 1
        else:
            estimated_gens = "unlimited"
        
        # Log search space
        search_space = 1
        for idea in self.ideas.values():
            if idea.is_binary():
                search_space *= 2
            else:
                search_space *= (len(idea.variants) + 1)  # +1 for "not included"
        
        self.log(f"Starting GA with {len(self.ideas)} ideas")
        for name, idea in self.ideas.items():
            if idea.is_binary():
                self.log(f"  - {name}: {idea.description} (binary)")
            else:
                self.log(f"  - {name}: {idea.description} (variants: {', '.join(idea.variants)})")
        self.log(f"Search space: {search_space:,} combinations")
        
        # Log initial ideas
        self._log_ideas("init")
        self.log(f"Population: {self.config.population_size}, Max evals: {self.config.max_evaluations}, Est. generations: ~{estimated_gens}")
        self.log(f"Convergence: stop after {self.config.convergence_evals} evals without improvement")
        if self.config.rethink_interval > 0:
            self.log(f"Rethink every {self.config.rethink_interval} evaluations")
        
        self.initialize_population()
        
        # Evaluate initial population
        for ind in self.population:
            if self.config.max_evaluations and self.evaluations >= self.config.max_evaluations:
                break
            if self.is_converged():
                self.log(f"Converged! No improvement for {self.evaluations - self.best_at_eval} evals")
                break
            if ind.fitness is None:
                self.evaluate(ind)
                self._maybe_rethink()
        
        # Evolution loop
        while True:
            if self.config.max_evaluations and self.evaluations >= self.config.max_evaluations:
                break
            if self.is_converged():
                self.log(f"Converged! No improvement for {self.evaluations - self.best_at_eval} evals")
                break
            
            self.evolve()
            
            for ind in self.population:
                if self.config.max_evaluations and self.evaluations >= self.config.max_evaluations:
                    break
                if self.is_converged():
                    break
                if ind.fitness is None:
                    self.evaluate(ind)
                    self._maybe_rethink()
        
        # Final summary (use absolute_best which is never reset)
        best = self.absolute_best or self.best
        self.log(f"\n=== Final Results ===")
        self.log(f"Best fitness: {best.fitness if best else 'N/A'}")
        self.log(f"Best genes: {best.describe() if best else 'N/A'}")
        self.log(f"Total evaluations: {self.evaluations}")
        self.log(f"Total eras: {self.era + 1}")
        
        self.target_path.write_text(self.original_code)
        
        return best
    
    def _maybe_rethink(self):
        """Check if it's time to rethink."""
        if self.config.rethink_interval <= 0:
            return
        if self.evaluations - self.last_rethink >= self.config.rethink_interval:
            # Commit best before rethink to accumulate improvements
            if self.best and self.best.code:
                self.log(f"\n=== Committing best config: {self.best.describe()} ({self.best.fitness}) ===")
                # Update baseline code (future edits build on this)
                self.original_code = self.best.code
                self.target_path.write_text(self.best.code)
                # Log era transition
                self.history.append({
                    "event": "era_transition", 
                    "timestamp": time.time(),
                    "era": self.era,
                    "fitness": self.best.fitness,
                    "genes": self.best.genes,
                })
                self.era += 1
                # Reset for new era (but absolute_best is preserved)
                self.best = None
                self.best_at_eval = self.evaluations
                self.generation = 0
                self.population = []
                self.initialize_population()
            self.rethink()
