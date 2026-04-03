"""
Session - Composable interface for evolutionary optimization.

This wraps the core GA logic and provides the interface used by CLI commands.
The Session is LLM-agnostic - LLM logic lives in the `run` command only.

Usage:
    # Create session
    session = Session.create(
        name="optimize-parser",
        ideas=[Idea("use_cache", "Enable caching"), ...],
        bench_command="./bench.sh",
        metric="time_ms",
    )
    
    # Evolution loop (called by extension or `run` command)
    while True:
        result = session.next()
        if result.status == "converged":
            break
        
        # Extension implements genes here...
        
        session.eval(result.individual_id)  # or session.record() + session.revert()
    
    session.stop()
"""

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

from .core import Idea, Individual, Config, SecondaryMetric
from .revert import get_reverter, Reverter
from .bench import run_benchmark, BenchmarkResult
from . import persistence


@dataclass
class NextResult:
    """Result from session.next()"""
    status: str  # "ready", "converged", "max_evals"
    individual_id: Optional[str] = None
    generation: int = 0
    genes: dict = field(default_factory=dict)
    active: list = field(default_factory=list)  # [{name, value, description}, ...]
    inactive: list = field(default_factory=list)
    is_baseline: bool = False
    best: Optional[dict] = None  # If converged/stopped
    message: Optional[str] = None


@dataclass
class EvalResult:
    """Result from session.eval() or session.record()"""
    individual_id: str
    fitness: Optional[float]
    metrics: dict
    is_best: bool
    improvement: Optional[str]
    evaluations: int
    status: str  # "continue", "converged", "max_evals"
    error: Optional[str] = None


class Session:
    """
    Session manages evolutionary optimization state.
    
    Provides the composable interface:
    - next() - get next individual to evaluate
    - eval() - run benchmark, record, revert (all-in-one)
    - record() - just record result (agent ran benchmark)
    - revert() - just revert files
    - rethink() - analyze and modify ideas
    - status() - get current state
    - stop() - finalize and generate reports
    """
    
    def __init__(
        self,
        config: Config,
        ideas: list[Idea],
        reverter: Reverter,
        log_callback: Callable[[str], None] = None,
    ):
        self.config = config
        self.ideas = {idea.name: idea for idea in ideas}
        self.reverter = reverter
        self.log_callback = log_callback
        
        # GA state
        self.population: list[Individual] = []
        self.evaluations = 0
        self.generation = 0
        self.era = 0
        self.best: Optional[Individual] = None
        self.best_at_eval = 0
        self.absolute_best: Optional[Individual] = None
        self.baseline_fitness: Optional[float] = None
        self.initial_baseline_fitness: Optional[float] = None  # Never resets across eras
        self.last_rethink = 0
        self.current_individual: Optional[str] = None
        
        # History
        self.history: list[dict] = []
        
        # Timing
        self.start_time = time.time()
        
        # Session directory
        self.session_dir = config.work_dir / ".cevolve" / config.name
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, msg: str):
        """Log a message with timestamp."""
        import sys
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        formatted = f"[{mins:02d}:{secs:02d}] {msg}"
        if self.log_callback:
            self.log_callback(formatted)
        else:
            # Write to stderr so JSON output on stdout stays clean
            print(formatted, file=sys.stderr)
    
    # =========================================================================
    # Factory methods
    # =========================================================================
    
    @classmethod
    def create(
        cls,
        name: str,
        ideas: list[Idea],
        bench_command: str,
        metric: str = "time_ms",
        direction: str = "lower",
        scope: list[str] = None,
        exclude: list[str] = None,
        population_size: int = 6,
        elitism: int = 2,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        max_evaluations: int = None,
        convergence_evals: int = None,
        rethink_interval: int = 5,
        experiment_timeout: int = 600,
        revert_strategy: str = "git",
        target_file: str = None,  # For single-file backward compat
        target_files: list[str] = None,  # For multi-file mode
        work_dir: str = ".",
        secondary_metrics: list[dict] = None,
        log_callback: Callable[[str], None] = None,
    ) -> "Session":
        """Create a new session."""
        
        work_dir = Path(work_dir)
        
        # Build config
        config = Config(
            name=name,
            population_size=population_size,
            max_evaluations=max_evaluations,
            elitism=elitism,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            experiment_timeout=experiment_timeout,
            train_command=bench_command,
            work_dir=work_dir,
            target_file=target_file or "",
            metric_name=metric,
            metric_direction=direction,
            rethink_interval=rethink_interval,
            convergence_evals=convergence_evals,
            num_ideas=len(ideas),
        )
        
        if secondary_metrics:
            config.secondary_metrics = [
                SecondaryMetric(**m) if isinstance(m, dict) else m
                for m in secondary_metrics
            ]
        
        # Create reverter
        if target_files and revert_strategy == "multi":
            reverter = get_reverter("multi", work_dir, target_files=target_files)
        elif target_file and revert_strategy == "single":
            reverter = get_reverter("single", work_dir, target_file=target_file)
        else:
            reverter = get_reverter(
                revert_strategy,
                work_dir,
                scope=scope,
                exclude=exclude,
                target_file=target_file,
            )
        
        # Take initial snapshot
        reverter.snapshot()
        
        # Create session
        session = cls(config, ideas, reverter, log_callback)
        
        # Initialize population
        session._initialize_population()
        
        # Save initial state
        session._save()
        
        return session
    
    @classmethod
    def load(cls, name: str = None, work_dir: str = ".", log_callback: Callable[[str], None] = None) -> "Session":
        """Load existing session."""
        work_dir = Path(work_dir)
        cevolve_dir = work_dir / ".cevolve"
        
        # Auto-detect session if not specified
        if name is None:
            name = cls._detect_session(work_dir)
        
        session_dir = cevolve_dir / name
        
        # Load config
        with open(session_dir / "config.json") as f:
            config_data = json.load(f)
        
        # Handle Path serialization
        config_data["work_dir"] = Path(config_data.get("work_dir", work_dir))
        config = Config(**config_data)
        
        # Load ideas
        with open(session_dir / "ideas.json") as f:
            ideas_data = json.load(f)
        ideas = [Idea(**d) for d in ideas_data]
        
        # Create reverter (recreate based on config)
        revert_strategy = config_data.get("revert_strategy", "git")
        reverter = get_reverter(
            revert_strategy,
            config.work_dir,
            target_file=config.target_file if config.target_file else None,
        )
        
        # Create session
        session = cls(config, ideas, reverter, log_callback)
        
        # Load state
        with open(session_dir / "state.json") as f:
            state = json.load(f)
        
        session.evaluations = state["evaluations"]
        session.generation = state["generation"]
        session.era = state["era"]
        session.best_at_eval = state["best_at_eval"]
        session.baseline_fitness = state.get("baseline_fitness")
        session.initial_baseline_fitness = state.get("initial_baseline_fitness")
        session.last_rethink = state["last_rethink"]
        session.current_individual = state.get("current_individual")
        
        if state.get("best"):
            session.best = Individual(**state["best"])
        if state.get("absolute_best"):
            session.absolute_best = Individual(**state["absolute_best"])
        
        # Load population
        with open(session_dir / "population.json") as f:
            pop_data = json.load(f)
        session.population = [Individual(**d) for d in pop_data]
        
        # Load history
        history_file = session_dir / "history.jsonl"
        if history_file.exists():
            session.history = []
            with open(history_file) as f:
                for line in f:
                    if line.strip():
                        session.history.append(json.loads(line))
        
        # Migration: find initial baseline from history if not set
        if session.initial_baseline_fitness is None and session.history:
            for entry in session.history:
                if "genes" in entry and entry.get("fitness") is not None:
                    # Check if all genes are null/off (baseline)
                    genes = entry["genes"]
                    if all(v is None for v in genes.values()):
                        session.initial_baseline_fitness = entry["fitness"]
                        session._save()  # Persist the migration
                        break
        
        return session
    
    @staticmethod
    def _detect_session(work_dir: Path) -> str:
        """Auto-detect session name."""
        cevolve_dir = work_dir / ".cevolve"
        
        if not cevolve_dir.exists():
            raise ValueError("No .cevolve directory found. Run 'cevolve init' first.")
        
        sessions = [d.name for d in cevolve_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not sessions:
            raise ValueError("No sessions found. Run 'cevolve init' first.")
        
        # Check for current marker
        current_file = cevolve_dir / ".current"
        if current_file.exists():
            return current_file.read_text().strip()
        
        # Check for session with current_individual (active evaluation)
        for name in sessions:
            state_file = cevolve_dir / name / "state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                if state.get("current_individual"):
                    return name
        
        # Default to most recent
        sessions.sort(key=lambda s: (cevolve_dir / s).stat().st_mtime, reverse=True)
        return sessions[0]
    
    # =========================================================================
    # Core interface
    # =========================================================================
    
    def next(self) -> NextResult:
        """
        Get next individual to evaluate.
        
        Returns:
            NextResult with status and individual details
            
        Status values:
            - "ready": Individual ready for evaluation
            - "rethink_required": Must commit best genes and call rethink() first
            - "converged": No improvement, optimization complete
            - "max_evals": Reached max evaluations, optimization complete
        """
        # Check stopping conditions
        if self._is_converged():
            return NextResult(
                status="converged",
                best=self._best_dict(),
                message=f"No improvement for {self.evaluations - self.best_at_eval} evaluations",
            )
        
        if self.config.max_evaluations and self.evaluations >= self.config.max_evaluations:
            return NextResult(
                status="max_evals",
                best=self._best_dict(),
                message=f"Reached maximum {self.config.max_evaluations} evaluations",
            )
        
        # Check if rethink is required (blocks until agent commits best and calls rethink)
        if self.should_rethink():
            return NextResult(
                status="rethink_required",
                best=self._best_dict_with_genes_to_implement(),
                message=f"Rethink due after {self.evaluations} evaluations. Commit best genes and call 'cevolve rethink'.",
            )
        
        # Find next unevaluated individual
        individual = self._get_next_unevaluated()
        
        if individual is None:
            # All evaluated - evolve to next generation
            self._evolve()
            individual = self._get_next_unevaluated()
        
        self.current_individual = individual.id
        self._save()
        
        # Build result
        active = []
        inactive = []
        for name, value in sorted(individual.genes.items()):
            if value is not None:
                active.append({
                    "name": name,
                    "value": value,
                    "description": self.ideas[name].description,
                })
            else:
                inactive.append(name)
        
        is_baseline = all(v is None for v in individual.genes.values())
        
        return NextResult(
            status="ready",
            individual_id=individual.id,
            generation=self.generation,
            genes=individual.genes.copy(),
            active=active,
            inactive=inactive,
            is_baseline=is_baseline,
        )
    
    def eval(self, individual_id: str, timeout: int = None, revert: bool = True) -> EvalResult:
        """
        Run benchmark and record result.
        
        Args:
            individual_id: ID of individual to evaluate
            timeout: Benchmark timeout in seconds
            revert: If True, revert changes after eval (default for `run` command).
                    For composable CLI, pass revert=False and handle revert yourself.
        """
        timeout = timeout or self.config.experiment_timeout
        
        # Run benchmark
        self.log(f"Running benchmark (timeout: {timeout}s)...")
        result = run_benchmark(
            command=self.config.train_command,
            work_dir=self.config.work_dir,
            metric_name=self.config.metric_name,
            timeout=timeout,
            log_callback=self.log,
        )
        
        # Revert if requested (for `run` command with SingleFileReverter)
        if revert:
            self.log("Reverting changes...")
            self.reverter.revert()
        
        # Record result
        if result.error:
            self.log(f"Benchmark failed: {result.error}")
            return self.record(individual_id, fitness=None, metrics={}, error=result.error)
        else:
            self.log(f"Benchmark completed in {result.duration:.1f}s")
            return self.record(individual_id, fitness=result.fitness, metrics=result.metrics)
    
    def record(
        self,
        individual_id: str,
        fitness: float = None,
        metrics: dict = None,
        error: str = None,
    ) -> EvalResult:
        """
        Record an evaluation result WITHOUT running benchmark or reverting.
        
        Use when the agent runs the benchmark itself.
        Call revert() separately if needed.
        """
        individual = self._get_individual(individual_id)
        if not individual:
            return EvalResult(
                individual_id=individual_id,
                fitness=None,
                metrics={},
                is_best=False,
                improvement=None,
                evaluations=self.evaluations,
                status="error",
                error=f"Individual {individual_id} not found",
            )
        
        # Record result
        individual.fitness = fitness if fitness is not None else float('inf')
        individual.metrics = metrics or {}
        self.evaluations += 1
        
        # Track baseline
        if all(v is None for v in individual.genes.values()):
            self.baseline_fitness = fitness
            # Capture initial baseline (never resets)
            if self.initial_baseline_fitness is None:
                self.initial_baseline_fitness = fitness
        
        # Track best
        is_best = False
        if fitness is not None and fitness != float('inf'):
            if self.best is None or self._is_better(fitness, self.best.fitness):
                self.best = individual
                self.best_at_eval = self.evaluations
                is_best = True
                self.log(f"NEW BEST: {fitness}")
            
            if self.absolute_best is None or self._is_better(fitness, self.absolute_best.fitness):
                self.absolute_best = individual
        
        # Calculate improvement
        improvement = self._calc_improvement(fitness)
        
        # Log to history
        self._log_history(individual, error)
        
        self.current_individual = None
        self._save()
        
        # Determine status
        if self._is_converged():
            status = "converged"
        elif self.config.max_evaluations and self.evaluations >= self.config.max_evaluations:
            status = "max_evals"
        else:
            status = "continue"
        
        return EvalResult(
            individual_id=individual_id,
            fitness=fitness,
            metrics=individual.metrics,
            is_best=is_best,
            improvement=improvement,
            evaluations=self.evaluations,
            status=status,
            error=error,
        )
    
    def revert(self):
        """Revert file changes. Use after record() if needed."""
        self.reverter.revert()
    
    def rethink(
        self,
        add_ideas: list[Idea] = None,
        remove_ideas: list[str] = None,
        commit_best: bool = False,
    ) -> dict:
        """
        Analyze results and optionally modify ideas.
        
        Era increments if ANY significant change happens:
        - Commit best (reset baseline)
        - Ideas added or removed
        
        Args:
            add_ideas: New ideas to add
            remove_ideas: Idea names to remove
            commit_best: If True, commit best config as new baseline
        
        Returns:
            Analysis dict with idea statistics
        """
        added = []
        removed = []
        committed = False
        
        # Commit best if requested (this resets population)
        if commit_best and self.best:
            self._commit_best()
            committed = True
        
        # Add new ideas
        if add_ideas:
            for idea in add_ideas:
                if idea.name not in self.ideas:
                    self.ideas[idea.name] = idea
                    added.append(idea.name)
            if added:
                self._normalize_all_genes()
        
        # Remove ideas
        if remove_ideas:
            for name in remove_ideas:
                if name in self.ideas:
                    del self.ideas[name]
                    removed.append(name)
            if removed:
                self._normalize_all_genes()
        
        # Increment era if ideas changed AND we didn't already commit
        # (commit_best already increments era)
        if (added or removed) and not committed:
            self.era += 1
        
        self.last_rethink = self.evaluations
        self._save()
        
        analysis = self._get_analysis()
        analysis["added"] = added
        analysis["removed"] = removed
        
        return analysis
    
    def status(self) -> dict:
        """Get current session state."""
        # Use initial baseline for improvement calculation (never resets across eras)
        best = self.absolute_best or self.best
        improvement_baseline = self.initial_baseline_fitness or self.baseline_fitness
        return {
            "session": self.config.name,
            "evaluations": self.evaluations,
            "generation": self.generation,
            "era": self.era,
            "population_size": self.config.population_size,
            "ideas": len(self.ideas),
            "best": self._best_dict(),
            "baseline_fitness": self.baseline_fitness,
            "initial_baseline_fitness": self.initial_baseline_fitness,
            "improvement": self._calc_improvement(best.fitness if best else None, improvement_baseline),
            "converged": self._is_converged(),
            "evals_since_improvement": self.evaluations - self.best_at_eval,
            "current_individual": self.current_individual,
        }
    
    def stop(self) -> dict:
        """Finalize session and generate reports."""
        from .charts import generate_charts
        
        # Generate summary
        summary = self._generate_summary()
        summary_path = self.session_dir / "RESULTS.md"
        summary_path.write_text(summary)
        
        # Generate charts
        try:
            chart_paths = generate_charts(self)
            chart_files = [p.name for p in chart_paths]
        except Exception as e:
            self.log(f"Chart generation failed: {e}")
            chart_files = []
        
        return {
            "session": self.config.name,
            "evaluations": self.evaluations,
            "best": self._best_dict(),
            "baseline_fitness": self.baseline_fitness,
            "initial_baseline_fitness": self.initial_baseline_fitness,
            "results_dir": str(self.session_dir),
            "files": ["RESULTS.md"] + chart_files,
        }
    
    def should_rethink(self) -> bool:
        """Check if it's time to rethink."""
        if self.config.rethink_interval <= 0:
            return False
        return self.evaluations - self.last_rethink >= self.config.rethink_interval
    
    # =========================================================================
    # GA Operations
    # =========================================================================
    
    def _initialize_population(self):
        """Create initial population."""
        self.population = [self._create_baseline()]
        while len(self.population) < self.config.population_size:
            self.population.append(self._create_random())
        self.log(f"Created population of {len(self.population)}")
    
    def _create_individual(self, genes: dict, generation: int = None) -> Individual:
        """Create an individual with normalized genes."""
        if generation is None:
            generation = self.generation
        full_genes = {name: genes.get(name) for name in self.ideas}
        return Individual(
            id=f"ind-{random.randint(0, 999999):06d}",
            genes=full_genes,
            generation=generation,
        )
    
    def _create_baseline(self) -> Individual:
        """Create baseline individual (all genes off)."""
        return self._create_individual({}, 0)
    
    def _create_random(self) -> Individual:
        """Create individual with random genes."""
        genes = {}
        for name, idea in self.ideas.items():
            if random.random() < 0.5:
                genes[name] = None
            elif idea.is_binary():
                genes[name] = "on"
            else:
                genes[name] = random.choice(idea.variants)
        return self._create_individual(genes)
    
    def _get_next_unevaluated(self) -> Optional[Individual]:
        """Get next individual without fitness."""
        for ind in self.population:
            if ind.fitness is None:
                return ind
        return None
    
    def _get_individual(self, individual_id: str) -> Optional[Individual]:
        """Find individual by ID."""
        for ind in self.population:
            if ind.id == individual_id:
                return ind
        return None
    
    def _evolve(self):
        """Create next generation."""
        self.generation += 1
        self.log(f"\n=== Generation {self.generation} ===")
        
        # Sort by fitness (best first, Infinity last)
        evaluated = [i for i in self.population if i.fitness is not None]
        evaluated.sort(
            key=lambda i: i.fitness,
            reverse=(self.config.metric_direction == "higher")
        )
        
        # Elitism
        new_population = []
        for i in range(min(self.config.elitism, len(evaluated))):
            if evaluated[i].fitness != float('inf'):
                elite = self._create_individual(evaluated[i].genes.copy())
                elite.fitness = evaluated[i].fitness
                elite.metrics = evaluated[i].metrics.copy()
                new_population.append(elite)
        
        # Fill with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self._select(evaluated)
            parent2 = self._select(evaluated)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        self.population = new_population
    
    def _select(self, evaluated: list[Individual]) -> Individual:
        """Tournament selection."""
        if not evaluated:
            return random.choice(self.population)
        
        valid = [i for i in evaluated if i.fitness != float('inf')]
        if not valid:
            return random.choice(evaluated)
        
        tournament_size = min(3, len(valid))
        tournament = random.sample(valid, tournament_size)
        
        if self.config.metric_direction == "lower":
            return min(tournament, key=lambda i: i.fitness)
        return max(tournament, key=lambda i: i.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Uniform crossover."""
        # Always record parent IDs (per DESIGN.md)
        if random.random() > self.config.crossover_rate:
            # No crossover - copy genes from random parent
            parent = random.choice([parent1, parent2])
            child = self._create_individual(parent.genes.copy())
        else:
            # Uniform crossover - each gene from either parent
            genes = {}
            for name in self.ideas:
                genes[name] = random.choice([
                    parent1.genes.get(name),
                    parent2.genes.get(name)
                ])
            child = self._create_individual(genes)
        
        child.parents = (parent1.id, parent2.id)
        return child
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutate genes."""
        for name, idea in self.ideas.items():
            if random.random() < self.config.mutation_rate:
                current = individual.genes.get(name)
                if idea.is_binary():
                    individual.genes[name] = None if current == "on" else "on"
                else:
                    options = list(idea.variants) + [None]
                    options = [v for v in options if v != current]
                    if options:
                        individual.genes[name] = random.choice(options)
        return individual
    
    def _normalize_all_genes(self):
        """Normalize genes for all individuals after idea changes."""
        for ind in self.population:
            # Add missing
            for name in self.ideas:
                if name not in ind.genes:
                    ind.genes[name] = None
            # Remove old
            for name in list(ind.genes.keys()):
                if name not in self.ideas:
                    del ind.genes[name]
    
    def _commit_best(self):
        """
        Lock in best config as new baseline for next era.
        
        For `run` command (SingleFileReverter): takes new snapshot of current file state.
        For composable CLI: agent should have already committed changes via git.
        
        Does NOT do git operations - agent handles git.
        """
        self.log(f"New baseline: {self.best.describe()}")
        
        # Take new snapshot - current state becomes the new baseline
        # For SingleFileReverter: caches current file content
        # For composable CLI: agent already committed, this is a no-op for GitReverter
        self.reverter.snapshot()
        
        # Log era transition to history
        self._log_era_transition()
        
        # Reset state for new era
        self.era += 1
        self.best = None
        self.best_at_eval = self.evaluations
        self.baseline_fitness = None  # Will be re-measured
        self.generation = 0
        self.population = []
        self._initialize_population()
    
    def _log_era_transition(self):
        """Log era transition to history."""
        entry = {
            "event": "era_transition",
            "timestamp": time.time(),
            "era": self.era,
            "evaluation": self.evaluations,
            "best_fitness": self.best.fitness if self.best else None,
            "best_genes": self.best.genes if self.best else None,
        }
        self.history.append(entry)
        with open(self.session_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _is_converged(self) -> bool:
        if self.best is None:
            return False
        return self.evaluations - self.best_at_eval >= self.config.convergence_evals
    
    def _is_better(self, a: float, b: float) -> bool:
        if self.config.metric_direction == "lower":
            return a < b
        return a > b
    
    def _best_dict(self) -> Optional[dict]:
        best = self.absolute_best or self.best
        if not best:
            return None
        # Use initial baseline for improvement (never resets across eras)
        improvement_baseline = self.initial_baseline_fitness or self.baseline_fitness
        return {
            "id": best.id,
            "fitness": best.fitness,
            "genes": {k: v for k, v in best.genes.items() if v is not None},
            "improvement": self._calc_improvement(best.fitness, improvement_baseline),
        }
    
    def _best_dict_with_genes_to_implement(self) -> Optional[dict]:
        """Best dict with full gene details for agent to implement."""
        best = self.absolute_best or self.best
        if not best:
            return None
        
        genes_to_implement = []
        for name, value in best.genes.items():
            if value is not None:
                idea = self.ideas.get(name)
                genes_to_implement.append({
                    "name": name,
                    "value": value,
                    "description": idea.description if idea else "",
                    "is_binary": idea.is_binary() if idea else False,
                })
        
        # Use initial baseline for improvement (never resets across eras)
        improvement_baseline = self.initial_baseline_fitness or self.baseline_fitness
        return {
            "id": best.id,
            "fitness": best.fitness,
            "genes": {k: v for k, v in best.genes.items() if v is not None},
            "genes_to_implement": genes_to_implement,
            "improvement": self._calc_improvement(best.fitness, improvement_baseline),
        }
    
    def _calc_improvement(self, fitness: float, baseline: float = None) -> Optional[str]:
        """Calculate improvement percentage vs baseline.
        
        Args:
            fitness: The fitness value to compare
            baseline: The baseline to compare against (defaults to current era baseline)
        """
        if baseline is None:
            baseline = self.baseline_fitness
        if not fitness or not baseline:
            return None
        pct = (fitness - baseline) / baseline * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"
    
    def _get_analysis(self) -> dict:
        """Analyze idea effectiveness."""
        # Use initial baseline for overall improvement (never resets across eras)
        best = self.absolute_best or self.best
        improvement_baseline = self.initial_baseline_fitness or self.baseline_fitness
        analysis = {
            "evaluations": self.evaluations,
            "best_fitness": best.fitness if best else None,
            "baseline_fitness": self.baseline_fitness,
            "initial_baseline_fitness": self.initial_baseline_fitness,
            "improvement": self._calc_improvement(best.fitness if best else None, improvement_baseline),
            "ideas": {},
            "era": self.era,
        }
        
        for name in self.ideas:
            on_fitnesses = []
            off_fitnesses = []
            
            for entry in self.history:
                if "genes" not in entry or entry.get("fitness") is None:
                    continue
                
                fitness = entry["fitness"]
                if entry["genes"].get(name) is not None:
                    on_fitnesses.append(fitness)
                else:
                    off_fitnesses.append(fitness)
            
            success_count = 0
            if self.baseline_fitness:
                for f in on_fitnesses:
                    if self._is_better(f, self.baseline_fitness):
                        success_count += 1
            
            analysis["ideas"][name] = {
                "eval_count": len(on_fitnesses),
                "success_rate": success_count / len(on_fitnesses) if on_fitnesses else 0,
                "avg_fitness_on": sum(on_fitnesses) / len(on_fitnesses) if on_fitnesses else None,
                "avg_fitness_off": sum(off_fitnesses) / len(off_fitnesses) if off_fitnesses else None,
            }
        
        return analysis
    
    def _generate_summary(self) -> str:
        """Generate RESULTS.md content."""
        lines = [f"# cEvolve Results: {self.config.name}", ""]
        
        best = self.absolute_best or self.best
        
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Evaluations**: {self.evaluations}")
        lines.append(f"- **Generations**: {self.generation + 1}")
        lines.append(f"- **Eras**: {self.era + 1}")
        
        # Show initial baseline (the one from start of run)
        initial_baseline = self.initial_baseline_fitness or self.baseline_fitness
        if initial_baseline is not None:
            lines.append(f"- **Baseline**: {initial_baseline}")
        
        if best and best.fitness:
            improvement = self._calc_improvement(best.fitness, initial_baseline) or ""
            lines.append(f"- **Best**: {best.fitness} {improvement}")
        
        lines.append("")
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
        
        return "\n".join(lines)
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _save(self):
        """Save session state to disk."""
        # Config
        config_data = {
            "name": self.config.name,
            "population_size": self.config.population_size,
            "max_evaluations": self.config.max_evaluations,
            "elitism": self.config.elitism,
            "mutation_rate": self.config.mutation_rate,
            "crossover_rate": self.config.crossover_rate,
            "experiment_timeout": self.config.experiment_timeout,
            "train_command": self.config.train_command,
            "work_dir": str(self.config.work_dir),
            "target_file": self.config.target_file,
            "metric_name": self.config.metric_name,
            "metric_direction": self.config.metric_direction,
            "rethink_interval": self.config.rethink_interval,
            "convergence_evals": self.config.convergence_evals,
        }
        with open(self.session_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        # Ideas
        ideas_data = [
            {"name": i.name, "description": i.description, "variants": i.variants}
            for i in self.ideas.values()
        ]
        with open(self.session_dir / "ideas.json", "w") as f:
            json.dump(ideas_data, f, indent=2)
        
        # State
        state = {
            "evaluations": self.evaluations,
            "generation": self.generation,
            "era": self.era,
            "best": self._individual_to_dict(self.best),
            "absolute_best": self._individual_to_dict(self.absolute_best),
            "best_at_eval": self.best_at_eval,
            "baseline_fitness": self.baseline_fitness,
            "initial_baseline_fitness": self.initial_baseline_fitness,
            "last_rethink": self.last_rethink,
            "current_individual": self.current_individual,
        }
        with open(self.session_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        # Population
        pop_data = [self._individual_to_dict(i) for i in self.population]
        with open(self.session_dir / "population.json", "w") as f:
            json.dump(pop_data, f, indent=2)
        
        # Update current marker
        current_file = self.config.work_dir / ".cevolve" / ".current"
        current_file.write_text(self.config.name)
    
    def _individual_to_dict(self, ind: Optional[Individual]) -> Optional[dict]:
        if ind is None:
            return None
        return {
            "id": ind.id,
            "genes": ind.genes,
            "fitness": ind.fitness if ind.fitness != float('inf') else None,
            "metrics": ind.metrics,
            "generation": ind.generation,
            "parents": ind.parents,
        }
    
    def _log_history(self, individual: Individual, error: str = None):
        """Log evaluation to history."""
        entry = {
            "timestamp": time.time(),
            "evaluation": self.evaluations,
            "generation": self.generation,
            "id": individual.id,
            "genes": individual.genes,
            "fitness": individual.fitness if individual.fitness != float('inf') else None,
            "metrics": individual.metrics,
            "error": error,
        }
        self.history.append(entry)
        
        # Append to file
        with open(self.session_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
