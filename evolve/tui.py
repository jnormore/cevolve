"""
TUI (Terminal User Interface) for evolution monitoring.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, RichLog, Static, DataTable
from textual.reactive import reactive

from .core import Idea


class FitnessGraph(Static):
    """ASCII fitness graph showing progress over evaluations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitness_history: list[tuple[int, float]] = []  # (eval_num, fitness)
        self.best_history: list[tuple[int, float]] = []  # (eval_num, best_so_far)
        self.baseline: float | None = None
        self.direction: str = "lower"
        self.width = 50
        self.height = 8
    
    def add_point(self, eval_num: int, fitness: float, best_so_far: float):
        """Add a fitness data point."""
        self.fitness_history.append((eval_num, fitness))
        self.best_history.append((eval_num, best_so_far))
        self.refresh_graph()
    
    def set_baseline(self, baseline: float):
        """Set the baseline fitness."""
        self.baseline = baseline
        self.refresh_graph()
    
    def refresh_graph(self):
        """Redraw the graph."""
        if not self.fitness_history:
            self.update("[dim]Waiting for data...[/dim]")
            return
        
        # Get all fitness values for scaling
        all_values = [f for _, f in self.fitness_history]
        best_values = [f for _, f in self.best_history]
        if self.baseline:
            all_values.append(self.baseline)
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        # Avoid division by zero
        if max_val == min_val:
            max_val = min_val + 1
        
        # Build the graph
        lines = []
        
        # Title with stats
        best = min(all_values) if self.direction == "lower" else max(all_values)
        lines.append(f"[bold cyan]Fitness Progress[/]  Best: [green]{best:.1f}[/]")
        lines.append("")
        
        # Create graph area
        graph_width = min(self.width, len(self.fitness_history))
        
        # Sample points if we have more than width
        if len(self.fitness_history) > graph_width:
            step = len(self.fitness_history) / graph_width
            sampled = [self.fitness_history[int(i * step)] for i in range(graph_width)]
            sampled_best = [self.best_history[int(i * step)] for i in range(graph_width)]
        else:
            sampled = self.fitness_history
            sampled_best = self.best_history
        
        # Calculate which row each point falls into
        def get_row(value):
            if max_val == min_val:
                return 0
            normalized = (value - min_val) / (max_val - min_val)
            return int(normalized * (self.height - 1))
        
        # Build rows from top to bottom
        for row in range(self.height - 1, -1, -1):
            # Y-axis label
            if row == self.height - 1:
                label = f"{max_val:>7.0f} │"
            elif row == 0:
                label = f"{min_val:>7.0f} │"
            else:
                label = "        │"
            
            row_chars = []
            for i, (_, fitness) in enumerate(sampled):
                _, best_fitness = sampled_best[i]
                fitness_row = get_row(fitness)
                best_row = get_row(best_fitness)
                
                # Check if this point falls in this row
                if fitness_row == row:
                    row_chars.append("[yellow]●[/]")
                elif best_row == row:
                    row_chars.append("[green]─[/]")
                else:
                    row_chars.append(" ")
            
            lines.append(label + "".join(row_chars))
        
        # X-axis
        lines.append("        └" + "─" * len(sampled))
        evals_str = str(len(self.fitness_history))
        padding = len(sampled) - len(evals_str) - 1
        x_label = f"         0{' ' * max(0, padding)}{evals_str}"
        lines.append(x_label)
        
        self.update("\n".join(lines))


class RacingLeaderboard(Static):
    """Racing-style leaderboard with position changes."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entries: list[dict] = []  # {id, fitness, genes, prev_rank, is_new, is_evaluating}
        self.direction: str = "lower"
        self.max_entries = 6
    
    def update_population(self, population: list, current_id: str | None = None, direction: str = "lower"):
        """Update leaderboard with current population."""
        self.direction = direction
        
        # Build previous rank map
        prev_ranks = {e["id"]: i for i, e in enumerate(self.entries)}
        
        # Sort population by fitness
        evaluated = [(ind, ind.fitness) for ind in population if ind.fitness is not None]
        if direction == "lower":
            evaluated.sort(key=lambda x: x[1])
        else:
            evaluated.sort(key=lambda x: x[1], reverse=True)
        
        # Build new entries
        new_entries = []
        for rank, (ind, fitness) in enumerate(evaluated[:self.max_entries]):
            # Get active genes
            active = [f"{k}={v}" if v != "on" else k for k, v in ind.genes.items() if v is not None]
            genes_str = ", ".join(active[:2]) if active else "baseline"
            if len(active) > 2:
                genes_str += f" +{len(active)-2}"
            
            prev_rank = prev_ranks.get(ind.id)
            is_new = prev_rank is None
            
            new_entries.append({
                "id": ind.id,
                "fitness": fitness,
                "genes": genes_str,
                "prev_rank": prev_rank,
                "current_rank": rank,
                "is_new": is_new,
                "is_evaluating": ind.id == current_id,
            })
        
        # Add currently evaluating individual if not in list
        if current_id:
            evaluating_ind = None
            for ind in population:
                if ind.id == current_id and ind.fitness is None:
                    evaluating_ind = ind
                    break
            
            if evaluating_ind and len(new_entries) < self.max_entries:
                active = [f"{k}={v}" if v != "on" else k for k, v in evaluating_ind.genes.items() if v is not None]
                genes_str = ", ".join(active[:2]) if active else "baseline"
                new_entries.append({
                    "id": evaluating_ind.id,
                    "fitness": None,
                    "genes": genes_str,
                    "prev_rank": None,
                    "current_rank": len(new_entries),
                    "is_new": True,
                    "is_evaluating": True,
                })
        
        self.entries = new_entries
        self.refresh_leaderboard()
    
    def refresh_leaderboard(self):
        """Redraw the leaderboard."""
        lines = []
        lines.append("[bold cyan]Leaderboard[/]")
        
        for entry in self.entries:
            rank = entry["current_rank"] + 1
            prev_rank = entry["prev_rank"]
            
            # Movement indicator
            if entry["is_evaluating"] and entry["fitness"] is None:
                move = "[yellow]~[/]"
            elif entry["is_new"]:
                move = "[cyan]*[/]"
            elif prev_rank is not None:
                diff = prev_rank - entry["current_rank"]
                if diff > 0:
                    move = f"[green]↑[/]"
                elif diff < 0:
                    move = f"[red]↓[/]"
                else:
                    move = "[dim]=[/]"
            else:
                move = " "
            
            # Rank with color
            if rank == 1:
                rank_fmt = f"[bold yellow]{rank}.[/]"
            elif rank == 2:
                rank_fmt = f"[bold white]{rank}.[/]"
            elif rank == 3:
                rank_fmt = f"[yellow]{rank}.[/]"
            else:
                rank_fmt = f"[dim]{rank}.[/]"
            
            # Fitness
            if entry["fitness"] is not None:
                fitness_str = f"[green]{entry['fitness']:>7.1f}[/]"
            else:
                fitness_str = "[yellow]    ...[/]"
            
            # ID (shortened)
            short_id = f"[dim]{entry['id'][-6:]}[/]"
            
            # Genes
            genes = entry["genes"][:28]
            
            # Simple format without box borders
            line = f"  {rank_fmt} {move} {short_id} {fitness_str}  {genes}"
            lines.append(line)
        
        # Pad if fewer entries
        for _ in range(self.max_entries - len(self.entries)):
            lines.append("")
        
        self.update("\n".join(lines))


class StatusBar(Static):
    """Status bar showing current progress."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = None
    
    def update_session(self, session):
        self.session = session
        self.refresh_status()
    
    def refresh_status(self):
        if not self.session:
            self.update("[dim]Initializing...[/dim]")
            return
        
        s = self.session
        
        max_evals = s.config.max_evaluations or "∞"
        best = s.absolute_best or s.best
        best_str = f"{best.fitness:.4f}" if best and best.fitness else "-"
        
        baseline = s.baseline_fitness
        if baseline:
            baseline_str = f"{baseline:.4f}"
        elif s.evaluations == 0 and s.current_individual:
            baseline_str = "(evaluating...)"
        else:
            baseline_str = "-"
        
        # Calculate improvement from current era baseline
        if best and best.fitness and baseline:
            pct = (best.fitness - baseline) / baseline * 100
            sign = "+" if pct >= 0 else ""
            is_good = (s.config.metric_direction == "lower" and pct < 0) or (s.config.metric_direction == "higher" and pct > 0)
            style = "green" if is_good else "red" if pct != 0 else "dim"
            improvement = f"[{style}]{sign}{pct:.1f}%[/{style}]"
        else:
            improvement = "-"
        
        # Calculate cumulative improvement from original baseline (across all eras)
        initial_baseline = getattr(s, 'initial_baseline_fitness', None)
        cumulative_str = ""
        if initial_baseline and best and best.fitness and initial_baseline != baseline:
            cum_pct = (best.fitness - initial_baseline) / initial_baseline * 100
            cum_sign = "+" if cum_pct >= 0 else ""
            is_good = (s.config.metric_direction == "lower" and cum_pct < 0) or (s.config.metric_direction == "higher" and cum_pct > 0)
            cum_style = "green" if is_good else "red"
            cumulative_str = f"  [bold cyan]Total:[/] [{cum_style}]{cum_sign}{cum_pct:.1f}%[/{cum_style}] [dim](from {initial_baseline:.1f})[/]"
        
        # Current individual
        current = "[dim]idle[/dim]"
        if s.current_individual:
            for ind in s.population:
                if ind.id == s.current_individual:
                    active = [f"{k}={v}" if v != "on" else k for k, v in ind.genes.items() if v is not None]
                    current = ", ".join(active) if active else "baseline"
                    break
        
        status = (
            f"[bold cyan]Eval:[/] {s.evaluations}/{max_evals}  "
            f"[bold cyan]Gen:[/] {s.generation}  "
            f"[bold cyan]Best:[/] {best_str}  "
            f"[bold cyan]Baseline:[/] [yellow]{baseline_str}[/]  "
            f"[bold cyan]Improvement:[/] {improvement}{cumulative_str}\n"
            f"[bold cyan]Current:[/] {current}"
        )
        
        self.update(status)


class IdeasTable(DataTable):
    """Table showing optimization ideas."""
    
    def __init__(self, ideas: list[Idea] = None, **kwargs):
        super().__init__(**kwargs)
        self.ideas = ideas or []
    
    def on_mount(self):
        self.add_columns("Idea", "Description", "Variants")
        self.cursor_type = "row"
        
        for idea in self.ideas:
            variants = ", ".join(idea.variants) if idea.variants else "[binary]"
            self.add_row(
                idea.name,
                idea.description[:60],
                variants[:50],
            )


class EvolveTUI(App):
    """Textual app for evolutionary optimization."""
    
    CSS = """
    #main-container {
        height: 1fr;
    }
    
    #log {
        height: 1fr;
        border: solid $primary;
    }
    
    #visualizations {
        width: 62;
        height: 100%;
        display: none;
        padding: 0 1;
    }
    
    #visualizations.visible {
        display: block;
    }
    
    #fitness-graph {
        height: auto;
        min-height: 12;
        border: solid green;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    #leaderboard {
        height: auto;
        min-height: 10;
        border: solid cyan;
        padding: 0 1;
    }
    
    #ideas {
        height: auto;
        max-height: 40%;
        border: solid green;
        display: none;
    }
    
    #ideas.visible {
        display: block;
    }
    
    #status {
        height: auto;
        min-height: 3;
        padding: 0 1;
        border: solid $accent;
    }
    
    DataTable {
        height: auto;
        max-height: 100%;
    }
    
    RichLog {
        scrollbar-gutter: stable;
    }
    """
    
    BINDINGS = [
        Binding("d", "toggle_details", "Details"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]
    
    show_details = reactive(True)
    
    def __init__(self, ideas: list[Idea] = None, **kwargs):
        super().__init__(**kwargs)
        self.ideas = ideas or []
        self.session = None
        self._log_widget: RichLog | None = None
        self._status_widget: StatusBar | None = None
        self._fitness_graph: FitnessGraph | None = None
        self._leaderboard: RacingLeaderboard | None = None
        self._last_history_len = 0
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            yield RichLog(id="log", highlight=True, markup=True, wrap=True)
            with Vertical(id="visualizations"):
                yield FitnessGraph(id="fitness-graph")
                yield RacingLeaderboard(id="leaderboard")
        yield IdeasTable(self.ideas, id="ideas")
        yield StatusBar(id="status")
        yield Footer()
    
    def on_mount(self):
        self._log_widget = self.query_one("#log", RichLog)
        self._status_widget = self.query_one("#status", StatusBar)
        self._fitness_graph = self.query_one("#fitness-graph", FitnessGraph)
        self._leaderboard = self.query_one("#leaderboard", RacingLeaderboard)
        self._log_widget.write("[dim]Starting evolutionary optimization...[/dim]")
        self._log_widget.write("[dim]Press 'd' to toggle details view[/dim]")
        
        # Show details by default
        viz_panel = self.query_one("#visualizations")
        ideas_table = self.query_one("#ideas")
        viz_panel.set_class(True, "visible")
        ideas_table.set_class(True, "visible")
    
    def action_toggle_details(self):
        self.show_details = not self.show_details
        viz_panel = self.query_one("#visualizations")
        ideas_table = self.query_one("#ideas")
        viz_panel.set_class(self.show_details, "visible")
        ideas_table.set_class(self.show_details, "visible")
    
    def add_log(self, msg: str):
        """Add a log line."""
        if self._log_widget:
            self._log_widget.write(msg)
    
    def update_session(self, session):
        """Update the session reference and refresh status."""
        self.session = session
        if self._status_widget:
            self._status_widget.update_session(session)
        
        # Update visualizations
        if session and self._fitness_graph:
            self._fitness_graph.direction = session.config.metric_direction
            
            # Set baseline from initial baseline (doesn't change across eras)
            initial_baseline = getattr(session, 'initial_baseline_fitness', None)
            if initial_baseline and self._fitness_graph.baseline is None:
                self._fitness_graph.set_baseline(initial_baseline)
            elif session.baseline_fitness and self._fitness_graph.baseline is None:
                self._fitness_graph.set_baseline(session.baseline_fitness)
            
            # Add new data points from history (survives rethinks)
            history = getattr(session, 'history', [])
            if len(history) > self._last_history_len:
                # Process new history entries
                for entry in history[self._last_history_len:]:
                    # Evaluation entries have 'fitness' key (not 'event' or 'best_fitness')
                    if 'fitness' in entry and entry.get('fitness') is not None and 'event' not in entry:
                        fitness = entry['fitness']
                        eval_num = entry.get('evaluation', len(self._fitness_graph.fitness_history) + 1)
                        
                        # Track best so far
                        best_so_far = fitness
                        if self._fitness_graph.fitness_history:
                            prev_best = min(f for _, f in self._fitness_graph.fitness_history) if session.config.metric_direction == 'lower' else max(f for _, f in self._fitness_graph.fitness_history)
                            if session.config.metric_direction == 'lower':
                                best_so_far = min(fitness, prev_best)
                            else:
                                best_so_far = max(fitness, prev_best)
                        
                        self._fitness_graph.add_point(eval_num, fitness, best_so_far)
                
                self._last_history_len = len(history)
        
        # Update leaderboard
        if session and self._leaderboard:
            self._leaderboard.update_population(
                session.population,
                current_id=session.current_individual,
                direction=session.config.metric_direction
            )
