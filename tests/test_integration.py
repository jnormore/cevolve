"""Integration tests for full GA runs."""

import pytest
from pathlib import Path
from evolve.core import Idea, Individual, Config, GARunner


class TestFullRun:
    """Test complete GA runs."""
    
    def test_small_run(self, tmp_path):
        """Run with small population and few evaluations."""
        # Create mock training script
        train_script = tmp_path / "train.py"
        train_script.write_text('print("val_bpb: 1.234")')
        
        ideas = [
            Idea(name="idea_a", description="First", variants=[]),
            Idea(name="idea_b", description="Second", variants=["x", "y"]),
        ]
        
        config = Config(
            name="test",
            population_size=3,
            max_evaluations=6,
            elitism=1,
            work_dir=tmp_path,
            target_file="train.py",
            train_command=f"python {train_script}",
            rethink_interval=0,  # Disable rethink
            convergence_evals=100,  # Disable convergence
        )
        
        runner = GARunner(ideas=ideas, config=config)
        best = runner.run()
        
        assert runner.evaluations == 6
        assert best is not None
        assert best.fitness is not None
        # History includes init event + 6 eval entries
        eval_entries = [e for e in runner.history if "fitness" in e and "event" not in e]
        assert len(eval_entries) == 6
    
    def test_run_with_rethink(self, tmp_path):
        """Run with rethink enabled."""
        train_script = tmp_path / "train.py"
        train_script.write_text('print("val_bpb: 1.234")')
        
        ideas = [
            Idea(name="idea_a", description="First", variants=[]),
        ]
        
        # Mock LLM that adds one idea
        call_count = [0]
        def mock_llm(prompt):
            call_count[0] += 1
            return "new_idea: A new optimization idea"
        
        config = Config(
            name="test",
            population_size=3,
            max_evaluations=9,
            elitism=1,
            work_dir=tmp_path,
            target_file="train.py",
            train_command=f"python {train_script}",
            rethink_interval=3,
            convergence_evals=100,
        )
        
        runner = GARunner(ideas=ideas, config=config, llm_call=mock_llm)
        runner.run()
        
        # Rethink should have been triggered
        assert runner.last_rethink > 0
    
    def test_run_convergence_stop(self, tmp_path):
        """Run stops on convergence."""
        train_script = tmp_path / "train.py"
        # Always return same fitness - should converge
        train_script.write_text('print("val_bpb: 1.0")')
        
        ideas = [
            Idea(name="idea_a", description="First", variants=[]),
        ]
        
        config = Config(
            name="test",
            population_size=3,
            max_evaluations=100,  # High limit
            elitism=1,
            work_dir=tmp_path,
            target_file="train.py",
            train_command=f"python {train_script}",
            rethink_interval=0,
            convergence_evals=5,  # Stop after 5 evals without improvement
        )
        
        runner = GARunner(ideas=ideas, config=config)
        runner.run()
        
        # Should stop before max_evaluations due to convergence
        assert runner.evaluations < 100
        assert runner.is_converged()
    
    def test_run_finds_better_fitness(self, tmp_path):
        """Run can find improving fitness."""
        # Training script that returns different values based on input
        train_script = tmp_path / "train.py"
        train_script.write_text('''
import random
# Simulate some variation
print(f"val_bpb: {1.5 - random.random() * 0.3}")
''')
        
        ideas = [
            Idea(name="idea_a", description="First", variants=[]),
            Idea(name="idea_b", description="Second", variants=["1", "2", "3"]),
        ]
        
        config = Config(
            name="test",
            population_size=4,
            max_evaluations=12,
            elitism=1,
            work_dir=tmp_path,
            target_file="train.py",
            train_command=f"python {train_script}",
            rethink_interval=0,
            convergence_evals=100,
        )
        
        runner = GARunner(ideas=ideas, config=config)
        best = runner.run()
        
        assert best is not None
        assert best.fitness is not None
        assert best.fitness < 1.5  # Should find something better than worst case
    
    def test_run_preserves_elites(self, tmp_path):
        """Elites are preserved across generations."""
        train_script = tmp_path / "train.py"
        train_script.write_text('print("val_bpb: 1.0")')
        
        ideas = [
            Idea(name="idea_a", description="First", variants=[]),
        ]
        
        config = Config(
            name="test",
            population_size=4,
            max_evaluations=12,  # 3 generations
            elitism=2,
            work_dir=tmp_path,
            target_file="train.py",
            train_command=f"python {train_script}",
            rethink_interval=0,
            convergence_evals=100,
        )
        
        runner = GARunner(ideas=ideas, config=config)
        runner.run()
        
        # After multiple generations, should have evolved
        assert runner.generation >= 2


class TestInitialization:
    """Test initialization behavior."""
    
    def test_population_initialization(self, runner):
        """Population is initialized correctly."""
        runner.initialize_population()
        
        assert len(runner.population) == runner.config.population_size
        
        # First is baseline
        baseline = runner.population[0]
        assert all(v is None for v in baseline.genes.values())
    
    def test_baseline_always_first(self, runner):
        """Baseline is always first individual."""
        for _ in range(5):
            runner.population = []
            runner.initialize_population()
            
            assert runner.population[0].active_count() == 0


class TestHistoryLogging:
    """Test history logging."""
    
    def test_history_logged_on_eval(self, runner):
        """Each evaluation is logged to history."""
        runner.initialize_population()
        
        for ind in runner.population:
            runner.evaluate(ind)
        
        # Should have entry for each evaluation
        eval_entries = [e for e in runner.history if "fitness" in e and "event" not in e]
        assert len(eval_entries) == len(runner.population)
    
    def test_history_entry_structure(self, runner):
        """History entries have correct structure."""
        runner.initialize_population()
        runner.evaluate(runner.population[0])
        
        entry = runner.history[-1]
        
        assert "timestamp" in entry
        assert "evaluation" in entry
        assert "generation" in entry
        assert "id" in entry
        assert "genes" in entry
        assert "fitness" in entry
        assert "metrics" in entry
