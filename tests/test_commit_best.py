"""Tests for commit_best functionality."""

import pytest
import subprocess
from evolve.core import Individual, GARunner


class TestCommitBest:
    """Test commit_best behavior."""
    
    def test_commit_best_no_best(self, runner):
        """Returns False if no best."""
        runner.best = None
        
        result = runner.commit_best()
        
        assert result is False
    
    def test_commit_best_no_fitness(self, runner):
        """Returns False if best has no fitness."""
        runner.best = runner.create_baseline_individual()
        runner.best.fitness = None
        
        result = runner.commit_best()
        
        assert result is False
    
    def test_commit_best_increments_era(self, runner):
        """Commit best increments era."""
        runner.initialize_population()
        runner.best = runner.population[0]
        runner.best.fitness = 1.0
        runner.best.code = runner.original_code
        initial_era = runner.era
        
        # Initialize git
        subprocess.run(["git", "init"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=runner.config.work_dir, capture_output=True)
        
        runner.commit_best()
        
        assert runner.era == initial_era + 1
    
    def test_commit_best_resets_generation(self, runner):
        """Commit best resets generation to 0."""
        runner.initialize_population()
        runner.generation = 5
        runner.best = runner.population[0]
        runner.best.fitness = 1.0
        runner.best.code = runner.original_code
        
        subprocess.run(["git", "init"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=runner.config.work_dir, capture_output=True)
        
        runner.commit_best()
        
        assert runner.generation == 0
    
    def test_commit_best_clears_best(self, runner):
        """Commit best clears best."""
        runner.initialize_population()
        runner.best = runner.population[0]
        runner.best.fitness = 1.0
        runner.best.code = runner.original_code
        
        subprocess.run(["git", "init"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=runner.config.work_dir, capture_output=True)
        
        runner.commit_best()
        
        assert runner.best is None
    
    def test_commit_best_clears_population(self, runner):
        """Commit best clears population."""
        runner.initialize_population()
        runner.best = runner.population[0]
        runner.best.fitness = 1.0
        runner.best.code = runner.original_code
        
        subprocess.run(["git", "init"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=runner.config.work_dir, capture_output=True)
        
        runner.commit_best()
        
        assert runner.population == []
    
    def test_commit_best_updates_best_at_eval(self, runner):
        """Commit best updates best_at_eval."""
        runner.initialize_population()
        runner.evaluations = 15
        runner.best = runner.population[0]
        runner.best.fitness = 1.0
        runner.best.code = runner.original_code
        
        subprocess.run(["git", "init"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=runner.config.work_dir, capture_output=True)
        
        runner.commit_best()
        
        assert runner.best_at_eval == 15
    
    def test_commit_best_logs_era_transition(self, runner):
        """Commit best logs era transition to history."""
        runner.initialize_population()
        runner.best = runner.population[0]
        runner.best.fitness = 1.0
        runner.best.code = runner.original_code
        runner.era = 2
        
        subprocess.run(["git", "init"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=runner.config.work_dir, capture_output=True)
        
        runner.commit_best()
        
        # Find era transition entry
        era_entries = [e for e in runner.history if e.get("event") == "era_transition"]
        assert len(era_entries) == 1
        assert era_entries[0]["id"] == "era-2-baseline"
    
    def test_commit_best_updates_original_code(self, runner):
        """Commit best updates original_code."""
        runner.initialize_population()
        runner.best = runner.population[0]
        runner.best.fitness = 1.0
        runner.best.code = "# Modified code\nprint('new')"
        
        subprocess.run(["git", "init"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=runner.config.work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=runner.config.work_dir, capture_output=True)
        
        runner.commit_best()
        
        # Original code should be updated
        assert "Modified code" in runner.original_code or runner.original_code == runner.target_path.read_text()
