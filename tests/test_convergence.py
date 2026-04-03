"""Tests for convergence detection."""

import pytest
from evolve.core import Individual, GARunner, Config


class TestConvergenceDetection:
    """Test is_converged() behavior."""
    
    def test_not_converged_no_best(self, runner):
        """Not converged if no best yet."""
        runner.best = None
        runner.evaluations = 100
        runner.best_at_eval = 0
        
        assert not runner.is_converged()
    
    def test_converged_no_improvement(self, runner):
        """Converged when no improvement for convergence_evals."""
        runner.config.convergence_evals = 10
        runner.best = runner.create_baseline_individual()
        runner.best.fitness = 1.0
        runner.evaluations = 15
        runner.best_at_eval = 5
        
        # 15 - 5 = 10 >= 10
        assert runner.is_converged()
    
    def test_not_converged_recent_improvement(self, runner):
        """Not converged if improvement was recent."""
        runner.config.convergence_evals = 10
        runner.best = runner.create_baseline_individual()
        runner.best.fitness = 1.0
        runner.evaluations = 15
        runner.best_at_eval = 12
        
        # 15 - 12 = 3 < 10
        assert not runner.is_converged()
    
    def test_converged_exactly_at_threshold(self, runner):
        """Converged at exactly convergence_evals."""
        runner.config.convergence_evals = 10
        runner.best = runner.create_baseline_individual()
        runner.best.fitness = 1.0
        runner.evaluations = 20
        runner.best_at_eval = 10
        
        # 20 - 10 = 10 >= 10
        assert runner.is_converged()
    
    def test_not_converged_one_before_threshold(self, runner):
        """Not converged one eval before threshold."""
        runner.config.convergence_evals = 10
        runner.best = runner.create_baseline_individual()
        runner.best.fitness = 1.0
        runner.evaluations = 19
        runner.best_at_eval = 10
        
        # 19 - 10 = 9 < 10
        assert not runner.is_converged()


class TestDefaultConvergenceEvals:
    """Test default convergence_evals calculation."""
    
    def test_default_is_3x_rethink_plus_1(self):
        """Default convergence_evals = rethink_interval * 3 + 1."""
        config = Config(rethink_interval=10)
        assert config.convergence_evals == 31
    
    def test_default_with_different_rethink(self):
        """Default works with different rethink_interval."""
        config = Config(rethink_interval=5)
        assert config.convergence_evals == 16
    
    def test_explicit_overrides_default(self):
        """Explicit convergence_evals overrides default."""
        config = Config(rethink_interval=10, convergence_evals=50)
        assert config.convergence_evals == 50
    
    def test_zero_rethink_interval(self):
        """Works with rethink_interval=0."""
        config = Config(rethink_interval=0)
        assert config.convergence_evals == 1  # 0 * 3 + 1
