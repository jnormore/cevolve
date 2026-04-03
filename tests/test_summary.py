"""Tests for summary generation."""

import pytest
from evolve.core import Individual, GARunner


class TestGenerateSummary:
    """Test generate_summary method."""
    
    def test_summary_has_title(self, runner):
        """Summary has title with session name."""
        runner.config.name = "test-session"
        
        summary = runner.generate_summary()
        
        assert "# cEvolve Results: test-session" in summary
    
    def test_summary_has_sections(self, runner):
        """Summary has expected sections."""
        runner.initialize_population()
        
        summary = runner.generate_summary()
        
        assert "## Summary" in summary
        assert "## Winning Configuration" in summary
        assert "## Ideas Effectiveness" in summary
    
    def test_summary_shows_evaluations(self, runner):
        """Summary shows evaluation count."""
        runner.evaluations = 25
        
        summary = runner.generate_summary()
        
        assert "25" in summary
        assert "Evaluations" in summary
    
    def test_summary_shows_best_fitness(self, runner):
        """Summary shows best fitness."""
        runner.initialize_population()
        runner.best = runner.population[0]
        runner.best.fitness = 1.234
        
        summary = runner.generate_summary()
        
        assert "1.234" in summary
        assert "Best" in summary
    
    def test_summary_shows_improvement(self, runner):
        """Summary shows improvement percentage."""
        runner.initialize_population()
        
        # Add baseline to history
        runner.history.append({
            "genes": {"flash_attn": None, "depth": None},
            "fitness": 2.0,
        })
        
        # Best is better
        runner.best = runner.population[0]
        runner.best.fitness = 1.5
        runner.best.genes = {"flash_attn": "on", "depth": None}
        
        summary = runner.generate_summary()
        
        # Should show -25% improvement (1.5 - 2.0) / 2.0 = -0.25
        assert "-25" in summary or "25" in summary
    
    def test_summary_winning_config_baseline(self, runner):
        """Summary shows baseline when best has no active genes."""
        runner.initialize_population()
        runner.best = runner.create_baseline_individual()
        runner.best.fitness = 1.0
        
        summary = runner.generate_summary()
        
        assert "Baseline" in summary or "baseline" in summary
    
    def test_summary_winning_config_with_genes(self, runner):
        """Summary shows active genes in winning config."""
        runner.initialize_population()
        runner.best = runner.create_individual({"flash_attn": "on", "depth": "6"})
        runner.best.fitness = 1.0
        
        summary = runner.generate_summary()
        
        assert "flash_attn" in summary
        assert "depth" in summary
        assert "6" in summary
    
    def test_summary_ideas_table(self, runner):
        """Summary has ideas effectiveness table."""
        runner.initialize_population()
        
        # Add some history
        runner.history.append({
            "genes": {"flash_attn": "on", "depth": "6"},
            "fitness": 1.0,
        })
        runner.history.append({
            "genes": {"flash_attn": "on", "depth": None},
            "fitness": 1.2,
        })
        
        summary = runner.generate_summary()
        
        # Should have table structure
        assert "| Idea |" in summary or "Idea" in summary
    
    def test_summary_no_best(self, runner):
        """Summary handles no best gracefully."""
        runner.best = None
        
        summary = runner.generate_summary()
        
        assert "No successful" in summary or "Winning Configuration" in summary
