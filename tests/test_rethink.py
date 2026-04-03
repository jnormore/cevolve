"""Tests for rethink mechanism."""

import pytest
from evolve.core import Individual, Idea, GARunner


class TestRethinkTrigger:
    """Test rethink triggering."""
    
    def test_maybe_rethink_at_interval(self, runner):
        """_maybe_rethink triggers at interval."""
        runner.config.rethink_interval = 5
        runner.last_rethink = 0
        runner.evaluations = 5
        
        # Should trigger (5 - 0 >= 5)
        initial_last = runner.last_rethink
        runner._maybe_rethink()
        
        assert runner.last_rethink == 5
    
    def test_maybe_rethink_not_before_interval(self, runner):
        """_maybe_rethink doesn't trigger before interval."""
        runner.config.rethink_interval = 5
        runner.last_rethink = 0
        runner.evaluations = 4
        
        runner._maybe_rethink()
        
        assert runner.last_rethink == 0  # Unchanged
    
    def test_maybe_rethink_disabled_with_zero(self, runner):
        """_maybe_rethink disabled when interval=0."""
        runner.config.rethink_interval = 0
        runner.last_rethink = 0
        runner.evaluations = 100
        
        runner._maybe_rethink()
        
        assert runner.last_rethink == 0  # Never updates


class TestRethinkStatistics:
    """Test get_rethink_statistics method."""
    
    def test_statistics_structure(self, runner):
        """Statistics has expected structure."""
        runner.initialize_population()
        
        stats = runner.get_rethink_statistics()
        
        assert "evaluations" in stats
        assert "best_fitness" in stats
        assert "baseline_fitness" in stats
        assert "ideas" in stats
    
    def test_statistics_per_idea(self, runner):
        """Statistics has per-idea data."""
        runner.initialize_population()
        
        stats = runner.get_rethink_statistics()
        
        for name in runner.ideas:
            assert name in stats["ideas"]
            idea_stats = stats["ideas"][name]
            assert "eval_count" in idea_stats
            assert "success_count" in idea_stats
            assert "variants" in idea_stats
    
    def test_statistics_counts_evaluations(self, runner):
        """Eval count reflects history."""
        runner.initialize_population()
        
        # Add some history entries
        runner.history.append({
            "genes": {"flash_attn": "on", "depth": "6"},
            "fitness": 1.0,
        })
        runner.history.append({
            "genes": {"flash_attn": "on", "depth": None},
            "fitness": 1.2,
        })
        runner.history.append({
            "genes": {"flash_attn": None, "depth": "6"},
            "fitness": 1.5,
        })
        
        stats = runner.get_rethink_statistics()
        
        assert stats["ideas"]["flash_attn"]["eval_count"] == 2  # on in 2 entries
        assert stats["ideas"]["depth"]["eval_count"] == 2  # non-None in 2 entries


class TestRethinkWithLLM:
    """Test rethink with LLM."""
    
    def test_rethink_updates_last_rethink(self, runner):
        """Rethink updates last_rethink."""
        runner.evaluations = 10
        runner.last_rethink = 0
        
        runner.rethink()
        
        assert runner.last_rethink == 10
    
    def test_rethink_without_llm(self, runner):
        """Rethink works without LLM (no-op)."""
        runner.llm_call = None
        runner.evaluations = 10
        
        # Should not crash
        runner.rethink()
        
        assert runner.last_rethink == 10


class TestRethinkAccumulation:
    """Test accumulation of improvements on rethink."""
    
    def test_rethink_commits_best_code(self, runner):
        """Rethink commits best code as new baseline."""
        runner.initialize_population()
        runner.config.rethink_interval = 5
        runner.evaluations = 5
        runner.last_rethink = 0
        
        # Set up a best individual with modified code
        runner.best = runner.population[0]
        runner.best.fitness = 1.5
        runner.best.code = "# Modified code\nprint('hello')"
        
        original_before = runner.original_code
        
        runner._maybe_rethink()
        
        # original_code should be updated to best's code
        assert runner.original_code == "# Modified code\nprint('hello')"
        assert runner.original_code != original_before
    
    def test_rethink_increments_era(self, runner):
        """Rethink increments era when committing."""
        runner.initialize_population()
        runner.config.rethink_interval = 5
        runner.evaluations = 5
        runner.last_rethink = 0
        
        runner.best = runner.population[0]
        runner.best.fitness = 1.5
        runner.best.code = "# Modified"
        
        era_before = runner.era
        
        runner._maybe_rethink()
        
        assert runner.era == era_before + 1
    
    def test_rethink_resets_population(self, runner):
        """Rethink resets population from new baseline."""
        runner.initialize_population()
        runner.config.rethink_interval = 5
        runner.evaluations = 5
        runner.last_rethink = 0
        
        runner.best = runner.population[0]
        runner.best.fitness = 1.5
        runner.best.code = "# Modified"
        
        old_pop_ids = [ind.id for ind in runner.population]
        
        runner._maybe_rethink()
        
        # Population should be re-initialized with new IDs
        new_pop_ids = [ind.id for ind in runner.population]
        assert new_pop_ids != old_pop_ids
        assert len(runner.population) == runner.config.population_size
    
    def test_rethink_clears_best(self, runner):
        """Rethink clears best for fresh start."""
        runner.initialize_population()
        runner.config.rethink_interval = 5
        runner.evaluations = 5
        runner.last_rethink = 0
        
        runner.best = runner.population[0]
        runner.best.fitness = 1.5
        runner.best.code = "# Modified"
        
        runner._maybe_rethink()
        
        assert runner.best is None
    
    def test_rethink_logs_era_transition(self, runner):
        """Rethink logs era transition in history."""
        runner.initialize_population()
        runner.config.rethink_interval = 5
        runner.evaluations = 5
        runner.last_rethink = 0
        
        runner.best = runner.population[0]
        runner.best.fitness = 1.5
        runner.best.code = "# Modified"
        runner.best.genes = {"flash_attn": "on", "depth": None}
        
        runner._maybe_rethink()
        
        # Should have era_transition entry
        transitions = [e for e in runner.history if e.get("event") == "era_transition"]
        assert len(transitions) == 1
        assert transitions[0]["fitness"] == 1.5
    
    def test_rethink_no_commit_without_best(self, runner):
        """Rethink doesn't commit if no best."""
        runner.initialize_population()
        runner.config.rethink_interval = 5
        runner.evaluations = 5
        runner.last_rethink = 0
        runner.best = None
        
        original_before = runner.original_code
        era_before = runner.era
        
        runner._maybe_rethink()
        
        # Should not change baseline or era
        assert runner.original_code == original_before
        assert runner.era == era_before


class TestParseIdeas:
    """Test _parse_ideas method."""
    
    def test_parse_binary_idea(self, runner):
        """Parse binary idea (no variants line)."""
        response = "flash_attn: Use flash attention"
        
        ideas = runner._parse_ideas(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "flash_attn"
        assert ideas[0].description == "Use flash attention"
        assert ideas[0].variants == []
    
    def test_parse_variant_idea(self, runner):
        """Parse idea with variants."""
        response = """depth: Model depth
  variants: 4, 6, 8"""
        
        ideas = runner._parse_ideas(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "depth"
        assert ideas[0].variants == ["4", "6", "8"]
    
    def test_parse_multiple_ideas(self, runner):
        """Parse multiple ideas."""
        response = """idea_a: First idea
idea_b: Second idea
  variants: x, y, z"""
        
        ideas = runner._parse_ideas(response)
        
        assert len(ideas) == 2
        assert ideas[0].name == "idea_a"
        assert ideas[1].name == "idea_b"
        assert ideas[1].variants == ["x", "y", "z"]
    
    def test_parse_normalizes_name(self, runner):
        """Name is normalized (lowercase, underscores)."""
        response = "My-Idea Name: Description"
        
        ideas = runner._parse_ideas(response)
        
        assert ideas[0].name == "my_idea_name"
