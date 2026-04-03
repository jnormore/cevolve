"""Tests for gene normalization."""

import pytest
from evolve.core import Individual, Idea, GARunner


class TestNormalizeGenes:
    """Test normalize_genes behavior."""
    
    def test_add_missing_ideas(self, runner):
        """Missing ideas are added as None."""
        # Create individual with only one gene
        ind = Individual(
            id="test",
            genes={"flash_attn": "on"},
        )
        
        runner.normalize_genes(ind)
        
        assert "flash_attn" in ind.genes
        assert "depth" in ind.genes
        assert ind.genes["flash_attn"] == "on"  # Preserved
        assert ind.genes["depth"] is None  # Added as None
    
    def test_remove_obsolete_ideas(self, runner):
        """Ideas not in pool are removed."""
        ind = Individual(
            id="test",
            genes={"flash_attn": "on", "depth": "6", "obsolete": "value"},
        )
        
        runner.normalize_genes(ind)
        
        assert "flash_attn" in ind.genes
        assert "depth" in ind.genes
        assert "obsolete" not in ind.genes
    
    def test_combined_add_and_remove(self, runner):
        """Add missing and remove obsolete at once."""
        # Start with only flash_attn in ideas
        runner.ideas = {"flash_attn": runner.ideas["flash_attn"]}
        
        # Individual has old_idea but not flash_attn
        ind = Individual(
            id="test",
            genes={"old_idea": "x"},
        )
        
        runner.normalize_genes(ind)
        
        assert "flash_attn" in ind.genes
        assert ind.genes["flash_attn"] is None
        assert "old_idea" not in ind.genes
    
    def test_preserve_existing_values(self, runner):
        """Existing valid values are preserved."""
        ind = Individual(
            id="test",
            genes={"flash_attn": "on", "depth": "6"},
        )
        
        runner.normalize_genes(ind)
        
        assert ind.genes["flash_attn"] == "on"
        assert ind.genes["depth"] == "6"
    
    def test_normalize_empty_genes(self, runner):
        """Empty genes get all ideas as None."""
        ind = Individual(id="test", genes={})
        
        runner.normalize_genes(ind)
        
        assert len(ind.genes) == 2
        assert all(v is None for v in ind.genes.values())


class TestAddRemoveIdeas:
    """Test add_ideas and remove_ideas methods."""
    
    def test_add_ideas_normalizes_population(self, runner_with_population):
        """Adding ideas normalizes all individuals."""
        runner = runner_with_population
        
        new_idea = Idea(name="new_idea", description="New", variants=["a", "b"])
        runner.add_ideas([new_idea])
        
        # All individuals should have the new idea
        for ind in runner.population:
            assert "new_idea" in ind.genes
            assert ind.genes["new_idea"] is None  # Added as off
    
    def test_add_ideas_increments_era(self, runner_with_population):
        """Adding ideas increments era."""
        runner = runner_with_population
        initial_era = runner.era
        
        new_idea = Idea(name="new_idea", description="New", variants=[])
        runner.add_ideas([new_idea])
        
        assert runner.era == initial_era + 1
    
    def test_add_duplicate_idea_ignored(self, runner_with_population):
        """Adding existing idea is ignored."""
        runner = runner_with_population
        initial_era = runner.era
        
        # Try to add existing idea
        dup_idea = Idea(name="flash_attn", description="Dup", variants=[])
        added = runner.add_ideas([dup_idea])
        
        assert added == []
        assert runner.era == initial_era  # No change
    
    def test_remove_ideas_normalizes_population(self, runner_with_population):
        """Removing ideas normalizes all individuals."""
        runner = runner_with_population
        
        # Set some values first
        for ind in runner.population:
            ind.genes["depth"] = "6"
        
        runner.remove_ideas(["depth"])
        
        # All individuals should not have depth
        for ind in runner.population:
            assert "depth" not in ind.genes
    
    def test_remove_ideas_increments_era(self, runner_with_population):
        """Removing ideas increments era."""
        runner = runner_with_population
        initial_era = runner.era
        
        runner.remove_ideas(["depth"])
        
        assert runner.era == initial_era + 1
    
    def test_remove_nonexistent_ignored(self, runner_with_population):
        """Removing nonexistent idea is ignored."""
        runner = runner_with_population
        initial_era = runner.era
        
        removed = runner.remove_ideas(["nonexistent"])
        
        assert removed == []
        assert runner.era == initial_era
