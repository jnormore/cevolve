"""Tests for Individual and gene representation."""

import pytest
from evolve.core import Individual, Idea, GARunner, Config


class TestGeneRepresentation:
    """Test that genes use full map with None for off."""
    
    def test_individual_has_all_ideas(self, runner, sample_ideas):
        """Individual should have entry for every idea."""
        ind = runner.create_random_individual()
        
        assert len(ind.genes) == len(sample_ideas)
        for idea in sample_ideas:
            assert idea.name in ind.genes
    
    def test_baseline_all_none(self, runner, sample_ideas):
        """Baseline individual has all genes as None."""
        ind = runner.create_baseline_individual()
        
        assert len(ind.genes) == len(sample_ideas)
        for name, value in ind.genes.items():
            assert value is None, f"Baseline gene {name} should be None, got {value}"
    
    def test_random_individual_valid_values(self, runner, sample_ideas):
        """Random individual has valid values for each gene."""
        # Create many to check randomness
        for _ in range(20):
            ind = runner.create_random_individual()
            
            for name, value in ind.genes.items():
                idea = runner.ideas[name]
                if value is None:
                    pass  # Off is valid
                elif idea.is_binary():
                    assert value == "on", f"Binary gene {name} should be 'on' or None"
                else:
                    assert value in idea.variants, f"Variant gene {name} has invalid value {value}"
    
    def test_create_individual_fills_missing(self, runner):
        """create_individual fills missing genes with None."""
        # Pass partial genes
        partial_genes = {"flash_attn": "on"}
        ind = runner.create_individual(partial_genes, generation=0)
        
        # Should have all ideas
        assert "flash_attn" in ind.genes
        assert "depth" in ind.genes
        assert ind.genes["flash_attn"] == "on"
        assert ind.genes["depth"] is None  # Filled in


class TestIndividualDescribe:
    """Test Individual.describe() method."""
    
    def test_describe_baseline(self):
        """Baseline returns 'baseline'."""
        ind = Individual(
            id="test",
            genes={"a": None, "b": None},
        )
        assert ind.describe() == "baseline"
    
    def test_describe_binary_on(self):
        """Binary gene shows just name."""
        ind = Individual(
            id="test",
            genes={"flash_attn": "on", "depth": None},
        )
        assert "flash_attn" in ind.describe()
        assert "=" not in ind.describe() or "depth=" not in ind.describe()
    
    def test_describe_variant(self):
        """Variant gene shows name=value."""
        ind = Individual(
            id="test",
            genes={"flash_attn": None, "depth": "6"},
        )
        assert "depth=6" in ind.describe()
    
    def test_describe_mixed(self):
        """Mixed genes shows both formats."""
        ind = Individual(
            id="test",
            genes={"flash_attn": "on", "depth": "6"},
        )
        desc = ind.describe()
        assert "flash_attn" in desc
        assert "depth=6" in desc


class TestIndividualActiveCount:
    """Test Individual.active_count() method."""
    
    def test_active_count_zero(self):
        """Baseline has 0 active."""
        ind = Individual(id="test", genes={"a": None, "b": None})
        assert ind.active_count() == 0
    
    def test_active_count_all(self):
        """All active."""
        ind = Individual(id="test", genes={"a": "on", "b": "x"})
        assert ind.active_count() == 2
    
    def test_active_count_partial(self):
        """Some active."""
        ind = Individual(id="test", genes={"a": "on", "b": None, "c": "y"})
        assert ind.active_count() == 2
