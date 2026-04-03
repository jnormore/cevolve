"""Tests for crossover operation."""

import pytest
from evolve.core import Individual, GARunner


class TestCrossover:
    """Test crossover behavior."""
    
    def test_crossover_records_parents(self, runner):
        """Child always records both parent IDs."""
        parent1 = runner.create_individual({"flash_attn": "on", "depth": "4"})
        parent2 = runner.create_individual({"flash_attn": None, "depth": "8"})
        
        child = runner.crossover(parent1, parent2)
        
        assert child.parents is not None
        assert child.parents == (parent1.id, parent2.id)
    
    def test_crossover_parents_recorded_even_no_crossover(self, runner):
        """Parents recorded even when crossover doesn't happen (rate=0)."""
        runner.config.crossover_rate = 0.0
        
        parent1 = runner.create_individual({"flash_attn": "on", "depth": "4"})
        parent2 = runner.create_individual({"flash_attn": None, "depth": "8"})
        
        child = runner.crossover(parent1, parent2)
        
        assert child.parents == (parent1.id, parent2.id)
    
    def test_crossover_genes_from_parents(self, runner):
        """Child genes come from either parent."""
        parent1 = runner.create_individual({"flash_attn": "on", "depth": "4"})
        parent2 = runner.create_individual({"flash_attn": None, "depth": "8"})
        
        # Try many times to catch both cases
        for _ in range(50):
            child = runner.crossover(parent1, parent2)
            
            # Each gene must be from one of the parents
            assert child.genes["flash_attn"] in ["on", None]
            assert child.genes["depth"] in ["4", "8"]
    
    def test_crossover_no_rate_copies_parent(self, runner):
        """With crossover_rate=0, child is exact copy of one parent."""
        runner.config.crossover_rate = 0.0
        
        parent1 = runner.create_individual({"flash_attn": "on", "depth": "4"})
        parent2 = runner.create_individual({"flash_attn": None, "depth": "8"})
        
        for _ in range(20):
            child = runner.crossover(parent1, parent2)
            
            # Should be exact copy of one parent
            is_p1 = (child.genes["flash_attn"] == "on" and child.genes["depth"] == "4")
            is_p2 = (child.genes["flash_attn"] is None and child.genes["depth"] == "8")
            assert is_p1 or is_p2, f"Child should be copy of parent, got {child.genes}"
    
    def test_crossover_full_rate_mixes(self, runner):
        """With crossover_rate=1.0, genes are mixed from both parents."""
        runner.config.crossover_rate = 1.0
        
        parent1 = runner.create_individual({"flash_attn": "on", "depth": "4"})
        parent2 = runner.create_individual({"flash_attn": None, "depth": "8"})
        
        # With enough tries, we should see mixed children
        seen_mixed = False
        for _ in range(50):
            child = runner.crossover(parent1, parent2)
            
            is_p1 = (child.genes["flash_attn"] == "on" and child.genes["depth"] == "4")
            is_p2 = (child.genes["flash_attn"] is None and child.genes["depth"] == "8")
            if not is_p1 and not is_p2:
                seen_mixed = True
                break
        
        assert seen_mixed, "Should see mixed children with full crossover"
    
    def test_crossover_covers_all_ideas(self, runner, three_ideas):
        """Child has all ideas from pool."""
        runner.ideas = {idea.name: idea for idea in three_ideas}
        
        parent1 = runner.create_individual({"idea_a": "on", "idea_b": "x", "idea_c": "1"})
        parent2 = runner.create_individual({"idea_a": None, "idea_b": "y", "idea_c": "2"})
        
        child = runner.crossover(parent1, parent2)
        
        assert len(child.genes) == 3
        assert "idea_a" in child.genes
        assert "idea_b" in child.genes
        assert "idea_c" in child.genes
    
    def test_crossover_child_has_null_fitness(self, runner):
        """New child starts with fitness=None."""
        parent1 = runner.create_individual({"flash_attn": "on", "depth": "4"})
        parent2 = runner.create_individual({"flash_attn": None, "depth": "8"})
        parent1.fitness = 1.0
        parent2.fitness = 2.0
        
        child = runner.crossover(parent1, parent2)
        
        assert child.fitness is None
