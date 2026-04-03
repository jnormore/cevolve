"""Tests for mutation operation."""

import pytest
from evolve.core import Individual, Idea, GARunner


class TestMutation:
    """Test mutation behavior."""
    
    def test_binary_toggle_off_to_on(self, runner):
        """Binary gene None -> 'on'."""
        runner.config.mutation_rate = 1.0  # Always mutate
        
        ind = runner.create_individual({"flash_attn": None, "depth": None})
        runner.mutate(ind)
        
        # flash_attn should toggle to "on"
        assert ind.genes["flash_attn"] == "on"
    
    def test_binary_toggle_on_to_off(self, runner):
        """Binary gene 'on' -> None."""
        runner.config.mutation_rate = 1.0
        
        ind = runner.create_individual({"flash_attn": "on", "depth": None})
        runner.mutate(ind)
        
        assert ind.genes["flash_attn"] is None
    
    def test_variant_mutation_different_value(self, runner):
        """Variant gene picks DIFFERENT value."""
        runner.config.mutation_rate = 1.0
        
        # depth has variants ["4", "6", "8"]
        ind = runner.create_individual({"flash_attn": None, "depth": "6"})
        
        # Mutate many times - should never stay "6"
        for _ in range(20):
            ind.genes["depth"] = "6"  # Reset
            runner.mutate(ind)
            assert ind.genes["depth"] != "6", "Should pick different value"
    
    def test_variant_mutation_can_turn_off(self, runner):
        """Variant gene can mutate to None."""
        runner.config.mutation_rate = 1.0
        
        ind = runner.create_individual({"flash_attn": None, "depth": "6"})
        
        # Mutate many times - should eventually see None
        seen_none = False
        for _ in range(50):
            ind.genes["depth"] = "6"
            runner.mutate(ind)
            if ind.genes["depth"] is None:
                seen_none = True
                break
        
        assert seen_none, "Should eventually mutate to None"
    
    def test_variant_mutation_off_to_on(self, runner):
        """Variant gene None -> picks from variants."""
        runner.config.mutation_rate = 1.0
        
        ind = runner.create_individual({"flash_attn": None, "depth": None})
        
        # Mutate - depth should become one of ["4", "6", "8"]
        runner.mutate(ind)
        
        assert ind.genes["depth"] in ["4", "6", "8"]
    
    def test_mutation_rate_zero_no_change(self, runner):
        """With rate=0, nothing changes."""
        runner.config.mutation_rate = 0.0
        
        ind = runner.create_individual({"flash_attn": "on", "depth": "6"})
        original = ind.genes.copy()
        
        runner.mutate(ind)
        
        assert ind.genes == original
    
    def test_mutation_invalidates_code(self, runner):
        """Mutation sets code to None."""
        ind = runner.create_individual({"flash_attn": "on", "depth": "6"})
        ind.code = "some code"
        
        runner.config.mutation_rate = 1.0
        runner.mutate(ind)
        
        assert ind.code is None
    
    def test_mutation_returns_individual(self, runner):
        """Mutate returns the individual."""
        ind = runner.create_individual({"flash_attn": "on", "depth": "6"})
        
        result = runner.mutate(ind)
        
        assert result is ind


class TestMutationProbability:
    """Test that mutation respects probability."""
    
    def test_mutation_rate_affects_frequency(self, runner):
        """Higher rate = more mutations."""
        # Count mutations at different rates
        def count_mutations(rate, trials=100):
            runner.config.mutation_rate = rate
            mutations = 0
            for _ in range(trials):
                ind = runner.create_individual({"flash_attn": "on", "depth": "6"})
                original = ind.genes.copy()
                runner.mutate(ind)
                if ind.genes != original:
                    mutations += 1
            return mutations
        
        low_rate_mutations = count_mutations(0.1)
        high_rate_mutations = count_mutations(0.5)
        
        # High rate should have more mutations
        assert high_rate_mutations > low_rate_mutations
