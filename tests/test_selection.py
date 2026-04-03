"""Tests for tournament selection."""

import pytest
from evolve.core import Individual, GARunner


class TestTournamentSelection:
    """Test tournament selection behavior."""
    
    def test_selection_returns_individual(self, runner_with_population):
        """Selection returns an Individual from population."""
        runner = runner_with_population
        
        # Give some fitness values
        for i, ind in enumerate(runner.population):
            ind.fitness = 1.0 + i * 0.1
        
        selected = runner.select()
        assert isinstance(selected, Individual)
        assert selected in runner.population
    
    def test_selection_prefers_lower_fitness(self, runner_with_population):
        """With direction=lower, lower fitness is preferred."""
        runner = runner_with_population
        runner.config.metric_direction = "lower"
        
        # Assign distinct fitness values
        runner.population[0].fitness = 2.0
        runner.population[1].fitness = 1.0  # Best
        runner.population[2].fitness = 3.0
        runner.population[3].fitness = 2.5
        runner.population[4].fitness = 1.5
        
        # Select many times - best should be selected more often
        selections = [runner.select() for _ in range(100)]
        best_count = sum(1 for s in selections if s.fitness == 1.0)
        
        # With tournament of 3, best should win when in tournament
        assert best_count > 10  # Should be selected fairly often
    
    def test_selection_prefers_higher_fitness(self, runner_with_population):
        """With direction=higher, higher fitness is preferred."""
        runner = runner_with_population
        runner.config.metric_direction = "higher"
        
        # Assign distinct fitness values
        runner.population[0].fitness = 2.0
        runner.population[1].fitness = 3.0  # Best
        runner.population[2].fitness = 1.0
        runner.population[3].fitness = 2.5
        runner.population[4].fitness = 1.5
        
        # Select many times
        selections = [runner.select() for _ in range(100)]
        best_count = sum(1 for s in selections if s.fitness == 3.0)
        
        assert best_count > 10
    
    def test_selection_excludes_none_fitness(self, runner_with_population):
        """Individuals with fitness=None are excluded from tournament."""
        runner = runner_with_population
        
        # Only one has fitness
        runner.population[0].fitness = 1.5
        # Rest are None
        
        # Selection should always return the one with fitness
        for _ in range(20):
            selected = runner.select()
            assert selected.fitness == 1.5
    
    def test_selection_excludes_infinity(self, runner_with_population):
        """Individuals with fitness=inf are excluded from tournament."""
        runner = runner_with_population
        
        runner.population[0].fitness = 1.5
        runner.population[1].fitness = float('inf')
        runner.population[2].fitness = float('inf')
        runner.population[3].fitness = 2.0
        runner.population[4].fitness = float('inf')
        
        # Selection should only pick from valid
        for _ in range(20):
            selected = runner.select()
            assert selected.fitness in [1.5, 2.0]
    
    def test_selection_all_invalid_returns_random(self, runner_with_population):
        """If all fitness is None/inf, returns random individual."""
        runner = runner_with_population
        
        # All invalid
        runner.population[0].fitness = None
        runner.population[1].fitness = float('inf')
        runner.population[2].fitness = None
        runner.population[3].fitness = float('inf')
        runner.population[4].fitness = None
        
        # Should not crash, returns something from population
        selected = runner.select()
        assert selected in runner.population
