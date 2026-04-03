"""Tests for evolution (generation transition)."""

import pytest
from evolve.core import Individual, GARunner


class TestEvolution:
    """Test evolve() behavior."""
    
    def test_generation_increment(self, runner_with_population):
        """Evolve increments generation counter."""
        runner = runner_with_population
        
        # Give fitness to all
        for i, ind in enumerate(runner.population):
            ind.fitness = 1.0 + i * 0.1
        
        initial_gen = runner.generation
        runner.evolve()
        
        assert runner.generation == initial_gen + 1
    
    def test_population_size_maintained(self, runner_with_population):
        """Population size stays the same."""
        runner = runner_with_population
        
        for ind in runner.population:
            ind.fitness = 1.5
        
        initial_size = len(runner.population)
        runner.evolve()
        
        assert len(runner.population) == initial_size
    
    def test_elitism_preserves_best(self, runner_with_population):
        """Top N individuals are preserved."""
        runner = runner_with_population
        runner.config.elitism = 2
        runner.config.metric_direction = "lower"
        
        # Assign fitness - lower is better
        fitnesses = [1.5, 1.0, 1.8, 1.2, 1.3]
        for i, ind in enumerate(runner.population):
            ind.fitness = fitnesses[i]
        
        runner.evolve()
        
        # Best two (1.0 and 1.2) should be in new population
        new_fitnesses = [ind.fitness for ind in runner.population if ind.fitness is not None]
        assert 1.0 in new_fitnesses
        assert 1.2 in new_fitnesses
    
    def test_elite_fitness_preserved(self, runner_with_population):
        """Elite individuals keep their fitness values."""
        runner = runner_with_population
        runner.config.elitism = 2
        
        for i, ind in enumerate(runner.population):
            ind.fitness = 1.0 + i * 0.1
        
        runner.evolve()
        
        # First two (elites) should have fitness
        elites = [ind for ind in runner.population if ind.fitness is not None]
        assert len(elites) >= 2
    
    def test_elite_metrics_preserved(self, runner_with_population):
        """Elite individuals keep their metrics."""
        runner = runner_with_population
        runner.config.elitism = 1
        
        # Best individual has metrics
        runner.population[0].fitness = 0.5  # Best
        runner.population[0].metrics = {"train_time": 45.2, "memory": 1024}
        for ind in runner.population[1:]:
            ind.fitness = 1.5
        
        runner.evolve()
        
        # Find the elite (lowest fitness)
        elite = min(runner.population, key=lambda i: i.fitness if i.fitness else float('inf'))
        assert elite.fitness == 0.5
        assert elite.metrics == {"train_time": 45.2, "memory": 1024}
    
    def test_children_need_evaluation(self, runner_with_population):
        """Non-elite children have fitness=None."""
        runner = runner_with_population
        runner.config.elitism = 2
        
        for ind in runner.population:
            ind.fitness = 1.5
        
        runner.evolve()
        
        # Children (after elites) should have None fitness
        children = [ind for ind in runner.population if ind.fitness is None]
        expected_children = runner.config.population_size - runner.config.elitism
        assert len(children) == expected_children
    
    def test_evolve_with_infinity_fitness(self, runner_with_population):
        """Infinity fitness individuals are not selected as elites."""
        runner = runner_with_population
        runner.config.elitism = 2
        
        runner.population[0].fitness = float('inf')
        runner.population[1].fitness = 1.0
        runner.population[2].fitness = float('inf')
        runner.population[3].fitness = 1.5
        runner.population[4].fitness = float('inf')
        
        runner.evolve()
        
        # Elites should be 1.0 and 1.5, not infinity
        elite_fitnesses = [ind.fitness for ind in runner.population if ind.fitness is not None]
        assert float('inf') not in elite_fitnesses[:2]
    
    def test_children_have_correct_generation(self, runner_with_population):
        """New individuals have the new generation number."""
        runner = runner_with_population
        runner.generation = 5
        
        for ind in runner.population:
            ind.fitness = 1.5
        
        runner.evolve()
        
        for ind in runner.population:
            assert ind.generation == 6
