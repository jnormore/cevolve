"""Tests for evaluation and metric parsing."""

import pytest
from evolve.core import Individual, GARunner, Config


class TestMetricParsing:
    """Test _parse_metrics method."""
    
    def test_colon_format(self, runner):
        """Parse 'name: value' format."""
        output = "val_bpb: 1.234\ntrain_time: 45.2\n"
        
        metrics = runner._parse_metrics(output)
        
        assert metrics["val_bpb"] == pytest.approx(1.234)
        assert metrics["train_time"] == pytest.approx(45.2)
    
    def test_metric_keyword_format(self, runner):
        """Parse 'METRIC name=value' format."""
        output = "METRIC val_bpb=1.234\nMETRIC train_time=45.2\n"
        
        metrics = runner._parse_metrics(output)
        
        assert metrics["val_bpb"] == pytest.approx(1.234)
        assert metrics["train_time"] == pytest.approx(45.2)
    
    def test_mixed_formats(self, runner):
        """Parse mixed formats in same output."""
        output = "val_bpb: 1.234\nMETRIC train_time=45.2\nmemory: 2048\n"
        
        metrics = runner._parse_metrics(output)
        
        assert metrics["val_bpb"] == pytest.approx(1.234)
        assert metrics["train_time"] == pytest.approx(45.2)
        assert metrics["memory"] == pytest.approx(2048)
    
    def test_scientific_notation(self, runner):
        """Parse scientific notation."""
        output = "loss: 1.5e-4\nMETRIC grad_norm=2.3E+2\n"
        
        metrics = runner._parse_metrics(output)
        
        assert metrics["loss"] == pytest.approx(1.5e-4)
        assert metrics["grad_norm"] == pytest.approx(2.3e+2)
    
    def test_negative_values(self, runner):
        """Parse negative values."""
        output = "delta: -0.5\nMETRIC change=-1.23\n"
        
        metrics = runner._parse_metrics(output)
        
        assert metrics["delta"] == pytest.approx(-0.5)
        assert metrics["change"] == pytest.approx(-1.23)
    
    def test_ignores_invalid_lines(self, runner):
        """Invalid lines are ignored."""
        output = "Starting training...\nval_bpb: 1.234\nDone!\n"
        
        metrics = runner._parse_metrics(output)
        
        assert len(metrics) == 1
        assert "val_bpb" in metrics
    
    def test_dotted_metric_names(self, runner):
        """Metric names with dots work."""
        output = "train.loss: 0.5\nMETRIC eval.accuracy=0.95\n"
        
        metrics = runner._parse_metrics(output)
        
        assert metrics["train.loss"] == pytest.approx(0.5)
        assert metrics["eval.accuracy"] == pytest.approx(0.95)


class TestEvaluation:
    """Test evaluate() behavior."""
    
    def test_evaluation_increments_counter(self, runner):
        """Evaluations counter increments."""
        runner.initialize_population()
        ind = runner.population[0]
        
        initial = runner.evaluations
        runner.evaluate(ind)
        
        assert runner.evaluations == initial + 1
    
    def test_evaluation_sets_fitness(self, runner):
        """Fitness is set from output."""
        runner.initialize_population()
        ind = runner.population[0]  # baseline
        
        runner.evaluate(ind)
        
        assert ind.fitness == pytest.approx(1.234)  # From mock train.py
    
    def test_evaluation_sets_metrics(self, runner):
        """Secondary metrics are captured."""
        runner.initialize_population()
        ind = runner.population[0]
        
        runner.evaluate(ind)
        
        assert "train_time" in ind.metrics
        assert ind.metrics["train_time"] == pytest.approx(45.2)
    
    def test_evaluation_updates_best(self, runner):
        """Best is updated when better fitness found."""
        runner.initialize_population()
        runner.config.metric_direction = "lower"
        
        # Evaluate first - becomes best
        runner.evaluate(runner.population[0])
        
        assert runner.best is not None
        assert runner.best.fitness == pytest.approx(1.234)
    
    def test_evaluation_updates_best_at_eval(self, runner):
        """best_at_eval tracks when best was found."""
        runner.initialize_population()
        
        runner.evaluate(runner.population[0])
        
        assert runner.best_at_eval == runner.evaluations
    
    def test_evaluation_clears_current_individual(self, runner):
        """current_individual is cleared after evaluation."""
        runner.initialize_population()
        
        runner.evaluate(runner.population[0])
        
        assert runner.current_individual is None
    
    def test_evaluation_logs_history(self, runner):
        """Evaluation is logged to history."""
        runner.initialize_population()
        
        runner.evaluate(runner.population[0])
        
        assert len(runner.history) >= 1
        entry = runner.history[-1]
        assert "fitness" in entry
        assert "genes" in entry
        assert "metrics" in entry


class TestEvaluationBestTracking:
    """Test best individual tracking."""
    
    def test_best_not_updated_if_worse(self, runner, tmp_work_dir):
        """Best not updated when new fitness is worse."""
        runner.initialize_population()
        runner.config.metric_direction = "lower"
        
        # First eval - becomes best
        runner.evaluate(runner.population[0])
        first_best = runner.best
        first_best_at = runner.best_at_eval
        
        # Modify train.py to return worse fitness
        (tmp_work_dir / "train.py").write_text('print("val_bpb: 2.0")')
        
        # Second eval - should not become best
        runner.evaluate(runner.population[1])
        
        assert runner.best is first_best
        assert runner.best_at_eval == first_best_at
    
    def test_best_updated_if_better(self, runner, tmp_work_dir):
        """Best updated when new fitness is better."""
        runner.initialize_population()
        runner.config.metric_direction = "lower"
        
        # First eval - will get 1.234
        runner.evaluate(runner.population[0])
        first_best = runner.best
        first_fitness = runner.best.fitness
        
        # Modify train.py AND runner's original_code to return better fitness
        better_code = 'print("val_bpb: 0.5")'
        (tmp_work_dir / "train.py").write_text(better_code)
        runner.original_code = better_code
        
        # Second eval - should become best since code changed
        runner.population[1].code = None  # Clear any cached code
        runner.evaluate(runner.population[1])
        
        assert runner.best is runner.population[1]
        assert runner.best.fitness == pytest.approx(0.5)
        assert runner.best.fitness < first_fitness
