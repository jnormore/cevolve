"""Test fixtures for GA tests."""

import pytest
from pathlib import Path
import tempfile
import os

from evolve.core import Idea, Individual, Config, GARunner, SecondaryMetric


@pytest.fixture
def sample_ideas():
    """Standard test ideas: 1 binary, 1 variant."""
    return [
        Idea(name="flash_attn", description="Use flash attention", variants=[]),
        Idea(name="depth", description="Model depth", variants=["4", "6", "8"]),
    ]


@pytest.fixture
def three_ideas():
    """Three ideas for testing."""
    return [
        Idea(name="idea_a", description="First idea", variants=[]),
        Idea(name="idea_b", description="Second idea", variants=["x", "y"]),
        Idea(name="idea_c", description="Third idea", variants=["1", "2", "3"]),
    ]


@pytest.fixture
def mock_llm():
    """LLM that returns no-op edits."""
    def llm_call(prompt: str) -> str:
        return ""  # No edits
    return llm_call


@pytest.fixture
def tmp_work_dir(tmp_path):
    """Create a temporary work directory with a train.py file."""
    train_script = tmp_path / "train.py"
    train_script.write_text('print("val_bpb: 1.234")\nprint("train_time: 45.2")\n')
    return tmp_path


@pytest.fixture
def mock_train_command(tmp_work_dir):
    """Command to run mock training."""
    return f"python {tmp_work_dir / 'train.py'}"


@pytest.fixture
def basic_config(tmp_work_dir, mock_train_command):
    """Basic config for testing."""
    return Config(
        name="test-session",
        population_size=5,
        max_evaluations=10,
        elitism=2,
        mutation_rate=0.2,
        crossover_rate=0.7,
        experiment_timeout=60,
        train_command=mock_train_command,
        work_dir=tmp_work_dir,
        target_file="train.py",
        metric_name="val_bpb",
        metric_direction="lower",
        rethink_interval=5,
    )


@pytest.fixture
def runner(sample_ideas, basic_config, mock_llm):
    """Create a basic GARunner for testing."""
    return GARunner(
        ideas=sample_ideas,
        config=basic_config,
        llm_call=mock_llm,
    )


@pytest.fixture
def runner_with_population(runner):
    """Runner with initialized population."""
    runner.initialize_population()
    return runner


def make_individual(genes: dict, fitness: float | None = None, generation: int = 0) -> Individual:
    """Helper to create test individuals."""
    import random
    return Individual(
        id=f"ind-{random.randint(0, 999999):06d}",
        genes=genes,
        fitness=fitness,
        generation=generation,
    )
