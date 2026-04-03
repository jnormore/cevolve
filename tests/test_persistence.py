"""Tests for session persistence."""

import pytest
import json
from pathlib import Path
from evolve.core import Idea, Individual, Config, GARunner
from evolve.persistence import (
    get_session_dir,
    save_config,
    save_ideas,
    save_population,
    save_state,
    append_history,
    save_summary,
    load_state,
    list_sessions,
    delete_session,
)


class TestSessionDir:
    """Test session directory structure."""
    
    def test_get_session_dir(self, tmp_path):
        """Session dir is under .cevolve/."""
        session_dir = get_session_dir(tmp_path, "my-session")
        
        assert session_dir == tmp_path / ".cevolve" / "my-session"


class TestSaveConfig:
    """Test config saving."""
    
    def test_save_config_creates_file(self, runner):
        """save_config creates config.json."""
        save_config(runner)
        
        config_path = get_session_dir(runner.config.work_dir, runner.config.name) / "config.json"
        assert config_path.exists()
    
    def test_save_config_content(self, runner):
        """config.json has expected fields."""
        save_config(runner)
        
        config_path = get_session_dir(runner.config.work_dir, runner.config.name) / "config.json"
        data = json.loads(config_path.read_text())
        
        assert data["name"] == runner.config.name
        assert data["population_size"] == runner.config.population_size
        assert data["metric_name"] == runner.config.metric_name
        assert data["metric_direction"] == runner.config.metric_direction


class TestSaveIdeas:
    """Test ideas saving."""
    
    def test_save_ideas_creates_file(self, runner):
        """save_ideas creates ideas.json."""
        save_ideas(runner)
        
        ideas_path = get_session_dir(runner.config.work_dir, runner.config.name) / "ideas.json"
        assert ideas_path.exists()
    
    def test_save_ideas_content(self, runner):
        """ideas.json has all ideas."""
        save_ideas(runner)
        
        ideas_path = get_session_dir(runner.config.work_dir, runner.config.name) / "ideas.json"
        data = json.loads(ideas_path.read_text())
        
        assert "flash_attn" in data
        assert "depth" in data
        assert data["flash_attn"]["variants"] == []
        assert data["depth"]["variants"] == ["4", "6", "8"]


class TestSavePopulation:
    """Test population saving."""
    
    def test_save_population_creates_file(self, runner_with_population):
        """save_population creates population.json."""
        save_population(runner_with_population)
        
        pop_path = get_session_dir(
            runner_with_population.config.work_dir, 
            runner_with_population.config.name
        ) / "population.json"
        assert pop_path.exists()
    
    def test_save_population_content(self, runner_with_population):
        """population.json has expected fields."""
        runner = runner_with_population
        runner.generation = 3
        runner.evaluations = 15
        runner.era = 2
        
        save_population(runner)
        
        pop_path = get_session_dir(runner.config.work_dir, runner.config.name) / "population.json"
        data = json.loads(pop_path.read_text())
        
        assert data["generation"] == 3
        assert data["evaluations"] == 15
        assert data["era"] == 2
        assert len(data["population"]) == len(runner.population)
    
    def test_save_population_with_best(self, runner_with_population):
        """Best individual is saved."""
        runner = runner_with_population
        runner.best = runner.population[0]
        runner.best.fitness = 1.5
        
        save_population(runner)
        
        pop_path = get_session_dir(runner.config.work_dir, runner.config.name) / "population.json"
        data = json.loads(pop_path.read_text())
        
        assert data["best"] is not None
        assert data["best"]["fitness"] == 1.5
    
    def test_save_population_infinity_handled(self, runner_with_population):
        """Infinity fitness is serialized correctly."""
        runner = runner_with_population
        runner.population[0].fitness = float('inf')
        
        save_population(runner)
        
        pop_path = get_session_dir(runner.config.work_dir, runner.config.name) / "population.json"
        data = json.loads(pop_path.read_text())
        
        assert data["population"][0]["fitness"] == "inf"


class TestAppendHistory:
    """Test history appending."""
    
    def test_append_history_creates_file(self, runner):
        """append_history creates history.jsonl."""
        entry = {"evaluation": 1, "fitness": 1.5}
        append_history(runner, entry)
        
        history_path = get_session_dir(runner.config.work_dir, runner.config.name) / "history.jsonl"
        assert history_path.exists()
    
    def test_append_history_content(self, runner):
        """Entries are appended correctly."""
        append_history(runner, {"evaluation": 1, "fitness": 1.5})
        append_history(runner, {"evaluation": 2, "fitness": 1.3})
        
        history_path = get_session_dir(runner.config.work_dir, runner.config.name) / "history.jsonl"
        lines = history_path.read_text().strip().split("\n")
        
        assert len(lines) == 2
        assert json.loads(lines[0])["evaluation"] == 1
        assert json.loads(lines[1])["evaluation"] == 2


class TestLoadState:
    """Test state loading."""
    
    def test_load_state_returns_none_if_missing(self, tmp_path):
        """Returns None if session doesn't exist."""
        result = load_state(tmp_path, "nonexistent")
        assert result is None
    
    def test_load_state_roundtrip(self, runner_with_population):
        """Save and load produces same state."""
        runner = runner_with_population
        runner.generation = 5
        runner.evaluations = 20
        runner.era = 2
        runner.best = runner.population[0]
        runner.best.fitness = 1.234
        
        save_state(runner)
        
        loaded = load_state(runner.config.work_dir, runner.config.name)
        
        assert loaded is not None
        assert loaded["population"]["generation"] == 5
        assert loaded["population"]["evaluations"] == 20
        assert loaded["population"]["era"] == 2
        assert loaded["population"]["best"]["fitness"] == 1.234
    
    def test_load_state_restores_infinity(self, runner_with_population):
        """Infinity fitness is restored."""
        runner = runner_with_population
        runner.population[0].fitness = float('inf')
        
        save_state(runner)
        loaded = load_state(runner.config.work_dir, runner.config.name)
        
        assert loaded["population"]["population"][0]["fitness"] == float('inf')
    
    def test_load_state_includes_history(self, runner):
        """History is loaded."""
        save_state(runner)
        append_history(runner, {"evaluation": 1, "fitness": 1.5})
        append_history(runner, {"evaluation": 2, "fitness": 1.3})
        
        loaded = load_state(runner.config.work_dir, runner.config.name)
        
        assert len(loaded["history"]) == 2


class TestListSessions:
    """Test session listing."""
    
    def test_list_sessions_empty(self, tmp_path):
        """Returns empty list if no sessions."""
        sessions = list_sessions(tmp_path)
        assert sessions == []
    
    def test_list_sessions_finds_sessions(self, runner_with_population):
        """Finds saved sessions."""
        save_state(runner_with_population)
        
        sessions = list_sessions(runner_with_population.config.work_dir)
        assert runner_with_population.config.name in sessions


class TestDeleteSession:
    """Test session deletion."""
    
    def test_delete_session(self, runner_with_population):
        """Deletes session directory."""
        save_state(runner_with_population)
        
        session_dir = get_session_dir(
            runner_with_population.config.work_dir,
            runner_with_population.config.name
        )
        assert session_dir.exists()
        
        result = delete_session(
            runner_with_population.config.work_dir,
            runner_with_population.config.name
        )
        
        assert result is True
        assert not session_dir.exists()
    
    def test_delete_nonexistent_session(self, tmp_path):
        """Returns False for nonexistent session."""
        result = delete_session(tmp_path, "nonexistent")
        assert result is False


class TestSaveSummary:
    """Test summary saving."""
    
    def test_save_summary_creates_file(self, runner_with_population):
        """save_summary creates RESULTS.md."""
        path = save_summary(runner_with_population)
        
        assert path.exists()
        assert path.name == "RESULTS.md"
    
    def test_save_summary_content(self, runner_with_population):
        """RESULTS.md has summary content."""
        path = save_summary(runner_with_population)
        
        content = path.read_text()
        assert "# cEvolve Results" in content
