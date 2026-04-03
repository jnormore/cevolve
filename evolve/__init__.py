"""
Simple GA-driven code optimization.

The genome is just a list of optimization ideas.
Each individual is a combination of which ideas to apply.
The LLM generates code for each individual.
"""

from .core import Idea, Individual, Config, SecondaryMetric, GARunner
from . import persistence

__all__ = [
    "Idea",
    "Individual", 
    "Config",
    "SecondaryMetric",
    "GARunner",
    "persistence",
]
