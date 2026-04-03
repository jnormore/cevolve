"""Tests for _parse_ideas_response in evolve/commands/run.py."""

import pytest
from evolve.commands.run import _parse_ideas_response


class TestParseIdeasResponse:
    """Test the idea parser for LLM responses."""

    def test_parse_simple_binary_idea(self):
        """Parse a simple binary idea (no variants)."""
        response = "use_cache: Enable caching for faster lookups"
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "use_cache"
        assert ideas[0].description == "Enable caching for faster lookups"
        assert ideas[0].variants == []

    def test_parse_idea_with_variants(self):
        """Parse idea with variants on next line."""
        response = """depth: Number of transformer layers
  variants: 4, 6, 8, 10"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "depth"
        assert ideas[0].variants == ["4", "6", "8", "10"]

    def test_parse_multiple_ideas(self):
        """Parse multiple ideas."""
        response = """insertion_threshold: Threshold for switching algorithms
  variants: 8, 16, 32

use_iterative: Use iterative instead of recursive

pivot_strategy: How to select pivot
  variants: first, middle, median_of_three"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 3
        assert ideas[0].name == "insertion_threshold"
        assert ideas[0].variants == ["8", "16", "32"]
        assert ideas[1].name == "use_iterative"
        assert ideas[1].variants == []
        assert ideas[2].name == "pivot_strategy"
        assert ideas[2].variants == ["first", "middle", "median_of_three"]

    def test_skip_markdown_headers(self):
        """Skip lines starting with markdown headers."""
        response = """## Optimization Ideas

depth: Model depth
  variants: 4, 6, 8

### Summary

**Notes:**"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "depth"

    def test_skip_markdown_bold_lines(self):
        """Skip lines starting with bold markdown."""
        response = """depth: Model depth
  variants: 4, 6

**Summary of tradeoffs:**

**Binary:** on/off

**Add:** new_idea_1, new_idea_2"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "depth"

    def test_skip_bullet_points(self):
        """Skip markdown bullet points."""
        response = """depth: Model depth

- `depth`: Main tuning knob
- `batch_size`: Memory tradeoff
* Another bullet point"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "depth"

    def test_skip_explanatory_text(self):
        """Skip lines containing explanatory keywords."""
        response = """depth: Model depth

Example output:
Note: This is important
Summary of results
Tradeoff analysis"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "depth"

    def test_clean_backticks_from_name(self):
        """Remove backticks from idea names."""
        response = "`use_cache`: Enable caching"
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "use_cache"

    def test_clean_leading_underscores(self):
        """Remove leading underscores from names."""
        response = "__private_idea: Some description"
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "private_idea"

    def test_reject_invalid_python_identifiers(self):
        """Reject names that aren't valid Python identifiers after cleaning."""
        response = """123_bad: Starts with number
good_idea: Valid name
-bad-name: Has dashes at start (cleaned to bad_name)
also_good: Another valid one"""
        
        ideas = _parse_ideas_response(response)
        
        # Names starting with numbers are rejected even after cleaning
        # Names with dashes are cleaned (- becomes _, leading _ stripped)
        names = [i.name for i in ideas]
        assert "good_idea" in names
        assert "also_good" in names
        assert "bad_name" in names  # -bad-name cleaned to bad_name
        assert len(ideas) == 3
        
        # 123_bad should be rejected (starts with number after cleaning)
        assert "123_bad" not in names
        assert "bad" not in names  # Not stripped to just "bad"

    def test_reject_blocklisted_names(self):
        """Reject names that are on the blocklist."""
        response = """binary: on/off
add: new things
variants: classic, modern
rationale: explanation
real_idea: This is a real idea"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "real_idea"

    def test_normalize_name_to_lowercase(self):
        """Names are normalized to lowercase."""
        response = "MyIdea: Description here"
        
        ideas = _parse_ideas_response(response)
        
        assert ideas[0].name == "myidea"

    def test_convert_spaces_and_dashes_to_underscores(self):
        """Spaces and dashes in names become underscores."""
        response = "my-idea name: Description"
        
        ideas = _parse_ideas_response(response)
        
        assert ideas[0].name == "my_idea_name"

    def test_collapse_multiple_underscores(self):
        """Multiple underscores are collapsed to one."""
        response = "my___idea: Description"
        
        ideas = _parse_ideas_response(response)
        
        assert ideas[0].name == "my_idea"

    def test_skip_empty_descriptions(self):
        """Skip ideas with empty descriptions after cleaning."""
        response = """good_idea: Real description
bad_idea: **
another_bad: ``"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "good_idea"

    def test_real_world_bad_response(self):
        """Test with a real-world badly formatted LLM response."""
        # This is similar to what caused the original bug
        response = """insertion_threshold: Threshold for switching from quicksort to insertion sort
  variants: 8, 12, 16, 24, 32, 48, 64

pivot_strategy: How to select the pivot element
  variants: first, middle, median_of_three, random

partition_scheme: Partitioning algorithm
  variants: lomuto, hoare

use_iterative: Use explicit stack instead of recursion

three_way_partition: Use Dutch National Flag 3-way partitioning

tail_recursion_opt: Recurse only on smaller partition

check_if_sorted: Check if subarray is already sorted

**Summary of tradeoffs:**

- `insertion_threshold`: Main tuning knob - balance recursion overhead
- `pivot_strategy`: median_of_three usually best
- `partition_scheme`: hoare typically faster
- `three_way_partition`: Big win for duplicate-heavy data
- `tail_recursion_opt`: Minor speed gain

**variants:** `classic`, `yaroslavskiy`

**rationale:**

**Binary:** on/off

**Add:** dual_pivot_quicksort, numpy_arrays"""
        
        ideas = _parse_ideas_response(response)
        
        # Should only get the 7 valid ideas
        assert len(ideas) == 7
        
        names = [i.name for i in ideas]
        assert "insertion_threshold" in names
        assert "pivot_strategy" in names
        assert "partition_scheme" in names
        assert "use_iterative" in names
        assert "three_way_partition" in names
        assert "tail_recursion_opt" in names
        assert "check_if_sorted" in names
        
        # These garbage "ideas" should NOT be present
        assert "summary_of_tradeoffs" not in names
        assert "binary" not in names
        assert "add" not in names
        assert "rationale" not in names
        assert "variants" not in names

    def test_skip_comments(self):
        """Skip lines starting with #."""
        response = """# This is a comment
depth: Model depth
# Another comment"""
        
        ideas = _parse_ideas_response(response)
        
        assert len(ideas) == 1
        assert ideas[0].name == "depth"
