"""
Simple tool-using agent for optimization experiments.

Uses pi or claude CLI for LLM calls.
"""

import subprocess
import re
from pathlib import Path
from tools import execute_tool, get_tool_descriptions

SCRIPT_DIR = Path(__file__).parent

# Few-shot examples for the agent
FEW_SHOT_EXAMPLES = [
    {
        "task": "What is 25% of 80?",
        "turns": [
            ("TOOL: calculator(80 * 0.25)", "20"),
            ("ANSWER: 20", None)
        ]
    },
    {
        "task": "What is the capital of Japan?",
        "turns": [
            ("TOOL: lookup(capital of Japan)", "Tokyo"),
            ("ANSWER: Tokyo", None)
        ]
    },
    {
        "task": "A book costs $24. If it's 15% off, what do I pay?",
        "turns": [
            ("TOOL: calculator(24 * 0.15)", "3.6"),
            ("TOOL: calculator(24 - 3.6)", "20.4"),
            ("ANSWER: 20.4", None)
        ]
    },
    {
        "task": "How many minutes are in an hour?",
        "turns": [
            ("TOOL: lookup(minutes in an hour)", "60"),
            ("ANSWER: 60", None)
        ]
    },
    {
        "task": "What is 144 divided by 12, then plus 7?",
        "turns": [
            ("TOOL: calculator(144 / 12)", "12"),
            ("TOOL: calculator(12 + 7)", "19"),
            ("ANSWER: 19", None)
        ]
    }
]


def load_system_prompt() -> str:
    """Load system prompt from prompts/system.md."""
    prompt_file = SCRIPT_DIR / "prompts" / "system.md"
    content = prompt_file.read_text()
    # Substitute tool descriptions
    content = content.replace("{tools}", get_tool_descriptions())
    return content


def format_examples(num_examples: int) -> str:
    """Format few-shot examples for the prompt."""
    if num_examples == 0:
        return ""
    
    examples = FEW_SHOT_EXAMPLES[:num_examples]
    formatted = "\n## Examples\n\n"
    
    for ex in examples:
        formatted += f"Task: {ex['task']}\n"
        for response, tool_result in ex['turns']:
            formatted += f"Response: {response}\n"
            if tool_result:
                formatted += f"Tool result: {tool_result}\n"
        formatted += "\n"
    
    return formatted


def llm_call(prompt: str, model: str = "smollm2:135m") -> str:
    """Call LLM via ollama CLI."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: LLM call timed out"
    except FileNotFoundError:
        return f"Error: CLI '{cli}' not found"
    except Exception as e:
        return f"Error: {str(e)}"


def parse_response(response: str) -> tuple[str, str | None]:
    """
    Parse agent response to extract action.
    
    Returns:
        (action_type, action_value) where action_type is 'tool', 'answer', or 'unknown'
    """
    # Look for TOOL: pattern
    tool_match = re.search(r'TOOL:\s*(\w+)\(([^)]*)\)', response, re.IGNORECASE)
    if tool_match:
        tool_name = tool_match.group(1).lower()
        tool_arg = tool_match.group(2).strip()
        return ('tool', (tool_name, tool_arg))
    
    # Look for ANSWER: pattern
    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        return ('answer', answer)
    
    return ('unknown', None)


def run_agent(task: str, config: dict) -> tuple[str | None, int]:
    """
    Run the agent on a task.
    
    Args:
        task: The task to solve
        config: Agent configuration
        
    Returns:
        (answer, num_iterations) or (None, num_iterations) if failed
    """
    model = config.get("model", "qwen2.5:0.5b")
    system_prompt = load_system_prompt()
    examples = format_examples(config.get("num_examples", 0))
    max_iterations = config.get("max_iterations", 5)
    include_history = config.get("include_history", True)
    
    # Build initial prompt
    history = []
    
    for iteration in range(max_iterations):
        # Construct prompt
        prompt_parts = [system_prompt]
        
        if examples:
            prompt_parts.append(examples)
        
        prompt_parts.append(f"\n## Current Task\n\nTask: {task}")
        
        if history and include_history:
            prompt_parts.append("\n## History\n")
            for h in history:
                prompt_parts.append(h)
        
        prompt_parts.append("\n## Your Response\n")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Get LLM response
        response = llm_call(full_prompt, model)
        
        if response.startswith("Error:"):
            return None, iteration + 1
        
        # Parse response
        action_type, action_value = parse_response(response)
        
        if action_type == 'answer':
            return action_value, iteration + 1
        
        elif action_type == 'tool':
            tool_name, tool_arg = action_value
            tool_result = execute_tool(tool_name, tool_arg)
            
            # Add to history
            history.append(f"You: TOOL: {tool_name}({tool_arg})")
            history.append(f"Tool result: {tool_result}")
        
        else:
            # Unknown response format, add to history and continue
            history.append(f"You: {response[:200]}")
            history.append("(Could not parse response, please use TOOL: or ANSWER: format)")
    
    # Max iterations reached
    return None, max_iterations


if __name__ == "__main__":
    # Quick test
    import yaml
    
    config = yaml.safe_load(open(SCRIPT_DIR / "config.yaml"))
    
    test_tasks = [
        "What is 15% of 240?",
        "What is the capital of France?",
    ]
    
    for task in test_tasks:
        print(f"\nTask: {task}")
        answer, iterations = run_agent(task, config)
        print(f"Answer: {answer} (in {iterations} iterations)")
