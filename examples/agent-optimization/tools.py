"""
Simple tools for the agent.
"""

import re
import math

# Knowledge base for lookup tool
KNOWLEDGE_BASE = {
    # Geography
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of brazil": "Brasilia",
    "capital of australia": "Canberra",
    "capital of canada": "Ottawa",
    "capital of germany": "Berlin",
    "capital of italy": "Rome",
    "capital of spain": "Madrid",
    "largest country": "Russia",
    "smallest country": "Vatican City",
    
    # Science
    "speed of light": "299,792,458 meters per second",
    "boiling point of water": "100 degrees Celsius",
    "freezing point of water": "0 degrees Celsius",
    "planets in solar system": "8",
    "chemical symbol for gold": "Au",
    "chemical symbol for water": "H2O",
    
    # General
    "days in a year": "365",
    "hours in a day": "24",
    "minutes in an hour": "60",
    "seconds in a minute": "60",
}


def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Supports: +, -, *, /, **, %, (), sqrt, abs, round
    """
    try:
        # Clean the expression
        expr = expression.strip()
        
        # Only allow safe characters and functions
        allowed = set("0123456789+-*/.%() ")
        safe_funcs = {"sqrt": math.sqrt, "abs": abs, "round": round, "pow": pow}
        
        # Replace common patterns
        expr = expr.replace("^", "**")
        expr = expr.replace("×", "*")
        expr = expr.replace("÷", "/")
        
        # Check for sqrt, abs, round
        for func_name, func in safe_funcs.items():
            if func_name in expr.lower():
                # Extract the argument
                pattern = rf"{func_name}\s*\(\s*([^)]+)\s*\)"
                match = re.search(pattern, expr, re.IGNORECASE)
                if match:
                    inner = match.group(1)
                    inner_result = calculator(inner)
                    try:
                        result = func(float(inner_result))
                        expr = re.sub(pattern, str(result), expr, flags=re.IGNORECASE)
                    except:
                        pass
        
        # Validate remaining characters
        for char in expr:
            if char not in allowed:
                return f"Error: Invalid character '{char}'"
        
        # Evaluate
        result = eval(expr)
        
        # Format result
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"


def lookup(query: str) -> str:
    """
    Look up information in the knowledge base.
    """
    query_lower = query.lower().strip()
    
    # Direct match
    if query_lower in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[query_lower]
    
    # Partial match
    for key, value in KNOWLEDGE_BASE.items():
        if query_lower in key or key in query_lower:
            return value
    
    return "Not found in knowledge base"


# Tool registry
TOOLS = {
    "calculator": calculator,
    "lookup": lookup,
}


def execute_tool(tool_name: str, argument: str) -> str:
    """Execute a tool by name with the given argument."""
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
        return TOOLS[tool_name](argument)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def get_tool_descriptions() -> str:
    """Get descriptions of available tools for the prompt."""
    return """Available tools:
- calculator(expression): Evaluate a mathematical expression. Example: calculator(15 * 0.20)
- lookup(query): Look up factual information. Example: lookup(capital of France)"""
