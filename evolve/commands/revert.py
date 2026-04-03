"""
cevolve revert - Instructions for reverting file changes.

The CLI no longer does git operations directly. Agent handles git.
"""


def handle(args) -> dict:
    """Handle revert command."""
    
    return {
        "status": "info",
        "message": "Run 'git checkout .' to revert uncommitted changes",
        "commands": [
            "git checkout .",           # Revert tracked files
            "git clean -fd",            # Remove untracked files (optional)
        ],
        "note": "The CLI no longer runs git commands. Agent handles git operations.",
    }
