"""
Revert strategies for undoing file changes after evaluation.

Strategies:
- git: Uses `git checkout .` to revert all tracked file changes
- stash: Uses `git stash push` / `git stash pop` (safer for uncommitted work)  
- cache: Stores file contents in memory, restores after eval (no git required)
"""

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from typing import Optional
import fnmatch


class Reverter(ABC):
    """Base class for revert strategies."""
    
    @abstractmethod
    def snapshot(self):
        """Take a snapshot before changes (if needed)."""
        pass
    
    @abstractmethod
    def revert(self):
        """Revert all changes since last snapshot."""
        pass


class GitReverter(Reverter):
    """
    Revert using `git checkout .`
    
    Fast and simple. Reverts all tracked file changes.
    Does NOT affect untracked files or staged changes.
    """
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
    
    def snapshot(self):
        # No snapshot needed - git tracks state
        pass
    
    def revert(self):
        """Revert all tracked file changes."""
        result = subprocess.run(
            ["git", "checkout", "."],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git checkout failed: {result.stderr}")


class StashReverter(Reverter):
    """
    Revert using git stash.
    
    Safer than checkout - preserves uncommitted work.
    Uses a unique stash message to identify our stash.
    """
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.stash_message = "cevolve-snapshot"
        self._has_stash = False
    
    def snapshot(self):
        """Stash current state."""
        # Check if there are changes to stash
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            self._has_stash = False
            return
        
        result = subprocess.run(
            ["git", "stash", "push", "-m", self.stash_message, "--include-untracked"],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
        )
        self._has_stash = result.returncode == 0
    
    def revert(self):
        """Pop the stash to restore original state."""
        if not self._has_stash:
            return
            
        # Find our stash
        result = subprocess.run(
            ["git", "stash", "list"],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
        )
        
        for line in result.stdout.split('\n'):
            if self.stash_message in line:
                stash_ref = line.split(':')[0]
                subprocess.run(
                    ["git", "stash", "pop", stash_ref],
                    cwd=self.work_dir,
                    capture_output=True,
                )
                self._has_stash = False
                return


class CacheReverter(Reverter):
    """
    Revert by caching file contents in memory.
    
    Works without git. Stores content of all in-scope files.
    """
    
    def __init__(
        self,
        work_dir: Path,
        scope: list[str] = None,
        exclude: list[str] = None,
    ):
        self.work_dir = Path(work_dir)
        self.scope = scope or ["**/*.py"]
        self.exclude = exclude or []
        self.cache: dict[Path, str] = {}
    
    def _matches_patterns(self, path: Path, patterns: list[str]) -> bool:
        """Check if path matches any glob pattern."""
        try:
            rel_path = str(path.relative_to(self.work_dir))
        except ValueError:
            rel_path = str(path)
        return any(fnmatch.fnmatch(rel_path, p) for p in patterns)
    
    def _get_in_scope_files(self) -> list[Path]:
        """Get all files matching scope patterns."""
        files = []
        for pattern in self.scope:
            for path in self.work_dir.glob(pattern):
                if path.is_file():
                    if not self._matches_patterns(path, self.exclude):
                        files.append(path)
        return files
    
    def snapshot(self):
        """Cache content of all in-scope files."""
        self.cache = {}
        for path in self._get_in_scope_files():
            try:
                self.cache[path] = path.read_text()
            except (UnicodeDecodeError, IOError):
                pass  # Skip binary or unreadable files
    
    def revert(self):
        """Restore all cached files."""
        for path, content in self.cache.items():
            try:
                path.write_text(content)
            except IOError as e:
                print(f"Warning: Could not restore {path}: {e}")


class SingleFileReverter(Reverter):
    """
    Revert a single file by storing its content.
    
    Used for backward compatibility with single-file mode.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.original_content: Optional[str] = None
    
    def snapshot(self):
        """Store the file's current content."""
        if self.file_path.exists():
            self.original_content = self.file_path.read_text()
        else:
            self.original_content = None
    
    def revert(self):
        """Restore the file's original content."""
        if self.original_content is not None:
            self.file_path.write_text(self.original_content)


class MultiFileReverter(Reverter):
    """
    Revert multiple files by storing their contents.
    
    Used for multi-file optimization mode.
    """
    
    def __init__(self, work_dir: Path, file_paths: list[str]):
        self.work_dir = Path(work_dir)
        self.file_paths = file_paths
        self.original_contents: dict[str, Optional[str]] = {}
    
    def snapshot(self):
        """Store all files' current content."""
        for file_path in self.file_paths:
            full_path = self.work_dir / file_path
            if full_path.exists():
                self.original_contents[file_path] = full_path.read_text()
            else:
                self.original_contents[file_path] = None
    
    def revert(self):
        """Restore all files' original content."""
        for file_path, content in self.original_contents.items():
            if content is not None:
                full_path = self.work_dir / file_path
                full_path.write_text(content)


def get_reverter(
    strategy: str,
    work_dir: Path,
    scope: list[str] = None,
    exclude: list[str] = None,
    target_file: str = None,
    target_files: list[str] = None,
) -> Reverter:
    """
    Factory for reverter instances.
    
    Args:
        strategy: "git", "stash", "cache", "single", or "multi"
        work_dir: Working directory
        scope: File patterns to include (for cache strategy)
        exclude: File patterns to exclude (for cache strategy)
        target_file: Single file path (for single strategy)
        target_files: Multiple file paths (for multi strategy)
    """
    work_dir = Path(work_dir)
    
    if strategy == "git":
        return GitReverter(work_dir)
    elif strategy == "stash":
        return StashReverter(work_dir)
    elif strategy == "cache":
        return CacheReverter(work_dir, scope or ["**/*"], exclude)
    elif strategy == "single":
        if not target_file:
            raise ValueError("target_file required for 'single' strategy")
        return SingleFileReverter(work_dir / target_file)
    elif strategy == "multi":
        if not target_files:
            raise ValueError("target_files required for 'multi' strategy")
        return MultiFileReverter(work_dir, target_files)
    else:
        raise ValueError(f"Unknown revert strategy: {strategy}")
