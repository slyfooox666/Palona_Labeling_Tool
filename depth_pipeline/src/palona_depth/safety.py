"""Filesystem guards for private generated research artifacts."""

from __future__ import annotations

from pathlib import Path


def git_worktree_root(path: Path) -> Path | None:
    """Return the nearest containing Git worktree, including uncreated paths."""

    resolved = path.expanduser().resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def require_private_json_suffix(path: Path, *, suffix: str, artifact_name: str) -> None:
    """Require an ignored, recognizable suffix for JSON generated inside Git."""

    resolved = path.expanduser().resolve()
    worktree = git_worktree_root(resolved)
    if worktree is not None and not resolved.name.endswith(suffix):
        raise ValueError(
            f"{artifact_name} inside Git worktree {worktree} must end with {suffix!r} "
            "so private generated data remains covered by .gitignore"
        )


def assert_artifacts_outside_git(path: Path) -> None:
    """Prevent retained CCTV frames/depth maps from entering a Git worktree."""

    if git_worktree_root(path) is not None:
        raise ValueError(
            "--keep-depth-artifacts must point outside every Git worktree to prevent private frame commits"
        )
