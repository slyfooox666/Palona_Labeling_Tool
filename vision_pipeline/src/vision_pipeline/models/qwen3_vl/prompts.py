"""Prompt helpers for Qwen3-VL."""

from __future__ import annotations


def render_prompt(template: str, **variables: object) -> str:
    """Render a prompt template using simple Python format variables."""
    return template.format(**variables)
