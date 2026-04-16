"""Tests for project-level constants and requirements file consistency."""

from pathlib import Path

from facecloak.project import PHASE_LABEL, requirements_lines


def test_requirements_file_matches_pinned_runtime_dependencies() -> None:
    requirements = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert requirements == requirements_lines()


def test_phase_label_is_not_a_numbered_phase() -> None:
    """PHASE_LABEL must not contain a phase number reference."""
    import re
    assert not re.search(r'\bPhase [0-9]\b', PHASE_LABEL), (
        f"PHASE_LABEL should not contain a numbered phase: {PHASE_LABEL!r}"
    )
