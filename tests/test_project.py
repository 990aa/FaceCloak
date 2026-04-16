from pathlib import Path

from facecloak.project import PHASE_LABEL, requirements_lines


def test_requirements_file_matches_pinned_runtime_dependencies() -> None:
    requirements = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert requirements == requirements_lines()


def test_phase_label_tracks_current_delivery() -> None:
    assert PHASE_LABEL == "Phases 2-3"
