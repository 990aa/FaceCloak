from pathlib import Path

from facecloak.project import requirements_lines


def test_requirements_file_matches_pinned_runtime_dependencies() -> None:
    requirements = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert requirements == requirements_lines()
