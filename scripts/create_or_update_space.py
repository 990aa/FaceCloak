from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uacloak.deploy import create_or_update_space, deployment_markdown


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else None
    result = create_or_update_space(repo_id=repo_id)
    print(deployment_markdown(result))


if __name__ == "__main__":
    main()
