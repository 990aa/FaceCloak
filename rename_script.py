import os
import shutil
from pathlib import Path

def rename_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content.replace('uacloak', 'uacloak')
    new_content = new_content.replace('UACloak', 'UACloak')
    new_content = new_content.replace('UACloak', 'UACloak')

    if content != new_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated content in {file_path}")

def walk_and_replace(directory):
    for root, dirs, files in os.walk(directory):
        # Exclude specific directories
        dirs[:] = [d for d in dirs if d not in ('.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', 'uv.lock')]
        for file in files:
            if file.endswith(('.py', '.md', '.toml', '.json', '.yaml', '.csv', '.txt', '.pyi', '.yaml')):
                file_path = Path(root) / file
                rename_content(file_path)

if __name__ == "__main__":
    CWD = Path.cwd()

    # 1. First, search and replace in files
    walk_and_replace(CWD)

    # 2. Rename directory
    old_dir = CWD / 'uacloak'
    new_dir = CWD / 'uacloak'
    if old_dir.exists() and old_dir.is_dir():
        shutil.move(str(old_dir), str(new_dir))
        print(f"Renamed directory {old_dir} to {new_dir}")
        walk_and_replace(new_dir)
