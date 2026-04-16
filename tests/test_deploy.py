from pathlib import Path
import shutil
import uuid

from facecloak.deploy import (
    HF_TOKEN_ENV_VAR,
    create_or_update_space,
    default_space_repo_id,
    read_env_file,
)


class DummyRuntime:
    stage = "BUILDING"
    hardware = "cpu-basic"


class DummyCommit:
    oid = "abc123"


class DummyApi:
    def __init__(self) -> None:
        self.created = None
        self.uploaded = None

    def whoami(self, token, cache=False):
        return {"name": "example-user"}

    def create_repo(self, **kwargs):
        self.created = kwargs

    def upload_folder(self, **kwargs):
        self.uploaded = kwargs
        return DummyCommit()

    def get_space_runtime(self, repo_id, token=None):
        return DummyRuntime()


def _local_temp_dir(name: str) -> Path:
    path = Path("tests") / "_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_read_env_file_parses_simple_key_values() -> None:
    temp_dir = _local_temp_dir("env")
    env_path = temp_dir / ".env"
    env_path.write_text(
        f"{HF_TOKEN_ENV_VAR}=hf_test_token\nOTHER=value\n", encoding="utf-8"
    )

    values = read_env_file(env_path)

    assert values[HF_TOKEN_ENV_VAR] == "hf_test_token"
    assert values["OTHER"] == "value"
    shutil.rmtree(temp_dir)


def test_default_space_repo_id_uses_hf_username() -> None:
    repo_id = default_space_repo_id(DummyApi(), "hf_test_token")

    assert repo_id == "example-user/facecloak"


def test_create_or_update_space_uses_space_repo_settings() -> None:
    api = DummyApi()
    temp_dir = _local_temp_dir("deploy")
    folder_path = temp_dir / "repo"
    folder_path.mkdir()
    (folder_path / "app.py").write_text("print('ok')\n", encoding="utf-8")

    result = create_or_update_space(
        api=api,
        token="hf_test_token",
        repo_id="example-user/facecloak",
        folder_path=folder_path,
    )

    assert api.created["repo_type"] == "space"
    assert api.created["space_sdk"] == "gradio"
    assert api.uploaded["repo_type"] == "space"
    assert result.repo_id == "example-user/facecloak"
    assert result.runtime_stage == "BUILDING"
    shutil.rmtree(temp_dir)
