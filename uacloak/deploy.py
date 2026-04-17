"""Programmatic Hugging Face Space creation and upload."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware

from uacloak.errors import UACloakError
from uacloak.project import (
    PROJECT_ROOT,
    PROJECT_SLUG,
    SPACE_UPLOAD_ALLOW_PATTERNS,
    SPACE_URL_TEMPLATE,
)

HF_TOKEN_ENV_VAR = "FACECLOAK_HF_TOKEN"


@dataclass(frozen=True, slots=True)
class SpaceDeploymentResult:
    repo_id: str
    space_url: str
    runtime_stage: str | None
    hardware: str | None
    commit_oid: str | None


def read_env_file(env_path: Path | None = None) -> dict[str, str]:
    env_path = env_path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip('"').strip("'")
        values[key] = value

    return values


def resolve_hf_token(env_path: Path | None = None) -> str:
    token = os.environ.get(HF_TOKEN_ENV_VAR)
    if token:
        return token

    token = read_env_file(env_path).get(HF_TOKEN_ENV_VAR)
    if token:
        return token

    raise UACloakError(
        f"{HF_TOKEN_ENV_VAR} was not found. Add it to the environment or to a local .env file."
    )


def default_space_repo_id(api: HfApi, token: str) -> str:
    whoami = api.whoami(token=token, cache=False)
    username = whoami.get("name")
    if not username:
        raise UACloakError(
            "Could not determine the Hugging Face username from the provided token."
        )
    return f"{username}/{PROJECT_SLUG}"


def create_or_update_space(
    *,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
    folder_path: Path | None = None,
) -> SpaceDeploymentResult:
    api = api or HfApi()
    token = token or resolve_hf_token()
    repo_id = repo_id or default_space_repo_id(api, token)
    folder_path = folder_path or PROJECT_ROOT

    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="space",
        exist_ok=True,
        space_sdk="gradio",
        space_hardware=SpaceHardware.CPU_BASIC,
    )
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=folder_path,
        token=token,
        allow_patterns=list(SPACE_UPLOAD_ALLOW_PATTERNS),
        commit_message="Deploy UACloak",
    )
    runtime = api.get_space_runtime(repo_id, token=token)

    return SpaceDeploymentResult(
        repo_id=repo_id,
        space_url=SPACE_URL_TEMPLATE.format(repo_id=repo_id),
        runtime_stage=getattr(runtime, "stage", None),
        hardware=getattr(runtime, "hardware", None),
        commit_oid=getattr(commit_info, "oid", None),
    )


def deployment_markdown(result: SpaceDeploymentResult) -> str:
    return "\n".join(
        [
            "### Hugging Face Space",
            f"- Repository: `{result.repo_id}`",
            f"- URL: {result.space_url}",
            f"- Runtime Stage: `{result.runtime_stage or 'unknown'}`",
            f"- Hardware: `{result.hardware or 'unknown'}`",
            f"- Commit OID: `{result.commit_oid or 'unknown'}`",
        ]
    )
