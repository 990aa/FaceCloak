"""Tests for ablation manifest and utility helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from facecloak.ablation import (
    _parse_float_list,
    _parse_int_list,
    load_ablation_manifest,
)
from facecloak.errors import FaceCloakError


def test_load_ablation_manifest_parses_rows(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(image)

    manifest = tmp_path / "ablation.csv"
    manifest.write_text(
        "image_id,modality,image_path\n"
        "a,face,sample.png\n"
        "b,general,sample.png\n",
        encoding="utf-8",
    )

    rows = load_ablation_manifest(manifest, require_fixed_set=False)

    assert len(rows) == 2
    assert rows[0].image_id == "a"
    assert rows[0].modality == "face"
    assert rows[1].modality == "general"


def test_load_ablation_manifest_enforces_fixed_set_by_default(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(image)

    manifest = tmp_path / "ablation.csv"
    manifest.write_text(
        "image_id,modality,image_path\n"
        "a,face,sample.png\n",
        encoding="utf-8",
    )

    with pytest.raises(FaceCloakError, match="exactly 40 rows"):
        load_ablation_manifest(manifest)


def test_parse_float_and_int_lists() -> None:
    assert _parse_float_list("0.01, 0.02,0.03") == [0.01, 0.02, 0.03]
    assert _parse_int_list("10, 25,100") == [10, 25, 100]
