from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from propagator.cli.main import PropagatorCLILegacy

EXAMPLE = Path(__file__).resolve().parents[2] / "example"


def parse(tmp_path, *extra: str) -> PropagatorCLILegacy:
    return PropagatorCLILegacy(
        _cli_parse_args=[
            "--config",
            str(EXAMPLE / "config.json"),
            "--dem",
            str(EXAMPLE / "dem.tif"),
            "--fuel",
            str(EXAMPLE / "fuel.tif"),
            "--output",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_fuzzy_freeze_is_rust_only(tmp_path):
    with pytest.raises(ValidationError, match="only supported by the Rust"):
        parse(
            tmp_path,
            "--freeze_dir",
            str(tmp_path / "freeze"),
            "--fuzzy_freeze_threshold",
            "0.9",
        )


def test_fuzzy_freeze_requires_freeze_dir(tmp_path):
    with pytest.raises(ValidationError, match="requires --freeze_dir"):
        parse(tmp_path, "--core", "rust", "--fuzzy_freeze_threshold", "0.9")


def test_fuzzy_freeze_flags_parse(tmp_path):
    cli = parse(
        tmp_path,
        "--core",
        "rust",
        "--freeze_dir",
        str(tmp_path / "freeze"),
        "--fuzzy_freeze_threshold",
        "0.7",
        "--fuzzy_freeze_margin",
        "2",
    )
    assert cli.fuzzy_freeze_threshold == 0.7
    assert cli.fuzzy_freeze_margin == 2
    assert parse(tmp_path).fuzzy_freeze_threshold is None
