from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[3] / "tools" / "deftmamba_reference" / "common.py"


def test_missing_minimal_patch_is_reported_before_git(monkeypatch, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("deftmamba_reference_common", COMMON)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    monkeypatch.setattr(common, "MINIMAL_PATCH", tmp_path / "absent.patch")
    with pytest.raises(FileNotFoundError, match="not distributed"):
        common.apply_minimal_patch(tmp_path)
