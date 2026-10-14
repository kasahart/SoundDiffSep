from __future__ import annotations

import importlib.util
import subprocess
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


def _git(cwd: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True)


def test_already_patched_checkout_with_extra_edits_is_rejected(monkeypatch, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("deftmamba_reference_common", COMMON)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    (checkout / "model.py").write_text("a = 1\nb = 2\n")
    (checkout / "other.py").write_text("c = 3\n")
    _git(checkout, "add", ".")
    _git(checkout, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    (checkout / "model.py").write_text("a = 1\nb = 20\n")
    patch = tmp_path / "fix.patch"
    patch.write_bytes(subprocess.run(["git", "diff"], cwd=checkout, check=True, capture_output=True).stdout)
    _git(checkout, "checkout", "--", "model.py")
    monkeypatch.setattr(common, "MINIMAL_PATCH", patch)

    assert common.apply_minimal_patch(checkout) == "applied"
    assert common.apply_minimal_patch(checkout) == "already_applied"
    (checkout / "other.py").write_text("c = 30\n")
    with pytest.raises(RuntimeError, match="other.py"):
        common.apply_minimal_patch(checkout)
    (checkout / "other.py").write_text("c = 3\n")
    (checkout / "extra.py").write_text("")
    with pytest.raises(RuntimeError, match="extra.py"):
        common.apply_minimal_patch(checkout)
