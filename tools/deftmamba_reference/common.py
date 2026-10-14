"""Utilities for the external GPL-2.0 DeFT-Mamba reference checkout."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


UPSTREAM_URL = "https://github.com/donghoney0416/DeFTMamba.git"
UPSTREAM_COMMIT = "34e2c9a59fd4e3b0135b55dac9be3d972602bfd8"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKOUT = REPOSITORY_ROOT / ".cache" / "upstream" / "DeFTMamba"
# The GPL-2.0 runtime-fix patch is not distributed in this repository (it contains
# excerpts of the upstream GPL source). Place it here to use --apply-minfix.
MINIMAL_PATCH = (
    REPOSITORY_ROOT
    / "third_party_patches"
    / "deftmamba"
    / "minimal_runtime_fixes.patch"
)


def run_checked(arguments: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def current_commit(checkout: Path) -> str:
    return run_checked(("git", "rev-parse", "HEAD"), checkout).stdout.strip()


def prepare_checkout(checkout: Path = DEFAULT_CHECKOUT) -> Path:
    """Clone once and pin the cache checkout without discarding local changes."""

    checkout = checkout.resolve()
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run_checked(("git", "clone", UPSTREAM_URL, str(checkout)), checkout.parent)

    try:
        run_checked(("git", "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}"), checkout)
    except subprocess.CalledProcessError:
        run_checked(("git", "fetch", "origin", UPSTREAM_COMMIT), checkout)

    if current_commit(checkout) != UPSTREAM_COMMIT:
        status = run_checked(("git", "status", "--porcelain"), checkout).stdout
        if status:
            raise RuntimeError(
                f"refusing to switch dirty upstream checkout at {checkout}; "
                "use a clean cache path"
            )
        run_checked(("git", "checkout", "--detach", UPSTREAM_COMMIT), checkout)
    return checkout


def apply_minimal_patch(checkout: Path) -> str:
    """Apply the tracked GPL-target patch, or report that it is already applied."""

    if not MINIMAL_PATCH.is_file():
        raise FileNotFoundError(
            f"{MINIMAL_PATCH} is not distributed with this repository (it contains "
            "GPL-2.0 code). Recreate the six runtime fixes listed in "
            "docs/research/deftmamba_upstream_audit.md as that patch file, then rerun."
        )

    check = subprocess.run(
        ("git", "apply", "--check", str(MINIMAL_PATCH)),
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check.returncode == 0:
        run_checked(("git", "apply", str(MINIMAL_PATCH)), checkout)
        verify_only_minimal_patch(checkout)
        return "applied"

    reverse = subprocess.run(
        ("git", "apply", "--reverse", "--check", str(MINIMAL_PATCH)),
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if reverse.returncode == 0:
        verify_only_minimal_patch(checkout)
        return "already_applied"
    raise RuntimeError(
        "minimal patch does not apply cleanly\n"
        f"forward check:\n{check.stderr}\nreverse check:\n{reverse.stderr}"
    )


def _write_tree(checkout: Path, *steps: Sequence[str]) -> str:
    """Tree id after ``git read-tree HEAD`` plus ``steps`` on a throwaway index."""

    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        for arguments in (("git", "read-tree", "HEAD"), *steps):
            subprocess.run(list(arguments), cwd=checkout, env=env, check=True,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return subprocess.run(("git", "write-tree"), cwd=checkout, env=env, check=True,
                              text=True, stdout=subprocess.PIPE).stdout.strip()


def verify_only_minimal_patch(checkout: Path) -> None:
    """Fail unless the checkout is the pinned commit plus exactly the minimal patch."""

    expected = _write_tree(checkout, ("git", "apply", "--cached", str(MINIMAL_PATCH)))
    actual = _write_tree(checkout, ("git", "add", "-u"))
    untracked = run_checked(
        ("git", "ls-files", "--others", "--exclude-standard", "--", "*.py"), checkout
    ).stdout.split()
    if expected != actual or untracked:
        changed = run_checked(("git", "diff", "--name-only", expected, actual), checkout).stdout.split()
        raise RuntimeError(
            f"upstream checkout at {checkout} has changes beyond the minimal patch "
            f"(modified: {changed}, untracked .py: {untracked}); use a clean cache path"
        )


def write_json(payload: object, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
