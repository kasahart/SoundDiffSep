#!/usr/bin/env python3
"""Run compile/import checks against an unmodified archive of the fixed commit."""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from common import DEFAULT_CHECKOUT, UPSTREAM_COMMIT, prepare_checkout, write_json


def execute(command: list[str], cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "command": ["<runtime-python>", *command[1:]],
        "command_redaction": (
            "The interpreter path is normalized; arguments and results are unchanged."
        ),
        "returncode": completed.returncode,
        "stdout": completed.stdout.replace(str(cwd), "."),
        "stderr": completed.stderr.replace(str(cwd), "."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkout = prepare_checkout(args.checkout)
    archive = subprocess.run(
        ("git", "archive", "--format=tar", UPSTREAM_COMMIT),
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout

    with tempfile.TemporaryDirectory(prefix="deftmamba-raw-") as temporary:
        raw_tree = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(raw_tree, filter="data")
        commands = [
            [sys.executable, "-m", "py_compile", "DeFTMamba.py"],
            [sys.executable, "-c", "import DeFTMamba"],
            [sys.executable, "-m", "compileall", "."],
        ]
        results = [execute(command, raw_tree) for command in commands]

    write_json(
        {
            "upstream_commit": UPSTREAM_COMMIT,
            "source": "unmodified git archive",
            "results": results,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
