#!/usr/bin/env python3
"""Prepare the pinned external DeFT-Mamba checkout used by reference tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_CHECKOUT,
    REPOSITORY_ROOT,
    UPSTREAM_COMMIT,
    apply_minimal_patch,
    current_commit,
    prepare_checkout,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--apply-minfix", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkout = prepare_checkout(args.checkout)
    patch_status = "not_requested"
    if args.apply_minfix:
        patch_status = apply_minimal_patch(checkout)
    try:
        checkout_label = str(checkout.relative_to(REPOSITORY_ROOT))
    except ValueError:
        checkout_label = "<external-checkout>"
    write_json(
        {
            "checkout": checkout_label,
            "expected_commit": UPSTREAM_COMMIT,
            "actual_commit": current_commit(checkout),
            "minimal_patch": patch_status,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
