from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.deftmamba_reference
def test_upstream_minfix_intermediates_match_independent_implementation(
    deftmamba_cuda_device: torch.device, tmp_path: Path
) -> None:
    del deftmamba_cuda_device
    output_path = tmp_path / "reference-equivalence.json"
    command = [
        sys.executable,
        "tools/deftmamba_reference/compare_with_sounddiffsep.py",
        "--n-mics",
        "2",
        "--n-srcs",
        "1",
        "--frequency-bins",
        "8",
        "--time-frames",
        "8",
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["upstream_commit"] == "34e2c9a59fd4e3b0135b55dac9be3d972602bfd8"
    assert result["configuration"]["n_mics"] == 2
    assert result["configuration"]["n_srcs"] == 1
    assert result["parameter_count"]["upstream_minfix"] == result["parameter_count"][
        "sounddiffsep"
    ]
    assert result["state_dict_entries"] == 84
    assert result["all_comparisons_within_tolerance"] is True
    assert result["first_mismatch"] is None
    assert result["final_output_metrics"]["maximum_absolute_error"] == 0.0
