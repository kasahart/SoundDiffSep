from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def deftmamba_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; the upstream-faithful FlashAttention path is CUDA-only")
    pytest.importorskip("flash_attn", reason="flash-attn optional dependency is not installed")
    pytest.importorskip("mamba_ssm", reason="mamba-ssm optional dependency is not installed")
    return torch.device("cuda")
