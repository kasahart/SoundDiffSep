from __future__ import annotations

import pytest
import torch
from torch import nn

from sounddiffsep.models.deftmamba.blocks import DeFTMambaBlock


class _AxisPath(nn.Module):
    """Stand-in for DirectionalHybridMamba that keeps the padded shape."""

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[torch.Size] = []

    def forward(self, features, trace=None, trace_prefix=""):
        self.shapes.append(features.shape)
        return features + 1.0


def _block(window_size: int, hop_size: int) -> DeFTMambaBlock:
    block = DeFTMambaBlock.__new__(DeFTMambaBlock)
    nn.Module.__init__(block)
    block.frequency_path, block.time_path = _AxisPath(), _AxisPath()
    block.window_size, block.hop_size = window_size, hop_size
    return block


@pytest.mark.parametrize("hop_size", [1, 2, 3])
@pytest.mark.parametrize("frames", [1, 3, 11])
def test_block_returns_input_grid_when_hop_pads(hop_size: int, frames: int) -> None:
    block = _block(window_size=4, hop_size=hop_size)
    features = torch.randn(2, 8, frames, 257)
    output = block(features)
    assert output.shape == features.shape
    torch.testing.assert_close(output, (features + 1.0) + 1.0, rtol=0, atol=0)
    padded = block.frequency_path.shapes[0]
    assert padded[2] >= 4 and padded[3] >= 257
    if hop_size > 1 or frames < 4:
        assert padded != features.shape
