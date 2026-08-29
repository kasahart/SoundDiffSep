from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

import sounddiffsep.models.deftmamba as deftmamba


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs" / "deftmamba"


class DummyCore(nn.Module):
    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed[..., :2]


def test_package_import_does_not_load_optional_cuda_dependencies() -> None:
    code = """
import sys
import sounddiffsep.models.deftmamba
assert 'mamba_ssm' not in sys.modules
assert 'flash_attn' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_named_factory_wires_model_and_audio_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_core_factory(**settings: object) -> DummyCore:
        captured.update(settings)
        return DummyCore()

    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", fake_core_factory
    )
    wrapper = deftmamba.create_deftmamba_waveform_model(
        deftmamba.DEFTMAMBA_OFFICIAL_2CH_1SRC,
        model_config={
            "name": "deftmamba_official_2ch_1src",
            "n_mics": 2,
            "n_srcs": 1,
            "n_layers": 6,
            "emb_dim": 64,
        },
        audio_config={
            "sample_rate": 16_000,
            "duration_seconds": 4,
            "n_fft": 128,
            "win_length": 128,
            "hop_length": 64,
            "window": "hamming",
            "ri_packing": "channel_major_interleaved",
            "center": False,
            "periodic": False,
            "normalized": True,
        },
    )

    assert captured == {
        "n_mics": 2,
        "n_srcs": 1,
        "n_layers": 6,
        "emb_dim": 64,
    }
    assert isinstance(wrapper, deftmamba.DeFTMambaWaveformWrapper)
    assert isinstance(wrapper.core_model, DummyCore)
    assert wrapper.n_mics == 2
    assert wrapper.n_srcs == 1
    assert wrapper.n_fft == 128
    assert wrapper.center is False
    assert wrapper.periodic is False
    assert wrapper.normalized is True


def test_factory_rejects_unknown_name_before_constructing_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**settings: object) -> DummyCore:
        raise AssertionError(f"core should not be constructed: {settings}")

    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", fail_if_called
    )

    with pytest.raises(ValueError, match="unknown experimental"):
        deftmamba.create_deftmamba_waveform_model("not_a_model")


def test_factory_rejects_unvalidated_ri_packing_before_constructing_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**settings: object) -> DummyCore:
        raise AssertionError(f"core should not be constructed: {settings}")

    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", fail_if_called
    )

    with pytest.raises(ValueError, match="channel_major_interleaved"):
        deftmamba.create_deftmamba_waveform_model(
            audio_config={"ri_packing": "real_channels_then_imag_channels"}
        )


def test_provided_yaml_configs_are_complete_and_differ_only_in_depth() -> None:
    with (CONFIG_DIR / "official_2ch_1src_16k.yaml").open() as stream:
        official = yaml.safe_load(stream)
    with (CONFIG_DIR / "paper_depth_2ch_1src_16k.yaml").open() as stream:
        paper_depth = yaml.safe_load(stream)

    expected_audio = {
        "sample_rate": 16_000,
        "duration_seconds": 4,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 256,
        "window": "hamming",
        "ri_packing": "channel_major_interleaved",
        "center": True,
        "periodic": True,
        "normalized": False,
    }
    assert official["audio"] == expected_audio
    assert paper_depth["audio"] == expected_audio
    assert official["model"]["name"] == deftmamba.DEFTMAMBA_OFFICIAL_2CH_1SRC
    assert official["model"]["n_mics"] == 2
    assert official["model"]["n_srcs"] == 1
    assert official["model"]["n_layers"] == 4
    assert paper_depth["model"]["n_layers"] == 6

    paper_depth["model"]["n_layers"] = 4
    assert paper_depth == official


def test_two_channel_system_batch_stack_contract() -> None:
    wrapper = deftmamba.DeFTMambaWaveformWrapper(
        DummyCore(),
        n_mics=2,
        n_srcs=1,
        n_fft=64,
        win_length=64,
        hop_length=32,
    )
    mixed = torch.randn(2, 511)
    reference = torch.randn(2, 511)
    model_input = torch.stack([mixed, reference], dim=1)

    output = wrapper(model_input)

    assert model_input.shape == (2, 2, 511)
    assert output.shape == (2, 1, 511)
