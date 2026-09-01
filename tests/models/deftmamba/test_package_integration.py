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


class ProjectionCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_projection = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed.new_zeros((*packed.shape[:3], 2))


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


def test_complex_mask_factory_initializes_only_output_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_core = ProjectionCore()
    direct_core.output_projection.weight.data.fill_(0.25)
    direct_core.output_projection.bias.data.fill_(0.5)
    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", lambda **_: direct_core
    )

    direct = deftmamba.create_deftmamba_waveform_model(
        deftmamba.DEFTMAMBA_OFFICIAL_2CH_1SRC
    )

    assert isinstance(direct, deftmamba.DeFTMambaWaveformWrapper)
    torch.testing.assert_close(
        direct_core.output_projection.weight,
        torch.full_like(direct_core.output_projection.weight, 0.25),
    )
    torch.testing.assert_close(
        direct_core.output_projection.bias,
        torch.full_like(direct_core.output_projection.bias, 0.5),
    )

    mask_core = ProjectionCore()
    mask_core.output_projection.weight.data.fill_(0.25)
    mask_core.output_projection.bias.data.fill_(0.5)
    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", lambda **_: mask_core
    )
    masked = deftmamba.create_deftmamba_waveform_model(
        deftmamba.DEFTMAMBA_COMPLEX_MASK_2CH_1SRC,
        mask_config={
            "parameterization": "bounded_cartesian",
            "reference_channel": 0,
            "component_bound": 2.0,
            "head_init_std": 1.0e-3,
            "identity_initialization": True,
        },
        model_contract={
            "output_mode": "complex_mask",
            "mask_parameterization": "bounded_cartesian",
            "reference_channel": 0,
            "component_bound": 2.0,
            "contract_version": 1,
        },
    )

    assert isinstance(masked, deftmamba.DeFTMambaComplexMaskWaveformWrapper)
    assert mask_core.output_projection.weight.std().item() == pytest.approx(
        1.0e-3, rel=0.1
    )
    torch.testing.assert_close(
        mask_core.output_projection.bias,
        torch.zeros_like(mask_core.output_projection.bias),
    )
    assert masked.model_contract["output_mode"] == "complex_mask"


def test_complex_mask_factory_rejects_contract_mismatch_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**settings: object) -> DummyCore:
        raise AssertionError(f"core should not be constructed: {settings}")

    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", fail_if_called
    )

    with pytest.raises(ValueError, match="model_contract does not match"):
        deftmamba.create_deftmamba_waveform_model(
            deftmamba.DEFTMAMBA_COMPLEX_MASK_2CH_1SRC,
            mask_config={"component_bound": 2.0},
            model_contract={
                "output_mode": "direct_ri",
                "mask_parameterization": "bounded_cartesian",
                "reference_channel": 0,
                "component_bound": 2.0,
                "contract_version": 1,
            },
        )


def test_complex_mask_factory_rejects_reference_channel_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**settings: object) -> DummyCore:
        raise AssertionError(f"core should not be constructed: {settings}")

    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", fail_if_called
    )

    with pytest.raises(ValueError, match="reference_channel must be in"):
        deftmamba.create_deftmamba_waveform_model(
            deftmamba.DEFTMAMBA_COMPLEX_MASK_2CH_1SRC,
            model_config={"n_mics": 2, "n_srcs": 1},
            mask_config={"reference_channel": 2},
        )


@pytest.mark.parametrize("head_init_std", [0.0, -1.0e-3, float("nan")])
def test_complex_mask_factory_rejects_invalid_head_std_before_core(
    monkeypatch: pytest.MonkeyPatch, head_init_std: float
) -> None:
    def fail_if_called(**settings: object) -> DummyCore:
        raise AssertionError(f"core should not be constructed: {settings}")

    monkeypatch.setattr(
        deftmamba, "create_official_2ch_1src_core", fail_if_called
    )

    with pytest.raises(ValueError, match="head_init_std must be finite"):
        deftmamba.create_deftmamba_waveform_model(
            deftmamba.DEFTMAMBA_COMPLEX_MASK_2CH_1SRC,
            mask_config={"head_init_std": head_init_std},
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


def test_complex_mask_yaml_declares_parameterization_and_contract() -> None:
    with (CONFIG_DIR / "complex_mask_2ch_1src_16k.yaml").open() as stream:
        config = yaml.safe_load(stream)

    assert config["model"]["name"] == deftmamba.DEFTMAMBA_COMPLEX_MASK_2CH_1SRC
    assert config["mask"] == {
        "parameterization": "bounded_cartesian",
        "reference_channel": 0,
        "component_bound": 2.0,
        "head_init_std": 0.001,
        "identity_initialization": True,
    }
    assert config["model_contract"] == {
        "output_mode": "complex_mask",
        "mask_parameterization": "bounded_cartesian",
        "reference_channel": 0,
        "component_bound": 2.0,
        "contract_version": 1,
    }


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
