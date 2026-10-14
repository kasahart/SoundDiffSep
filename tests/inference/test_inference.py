from __future__ import annotations

import numpy as np
import pytest
import torch

from sounddiffsep.inference import Separator, check_cfg
from sounddiffsep.normalization import NORM_TYPES, peak_gains


class HalfFirstChannel(torch.nn.Module):
    """Linear stand-in separator: output = 0.5 * first input channel, [B, 1, T]."""

    def forward(self, x):
        return 0.5 * x[:, :1]


def make(norm_type, order=("close", "far"), peak=0.5):
    cfg = {"model": "resunet30", "input_order": list(order), "norm": {"type": norm_type, "peak": peak}}
    return Separator(HalfFirstChannel(), cfg, "cpu")


def pair(seed=0, t=4000, b=None):
    rng = np.random.default_rng(seed)
    shape = (t,) if b is None else (b, t)
    return (0.3 * rng.standard_normal(shape)).astype(np.float32), (0.1 * rng.standard_normal(shape)).astype(np.float32)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_original_level_and_identity(norm_type):
    close, far = pair()
    sep = make(norm_type)
    s, n = sep.separate(close, far)
    np.testing.assert_allclose(s, 0.5 * close, rtol=1e-6, atol=1e-7)   # gain undone
    np.testing.assert_allclose(s + n, close, rtol=0, atol=1e-7)        # n_hat = close - s_hat
    for k in (1e-3, 7.0):                                               # level-invariant input scaling
        sk, _ = sep.separate(k * close, k * far)
        np.testing.assert_allclose(sk / k, s, rtol=1e-5, atol=1e-7)


def test_batch_and_torch_inputs():
    close, far = pair(b=3)
    sep = make("close_peak")
    s, n = sep.separate(close, far)
    assert s.shape == n.shape == close.shape and s.dtype == np.float32
    for i in range(3):
        np.testing.assert_array_equal(sep.separate(close[i], far[i])[0], s[i])
    st, nt = sep.separate(torch.from_numpy(close), torch.from_numpy(far))
    assert isinstance(st, torch.Tensor) and torch.equal(st, torch.from_numpy(s))


def test_swapped_order_returns_far_image():
    close, far = pair()
    sep = make("common_close_anchor", order=("far", "close"), peak=0.9)
    with pytest.raises(ValueError):
        sep.separate(close, far)
    np.testing.assert_allclose(sep.far_image(close, far), 0.5 * far, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):
        make("close_peak").far_image(close, far)


def test_gains():
    close, far = pair()
    pc, pf = np.abs(close).max(), np.abs(far).max()
    gc, gf = peak_gains(close, far, "independent_peak", 0.9)
    assert np.isclose(gc[0] * pc, 0.9) and np.isclose(gf[0] * pf, 0.9)
    g, g2 = peak_gains(close, far, "joint_peak", 0.9)
    assert g is g2 and np.isclose(g[0] * max(pc, pf), 0.9)
    g, _ = peak_gains(close, far, "close_peak", 0.5)
    assert np.isclose(g[0] * pc, 0.5)
    gc, gf = peak_gains(np.zeros(8), np.zeros(8), "independent_peak", 0.9)
    assert gc[0] == gf[0] == 1.0


def test_cfg_validation():
    with pytest.raises(ValueError):
        check_cfg({"model": "x", "input_order": ["close", "far"], "norm": {"type": "none"}})
    with pytest.raises(ValueError):
        check_cfg({"model": "resunet30", "input_order": ["close", "close"], "norm": {"type": "none"}})
    with pytest.raises(ValueError):
        check_cfg({"model": "resunet30", "input_order": ["close", "far"], "norm": {"type": "rms"}})


def test_resunet30_io_shape():
    pytest.importorskip("torchlibrosa")
    from sounddiffsep.models.resunet.resunet_unconditional import ResUNet30_Unconditional

    net = ResUNet30_Unconditional(input_channels=2, output_channels=1).eval()
    sep = Separator(net, {"model": "resunet30", "input_order": ["close", "far"],
                          "norm": {"type": "close_peak", "peak": 0.5}}, "cpu")
    close, far = pair(t=16000, b=2)
    s, n = sep.separate(close, far)
    assert s.shape == (2, 16000) and np.isfinite(s).all()
    np.testing.assert_allclose(s + n, close, atol=1e-6)


def test_register_model_plugs_in_a_builder(monkeypatch):
    import sounddiffsep.inference as inference

    class WithDropout(torch.nn.Module):
        dropout_probability = 0.1

        def forward(self, x):
            return x[:, :1]

    monkeypatch.setattr(inference, "_BUILDERS", dict(inference._BUILDERS))
    cfg = {"model": "toy", "input_order": ["close", "far"], "norm": {"type": "none"}}
    with pytest.raises(ValueError, match="cfg\\['model'\\]"):
        check_cfg(cfg)
    seen = {}
    inference.register_model("toy", lambda path, c: seen.setdefault("args", (path, c)) and WithDropout())
    net = inference._build("unused.pt", cfg, "cpu")
    assert seen["args"] == ("unused.pt", cfg)
    assert not net.training and net.dropout_probability == 0.0


def test_bfloat16_torch_inputs():
    close, far = pair()
    sep = make("close_peak")
    c16, f16 = torch.from_numpy(close).bfloat16(), torch.from_numpy(far).bfloat16()
    st, nt = sep.separate(c16, f16)
    expected, _ = sep.separate(c16.float().numpy(), f16.float().numpy())
    assert st.dtype == torch.float32 and torch.equal(st, torch.from_numpy(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_torch_results_stay_on_the_input_device():
    close, far = pair()
    sep = make("close_peak", order=("far", "close"))
    c, f = torch.from_numpy(close).cuda(), torch.from_numpy(far).cuda()
    y = sep.far_image(c, f)
    assert y.device == f.device
    _ = f - y  # documented residual expression
    s, n = make("close_peak").separate(c, f)
    assert s.device == n.device == c.device
