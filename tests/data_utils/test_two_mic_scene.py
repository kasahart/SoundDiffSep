from __future__ import annotations

import numpy as np
import pytest

from sounddiffsep.data_utils.two_mic_scene import make_scene, shift

T = 16000


def db(a, b):
    return 10 * np.log10(np.sum(a ** 2) / np.sum(b ** 2))


def lag(ref, x, w=400):
    """d such that x ~ ref delayed by d (circular cross-correlation peak)."""
    cc = np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(ref)), len(x))
    cc = np.concatenate([cc[-w:], cc[:w + 1]])
    return int(np.argmax(np.abs(cc))) - w


def sources(seed, k=2, n=T):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), rng.standard_normal((k, n)), rng


@pytest.mark.parametrize("mode", ["circular", "zero", "margin"])
@pytest.mark.parametrize("dl", [-5.0, 0.0, 8.0, 30.0])
def test_requested_dl_snr_d0(mode, dl):
    margin = 400 if mode == "margin" else 0
    s, n, rng = sources(1, n=T + 2 * margin)
    d0 = int(rng.integers(-350, 351))
    fir = rng.standard_normal(3) * [1, 0.3, 0.1]
    sc = make_scene(s, n, dl_db=dl, snr_db=-3.0, d0=d0, d0_mode=mode, margin=margin,
                    target_delay=7, target_fir=fir, noise_delays=[3, 20], noise_firs=[fir, None],
                    noise_ratio_db=[2.0, -4.0], norm="none")
    assert np.isclose(db(sc["target_close"], sc["target_far"]), dl, atol=1e-6)
    assert np.isclose(db(sc["target_close"], sc["noise_close"]), -3.0, atol=1e-6)
    np.testing.assert_allclose(sc["close"], sc["target_close"] + sc["noise_close"])
    np.testing.assert_allclose(sc["far"], sc["target_far"] + sc["noise_far"])
    # the far noise image is the dry noise sum (shifted): its lag vs the unshifted sum is d0
    nsum = n.sum(0)[margin:margin + T] if margin else n.sum(0)
    assert lag(nsum, sc["noise_far"]) == d0


def test_per_noise_ratio_and_given_images():
    s, n, rng = sources(2)
    s_far, n_far = 0.5 * rng.standard_normal(T), rng.standard_normal((2, T))
    sc = make_scene(s, n, target_far=s_far, noises_far=n_far, noise_ratio_db=[6.0, -2.0], snr_db=None)
    k = sc["noise_close"] - (n[0] * np.sqrt(np.sum(n_far[0] ** 2) * 10 ** 0.6 / np.sum(n[0] ** 2))
                             + n[1] * np.sqrt(np.sum(n_far[1] ** 2) * 10 ** -0.2 / np.sum(n[1] ** 2)))
    assert np.abs(k).max() < 1e-9
    np.testing.assert_array_equal(sc["target_far"], s_far)


@pytest.mark.parametrize("norm,check", [
    ("close_peak", lambda sc: np.isclose(np.abs(sc["close"]).max(), 0.5)),
    ("joint_peak", lambda sc: np.isclose(max(np.abs(sc["close"]).max(), np.abs(sc["far"]).max()), 0.5)),
    ("independent_peak", lambda sc: np.isclose(np.abs(sc["far"]).max(), 0.5) and np.isclose(np.abs(sc["close"]).max(), 0.5)),
])
def test_output_normalization(norm, check):
    s, n, _ = sources(3)
    sc = make_scene(s, n, dl_db=3.0, snr_db=0.0, norm=norm, peak=0.5)
    assert check(sc)
    np.testing.assert_allclose(sc["close"], sc["target_close"] + sc["noise_close"])


def test_far_gain_and_tilt():
    s, n, _ = sources(4)
    base = make_scene(s, n, dl_db=0.0, snr_db=0.0)
    g = make_scene(s, n, dl_db=0.0, snr_db=0.0, far_gain_db=-6.0)
    np.testing.assert_allclose(g["far"], base["far"] * 10 ** (-6 / 20))
    tl = make_scene(s, n, dl_db=0.0, snr_db=0.0, far_tilt_db_per_oct=3.0)
    spec = lambda x: np.abs(np.fft.rfft(x)) ** 2  # noqa: E731
    f = np.fft.rfftfreq(T, 1 / 16000)
    hi, lo = f > 4000, (f > 100) & (f < 500)
    ratio = lambda x: spec(x)[hi].mean() / spec(x)[lo].mean()  # noqa: E731
    assert ratio(tl["far"]) > 10 * ratio(base["far"])


def test_zero_mode_shift_leaves_zeros():
    s, n, _ = sources(5)
    sc = make_scene(s, n, d0=100, d0_mode="zero")
    assert np.all(sc["far"][:100] == 0)
    np.testing.assert_allclose(sc["noise_far"], shift(n.sum(0), 100))


@pytest.mark.parametrize("d", [T, T + 1, -T, -(T + 7)])
def test_shift_beyond_length_is_silence(d):
    x = np.arange(1.0, T + 1)
    assert np.all(shift(x, d) == 0)


def test_noise_ratio_does_not_modify_caller_images():
    s, n, rng = sources(6)
    n_far = rng.standard_normal(n.shape)
    s_far = rng.standard_normal(T)
    before = n.copy()
    a = make_scene(s, n, target_far=s_far, noises_far=n_far, noise_ratio_db=[3.0, -2.0], snr_db=0)
    np.testing.assert_array_equal(n, before)
    b = make_scene(s, n, target_far=s_far, noises_far=n_far, noise_ratio_db=[3.0, -2.0], snr_db=0)
    np.testing.assert_array_equal(a["close"], b["close"])


def test_noise_counts_must_match():
    s, n, rng = sources(7)
    with pytest.raises(ValueError, match="same K"):
        make_scene(s, n, target_far=s, noises_far=n[:1])
    with pytest.raises(ValueError, match="one value per noise"):
        make_scene(s, n, noise_ratio_db=[3.0])
