"""Two-microphone (close/far) scene generator, numpy, one scene per call.

    scene = make_scene(target, noises, dl_db=8, snr_db=0, d0=120, norm="close_peak", peak=0.9)
    scene["close"], scene["far"], scene["target_close"]          # target_close = clean target image at close

Signal model (all images are on the same sample grid of length T)::

    close = t_c + a * sum_k r_k * n_ck
    far   = EQ( shift_D0( g_t * t_f + a * sum_k n_fk ) )

Images.  Either pass per-mic images (``target_far`` / ``noises_far``; ``target`` / ``noises`` are then
the close images, e.g. RIR convolutions as in the NA-BEATs mixer), or pass dry waveforms and let the
generator build the far image of the target and the close image of each noise as
``delay + optional short FIR`` of the dry signal (the e2c benchmark v2 model: each source reaches the
OTHER microphone through ``[0]*delay + fir``).

Levels (each measured as energy over the output window, so they are hit exactly):
  * ``dl_db``: target leakage, ``E(t_c) / E(g_t t_f) = 10^(dl/10)`` (the far target image is rescaled,
    as in the S2b mixer).  Negative values = target louder at far.  ``None`` keeps the given level.
  * ``noise_ratio_db[k]``: per-noise ``E(close_k) / E(far_k)`` (the close image is rescaled; ``None``
    keeps the given level).
  * ``snr_db``: ``E(t_c) / E(a * sum n_ck)`` at the close mic; one noise gain ``a`` is shared by both mics.

Inter-channel offset ``d0`` (samples, ``far[t] = far_unshifted[t - d0]``, d0 > 0 delays far), mode:
  * ``"circular"``: ``np.roll`` of the far images (no zeros; a wrap splice at the edge).  With dry
    sources this matches the S2b mixer's circular padding except near the splice.
  * ``"zero"``: zero-filled shift (e2c benchmark v2; leaves |d0| zeros at one edge).
  * ``"margin"``: inputs are ``T + 2 * margin`` long; close uses ``[margin, margin + T)`` and far uses
    ``[margin - d0, margin - d0 + T)`` (no zeros, no splice; needs ``|d0| <= margin``).

Far microphone type (optional, after the offset): ``far_gain_db`` and ``far_tilt_db_per_oct`` (zero-phase
tilt around 1 kHz, as in the NA-BEATs mixer).  These change the observed close/far ratios of every source,
so ``dl_db`` and ``noise_ratio_db`` refer to the levels before them.

Output normalization: ``norm`` / ``peak`` are the modes of :mod:`sounddiffsep.normalization`; the close
gain also scales ``target_close`` / ``noise_close`` and the far gain the far images.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from ..normalization import peak_gains

D0_MODES = ("circular", "zero", "margin")


def _energy(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def cross_path(x: np.ndarray, delay: int = 0, fir=None) -> np.ndarray:
    """``x`` through ``[0] * delay + fir`` (fir defaults to [1]), truncated to ``len(x)``."""
    h = np.concatenate([np.zeros(int(delay)), np.atleast_1d(np.asarray(fir if fir is not None else [1.0], float))])
    return lfilter(h, [1.0], x)


def shift(x: np.ndarray, d: int) -> np.ndarray:
    """Zero-filled delay by ``d`` samples (d < 0 advances)."""
    y = np.zeros_like(x)
    if d >= 0:
        y[d:] = x[: len(x) - d]
    else:
        y[: len(x) + d] = x[-d:]
    return y


def tilt(x: np.ndarray, db_per_oct: float, sample_rate: int = 16000) -> np.ndarray:
    """Zero-phase spectral tilt around 1 kHz (NA-BEATs mixer curve, clipped to +-12 dB)."""
    n_fft = 1 << (2 * len(x) - 1).bit_length()
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    curve = np.clip(db_per_oct * (np.log2(np.maximum(f, 20.0)) - math.log2(1000.0)), -12.0, 12.0)
    return np.fft.irfft(np.fft.rfft(x, n_fft) * 10 ** (curve / 20), n_fft)[: len(x)]


def make_scene(target, noises, *, target_far=None, noises_far=None, dl_db=None, snr_db=None,
               noise_ratio_db=None, target_delay=0, target_fir=None, noise_delays=None, noise_firs=None,
               d0=0, d0_mode="circular", margin=0, far_gain_db=0.0, far_tilt_db_per_oct=0.0,
               norm="none", peak=0.9, sample_rate=16000, eps=1e-12) -> dict:
    """Build one scene; see the module docstring.  ``noises``: ``[K, L]`` (K >= 1)."""
    if d0_mode not in D0_MODES:
        raise ValueError(f"d0_mode must be one of {D0_MODES}")
    d0 = int(d0)
    t_c = np.asarray(target, dtype=np.float64)
    n_c = np.atleast_2d(np.asarray(noises, dtype=np.float64))
    k = n_c.shape[0]
    if target_far is None:
        t_f = cross_path(t_c, target_delay, target_fir)
    else:
        t_f = np.asarray(target_far, dtype=np.float64)
    if noises_far is None:
        n_f = n_c
        delays = noise_delays if noise_delays is not None else [0] * k
        firs = noise_firs if noise_firs is not None else [None] * k
        n_c = np.stack([cross_path(n_f[i], delays[i], firs[i]) for i in range(k)])
    else:
        n_f = np.atleast_2d(np.asarray(noises_far, dtype=np.float64))
    if not (t_c.shape == t_f.shape == n_c.shape[1:] == n_f.shape[1:]):
        raise ValueError("all images must have the same length")

    # output window
    if d0_mode == "margin":
        m = int(margin)
        if abs(d0) > m:
            raise ValueError(f"|d0| = {abs(d0)} exceeds margin {m}")
        length = t_c.shape[-1] - 2 * m
        close_win = lambda x: x[..., m:m + length]  # noqa: E731
        far_win = lambda x: x[..., m - d0:m - d0 + length]  # noqa: E731
    elif d0_mode == "circular":
        close_win = lambda x: x  # noqa: E731
        far_win = lambda x: np.roll(x, d0, axis=-1)  # noqa: E731
    else:
        close_win = lambda x: x  # noqa: E731
        far_win = lambda x: np.stack([shift(v, d0) for v in np.atleast_2d(x)]).reshape(x.shape)  # noqa: E731
    t_c, n_c, t_f, n_f = close_win(t_c), close_win(n_c), far_win(t_f), far_win(n_f)

    if noise_ratio_db is not None:
        r = np.asarray(noise_ratio_db, dtype=np.float64).reshape(-1)
        for i in range(k):
            n_c[i] = n_c[i] * math.sqrt(_energy(n_f[i]) * 10 ** (r[i] / 10) / (_energy(n_c[i]) + eps))
    g_t = 1.0 if dl_db is None else math.sqrt(_energy(t_c) / (_energy(t_f) * 10 ** (dl_db / 10) + eps))
    t_f = g_t * t_f
    nc, nf = n_c.sum(0), n_f.sum(0)
    a = 1.0 if snr_db is None else math.sqrt(_energy(t_c) / (_energy(nc) * 10 ** (snr_db / 10) + eps))
    nc, nf = a * nc, a * nf

    def far_mic(x):
        if far_tilt_db_per_oct:
            x = tilt(x, far_tilt_db_per_oct, sample_rate)
        return x * 10 ** (far_gain_db / 20)

    t_f, nf = far_mic(t_f), far_mic(nf)
    close, far = t_c + nc, t_f + nf
    gc, gf = (float(v[0]) for v in peak_gains(close, far, norm, peak))
    return {"close": close * gc, "far": far * gf, "target_close": t_c * gc, "target_far": t_f * gf,
            "noise_close": nc * gc, "noise_far": nf * gf, "gain_close": gc, "gain_far": gf,
            "noise_gain": a, "far_target_gain": g_t}
