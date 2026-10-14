"""Per-file peak normalization of a close/far pair (numpy, CPU).

``peak_gains(close, far, norm_type, peak)`` returns float64 gains ``(g_close, g_far)``
of shape ``close.shape[:-1] + (1,)``; the model input is ``(close * g_close, far * g_far)``.

norm types (see docs/inference.md for which model was trained with which):

==================== ============================================== =====================================
type                 gains                                          used by
==================== ============================================== =====================================
independent_peak     g_close = p / peak(close), g_far = p / peak(far) DeFT-Mamba training (p = 0.9),
                     (a channel with peak 0 keeps g = 1)            original SoundDiffSep models
common_close_anchor  g = p / (peak(close) + 1e-12), both channels   DeFT-Mamba production worker (0.9)
joint_peak           g = p / max(peak(close), peak(far), 1e-8)      ResUNet30 2chssDNN init (0.9)
close_peak           g = p / max(peak(close), 1e-8), both channels  ResUNet30 fine-tuned with close-peak U(0.1, 0.9) training (0.5)
none                 g = 1
==================== ============================================== =====================================

The small epsilons follow the reference implementations each mode was copied from, so that
results reproduce them bit for bit.
"""

from __future__ import annotations

import numpy as np

NORM_TYPES = ("independent_peak", "common_close_anchor", "joint_peak", "close_peak", "none")


def _peak(x: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(x, dtype=np.float64)).max(axis=-1, keepdims=True)


def peak_gains(close, far, norm_type: str, peak: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Gains ``(g_close, g_far)`` (float64, shape ``[..., 1]``) for one or a batch of pairs."""

    if norm_type not in NORM_TYPES:
        raise ValueError(f"unknown norm type {norm_type!r}; expected one of {NORM_TYPES}")
    pc, pf = _peak(close), _peak(far)
    if norm_type == "none":
        g = np.ones_like(pc)
        return g, g
    if norm_type == "independent_peak":
        gc = np.where(pc > 0, peak / np.where(pc > 0, pc, 1.0), 1.0)
        gf = np.where(pf > 0, peak / np.where(pf > 0, pf, 1.0), 1.0)
        return gc, gf
    if norm_type == "common_close_anchor":
        g = peak / (pc + 1e-12)
    elif norm_type == "joint_peak":
        g = peak / np.maximum(np.maximum(pc, pf), 1e-8)
    else:  # close_peak
        g = peak / np.maximum(pc, 1e-8)
    return g, g
