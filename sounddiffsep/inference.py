"""Two-microphone separator inference with the preprocessing stated next to the checkpoint.

    from sounddiffsep.inference import load
    sep = load("separator.pt", "cuda:0")          # reads separator.pt.preprocess.json
    s_hat, n_hat = sep.separate(close, far)                # original level, n_hat = close - s_hat

Sidecar ``<ckpt>.preprocess.json`` (or pass the same dict as ``cfg=``)::

    {"model": "resunet30" | "deftmamba" | "dcunet",
     "input_order": ["close", "far"],                     # or ["far", "close"] (production DeFT swap)
     "norm": {"type": "close_peak", "peak": 0.5},         # see sounddiffsep.normalization
     "checkpoint_schema": "relative_mixture_training_v1", # deftmamba only (the only schema supported)
     "config": "<yaml>",                                   # dcunet only (training config)
     "tf32": false}                                        # optional; true = TF32 matmul/cuDNN (process-global)

Conventions (docs/inference.md):
  * ch0 = close (near the machine), ch1 = far.  All models output the image of their target at the
    FIRST model-input channel; ResUNet30 applies its mask to that channel (DeFT-Mamba complex-mask
    checkpoints with reference_channel != 0 are rejected).
  * input_order ["close", "far"] is the designed use: ``separate`` returns ``(s_hat, n_hat)`` with
    ``n_hat = close - s_hat``.
  * input_order ["far", "close"] (production DeFT-Mamba swap) makes the output the image at the FAR
    microphone of the source dominant there (the far-noise estimate N_f).  ``separate`` refuses this
    order; use ``far_image(close, far)`` (and ``far - far_image`` if a residual is wanted).
  * Gains come from the inputs passed in (pass the RAW pair), are applied per channel and undone on the
    output with the gain of the first model-input channel.  Dropout is off (eval(), and attention
    ``dropout_probability`` set to 0 for DeFT-Mamba).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .normalization import NORM_TYPES, peak_gains

MODELS = ("resunet30", "deftmamba", "dcunet")
# Modes taken from 2chssDNN-side scripts divide by the float32 gain; modes taken from the DeFT worker
# multiply by float32(1 / gain).  Kept apart so both references are reproduced bit for bit.
_RESTORE_BY_DIVISION = ("close_peak", "joint_peak", "none")


def read_sidecar(ckpt_path) -> dict:
    path = Path(str(ckpt_path) + ".preprocess.json")
    if not path.exists():
        raise FileNotFoundError(f"missing sidecar {path}")
    return json.loads(path.read_text())


def check_cfg(cfg: dict) -> dict:
    if cfg.get("model") not in MODELS:
        raise ValueError(f"cfg['model'] must be one of {MODELS}, got {cfg.get('model')!r}")
    order = list(cfg.get("input_order", ()))
    if sorted(order) != ["close", "far"]:
        raise ValueError(f"input_order must be a permutation of ['close', 'far'], got {order}")
    norm = cfg.get("norm", {})
    if norm.get("type") not in NORM_TYPES:
        raise ValueError(f"norm.type must be one of {NORM_TYPES}, got {norm.get('type')!r}")
    if norm["type"] != "none" and not 0 < float(norm.get("peak", 0)) <= 1:
        raise ValueError(f"norm.peak must be in (0, 1], got {norm.get('peak')!r}")
    return cfg


class Separator:
    def __init__(self, net: torch.nn.Module, cfg: dict, device):
        self.net, self.cfg, self.device = net, check_cfg(cfg), torch.device(device)

    @property
    def swapped(self) -> bool:
        return self.cfg["input_order"][0] == "far"

    def _run(self, close, far):
        is_torch = isinstance(close, torch.Tensor)
        # float() on the torch side: NumPy has no bfloat16.
        c = close.detach().float().cpu().numpy() if is_torch else np.asarray(close)
        f = far.detach().float().cpu().numpy() if isinstance(far, torch.Tensor) else np.asarray(far)
        c, f = c.astype(np.float32, copy=False), f.astype(np.float32, copy=False)
        one = c.ndim == 1
        if one:
            c, f = c[None], f[None]
        if c.shape != f.shape or c.ndim != 2:
            raise ValueError(f"close/far must both be [T] or [B, T], got {c.shape} and {f.shape}")
        norm = self.cfg["norm"]
        gains = dict(zip(("close", "far"), peak_gains(c, f, norm["type"], float(norm.get("peak", 1.0)))))
        scaled = {"close": c * gains["close"].astype(np.float32), "far": f * gains["far"].astype(np.float32)}
        x = np.stack([scaled[k] for k in self.cfg["input_order"]], axis=1)
        with torch.inference_mode():
            out = self.net(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
        out = out.reshape(out.shape[0], -1, out.shape[-1])[:, 0].float().cpu().numpy()
        g = gains[self.cfg["input_order"][0]]
        if norm["type"] in _RESTORE_BY_DIVISION:
            y = out / g.astype(np.float32)
        else:
            y = out * (1.0 / g).astype(np.float32)
        y = y.astype(np.float32)
        return y, c, (lambda a: a[0] if one else a), is_torch

    def separate(self, close, far):
        """close, far: [T] or [B, T] (numpy or torch). Returns float32 ``(s_hat, n_hat)``, same kind as ``close``."""
        if self.swapped:
            raise ValueError("input_order ['far', 'close'] outputs the far-channel image; use far_image()")
        s, c, squeeze, is_torch = self._run(close, far)
        n = (c - s).astype(np.float32)
        s, n = squeeze(s), squeeze(n)
        if is_torch:
            return torch.from_numpy(s), torch.from_numpy(n)
        return s, n

    def far_image(self, close, far):
        """Swapped order only: the model output at the original far level (production: far-noise N_f)."""
        if not self.swapped:
            raise ValueError("far_image() needs input_order ['far', 'close']")
        y, _, squeeze, is_torch = self._run(close, far)
        y = squeeze(y)
        return torch.from_numpy(y) if is_torch else y


def _strip(state: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in state):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    return state


def _build(ckpt_path, cfg: dict, device) -> torch.nn.Module:
    kind = cfg["model"]
    if kind == "resunet30":
        from .models.resunet.resunet_unconditional import ResUNet30_Unconditional

        net = ResUNet30_Unconditional(input_channels=2, output_channels=1)
        from .models.deftmamba.checkpoint import load_weights_only

        payload = load_weights_only(ckpt_path)
        state = payload["state_dict"] if "state_dict" in payload else payload
        net.load_state_dict(_strip(dict(state), "ss_model."), strict=True)
    elif kind == "deftmamba":
        from .models.deftmamba.checkpoint import RELATIVE_MIXTURE_SCHEMA, load_relative_mixture_checkpoint

        schema = cfg.get("checkpoint_schema", RELATIVE_MIXTURE_SCHEMA)
        if schema != RELATIVE_MIXTURE_SCHEMA:
            raise ValueError(f"unsupported DeFT-Mamba checkpoint schema {schema!r}")
        net, _ = load_relative_mixture_checkpoint(ckpt_path, "cpu")
        if getattr(net, "reference_channel", 0) != 0:
            # The gain restore and n_hat = close - s_hat assume the image at the first input channel.
            raise ValueError(
                f"complex-mask reference_channel={net.reference_channel} is not supported; "
                "the separator convention needs the image at the first model-input channel"
            )
    else:  # dcunet
        from .model_loader import _load_dcunet

        net = _load_dcunet(cfg["config"], str(ckpt_path), "cpu")
    net = net.to(device).eval()
    for module in net.modules():
        if hasattr(module, "dropout_probability"):
            module.dropout_probability = 0.0
    net.requires_grad_(False)
    return net


def load(ckpt_path, device="cuda:0", cfg: dict[str, Any] | None = None) -> Separator:
    """Build a :class:`Separator`; ``cfg`` defaults to the sidecar ``<ckpt>.preprocess.json``."""
    cfg = check_cfg(dict(cfg) if cfg is not None else read_sidecar(ckpt_path))
    # Process-global.  "tf32": true reproduces the DeFT production worker, whose trainer import sets
    # torch.set_float32_matmul_precision("high") (TF32 matmuls, ~1e-4 different outputs).
    tf32 = bool(cfg.get("tf32", False))
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cudnn.allow_tf32 = tf32
    return Separator(_build(ckpt_path, cfg, device), cfg, device)
