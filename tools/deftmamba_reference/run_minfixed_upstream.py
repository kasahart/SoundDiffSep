#!/usr/bin/env python3
"""Exercise the patched external reference with the validated 2ch/1src shape."""

from __future__ import annotations

import argparse
import importlib.util
import io
import time
from pathlib import Path

from common import (
    DEFAULT_CHECKOUT,
    UPSTREAM_COMMIT,
    apply_minimal_patch,
    prepare_checkout,
    write_json,
)


def load_module(source: Path):
    specification = importlib.util.spec_from_file_location(
        "deftmamba_upstream_minfix", source
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def all_finite(tensors) -> bool:
    import torch

    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--n-mics", type=int, default=2)
    parser.add_argument("--n-srcs", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--frequency-bins", type=int, default=12)
    parser.add_argument("--time-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    checkout = prepare_checkout(args.checkout)
    patch_status = apply_minimal_patch(checkout)
    upstream = load_module(checkout / "DeFTMamba.py")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream FlashAttention path")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model_arguments = dict(
        audio_mode="separation",
        class_mode="unused",
        onoff_mode="unused",
        n_mics=args.n_mics,
        n_srcs=args.n_srcs,
        n_layers=args.n_layers,
        emb_dim=64,
        hidden_dim=256,
        n_head=4,
        emb_ks=4,
        emb_hs=1,
        dropout=0.0,
    )
    model = upstream.DeFTMamba(**model_arguments).to(device)
    model.train()
    model_input = torch.randn(
        1,
        args.frequency_bins,
        args.time_frames,
        2 * args.n_mics,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model(model_input)
    loss = output.square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    parameters = list(model.parameters())
    populated_gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    state_buffer = io.BytesIO()
    torch.save(model.state_dict(), state_buffer)
    state_buffer.seek(0)
    restored = upstream.DeFTMamba(**model_arguments).to(device)
    restored.load_state_dict(torch.load(state_buffer, map_location=device, weights_only=True))
    restored.eval()
    model.eval()
    with torch.no_grad():
        original_after_save = model(model_input.detach())
        restored_output = restored(model_input.detach())

    payload = {
        "upstream_commit": UPSTREAM_COMMIT,
        "patch_status": patch_status,
        "configuration": model_arguments,
        "input_shape": list(model_input.shape),
        "output_shape": list(output.shape),
        "expected_output_shape": [
            1,
            args.frequency_bins,
            args.time_frames,
            2 * args.n_srcs,
        ],
        "output_finite": bool(torch.isfinite(output).all().item()),
        "input_gradient_finite": bool(torch.isfinite(model_input.grad).all().item()),
        "parameter_gradients_present": len(populated_gradients),
        "parameter_gradients_finite": all_finite(populated_gradients),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "serialization_max_abs_error": float(
            (original_after_save - restored_output).abs().max().item()
        ),
        "forward_backward_seconds": elapsed,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    write_json(payload, args.output)


if __name__ == "__main__":
    main()
