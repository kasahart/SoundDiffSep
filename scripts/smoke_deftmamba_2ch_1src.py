#!/usr/bin/env python3
"""Run the primary 2-microphone/1-source waveform smoke test on CUDA."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time

import torch

from sounddiffsep.models.deftmamba.model import DeFTMambaCore
from sounddiffsep.models.deftmamba.waveform_wrapper import DeFTMambaWaveformWrapper


def reset_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> DeFTMambaWaveformWrapper:
    core = DeFTMambaCore(
        n_mics=2,
        n_srcs=1,
        n_layers=args.n_layers,
        emb_dim=64,
        n_heads=4,
        emb_kernel_size=4,
        emb_hop_size=1,
        hidden_dim=256,
        dropout=0.1,
    )
    return DeFTMambaWaveformWrapper(
        core,
        n_mics=2,
        n_srcs=1,
        sample_rate=args.sample_rate,
        n_fft=512,
        win_length=512,
        hop_length=256,
        window="hamming",
        center=True,
        periodic=True,
        normalized=False,
    ).to(args.device)


def finite_gradients(model: torch.nn.Module) -> tuple[int, bool]:
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    return len(gradients), bool(
        gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The upstream-faithful smoke test requires CUDA")
    if args.sample_rate <= 0 or args.seconds <= 0 or args.batch_size <= 0:
        raise ValueError("sample rate, seconds, and batch size must be positive")

    reset_seed(args.seed)
    device = torch.device(args.device)
    samples = round(args.sample_rate * args.seconds)
    model = build_model(args).train()
    waveform = torch.randn(
        args.batch_size,
        2,
        samples,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    total_started = time.perf_counter()
    forward_started = total_started
    output = model(waveform)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    loss = output.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    forward_backward_seconds = time.perf_counter() - total_started
    populated_gradients, gradients_finite = finite_gradients(model)
    forward_backward_peak_allocated = torch.cuda.max_memory_allocated(device)
    forward_backward_peak_reserved = torch.cuda.max_memory_reserved(device)

    state_buffer = io.BytesIO()
    torch.save(model.state_dict(), state_buffer)
    state_buffer.seek(0)
    restored = build_model(args)
    restored.load_state_dict(
        torch.load(state_buffer, map_location=device, weights_only=True), strict=True
    )
    model.eval()
    restored.eval()
    replay_seed = args.seed + 1
    with torch.no_grad():
        reset_seed(replay_seed)
        before_reload = model(waveform.detach())
        reset_seed(replay_seed)
        after_reload = restored(waveform.detach())
        reset_seed(replay_seed + 1)
        first_eval = model(waveform.detach())
        second_eval = model(waveform.detach())
    torch.cuda.synchronize(device)
    eval_difference = (first_eval - second_eval).abs()

    parameters = list(model.parameters())
    payload = {
        "configuration": {
            "sample_rate": args.sample_rate,
            "duration_seconds": args.seconds,
            "n_mics": 2,
            "n_srcs": 1,
            "n_layers": args.n_layers,
            "emb_dim": 64,
            "batch_size": args.batch_size,
            "dtype": "float32",
            "dropout": 0.1,
            "n_fft": 512,
            "win_length": 512,
            "hop_length": 256,
            "window": "hamming",
        },
        "input_shape": list(waveform.shape),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "input_gradient_finite": bool(torch.isfinite(waveform.grad).all().item()),
        "parameter_gradients_present": populated_gradients,
        "parameter_gradients_finite": gradients_finite,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "loss": float(loss.item()),
        "forward_seconds": forward_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "serialization_replay_seed": replay_seed,
        "serialization_max_abs_error": float(
            (before_reload - after_reload).abs().max().item()
        ),
        "eval_repeat_without_rng_reset": {
            "exactly_equal": bool(torch.equal(first_eval, second_eval)),
            "maximum_absolute_difference": float(eval_difference.max().item()),
            "mean_absolute_difference": float(eval_difference.mean().item()),
        },
        "forward_backward_peak_allocated_bytes": forward_backward_peak_allocated,
        "forward_backward_peak_reserved_bytes": forward_backward_peak_reserved,
        "note": (
            "Attention dropout is disabled in eval mode, so repeated eval outputs are "
            "deterministic; the CUDA RNG seed is still replayed for the serialization "
            "comparison so results stay comparable with earlier evidence."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
