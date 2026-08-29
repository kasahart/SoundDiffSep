#!/usr/bin/env python3
"""Benchmark the primary 2ch/1src DeFT-Mamba waveform configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from sounddiffsep.models.deftmamba.model import DeFTMambaCore
from sounddiffsep.models.deftmamba.waveform_wrapper import DeFTMambaWaveformWrapper


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(
    model: torch.nn.Module, waveform: torch.Tensor, device: torch.device
) -> tuple[float, torch.Tensor]:
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(waveform)
    synchronize(device)
    return time.perf_counter() - started, output


def timed_forward_backward(
    model: torch.nn.Module, waveform: torch.Tensor, device: torch.device
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    output = model(waveform)
    output.square().mean().backward()
    synchronize(device)
    return time.perf_counter() - started, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The upstream-faithful benchmark requires CUDA")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
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
    model = DeFTMambaWaveformWrapper(
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
    ).to(device)
    samples = round(args.sample_rate * args.seconds)
    waveform = torch.randn(
        args.batch_size, 2, samples, device=device, dtype=torch.float32
    )
    for _ in range(args.warmup):
        model.train()
        timed_forward_backward(model, waveform, device)

    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    output = None
    for _ in range(args.iterations):
        elapsed, output = timed_forward(model, waveform, device)
        forward_times.append(elapsed)
    forward_peak_allocated = torch.cuda.max_memory_allocated(device)
    forward_peak_reserved = torch.cuda.max_memory_reserved(device)

    output = None
    torch.cuda.empty_cache()
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    forward_backward_times = []
    for _ in range(args.iterations):
        elapsed, output = timed_forward_backward(model, waveform, device)
        forward_backward_times.append(elapsed)
    backward_peak_allocated = torch.cuda.max_memory_allocated(device)
    backward_peak_reserved = torch.cuda.max_memory_reserved(device)
    assert output is not None
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    mean_forward = statistics.mean(forward_times)
    mean_forward_backward = statistics.mean(forward_backward_times)
    processed_audio_seconds = args.batch_size * args.seconds
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
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "input_shape": list(waveform.shape),
        "output_shape": list(output.shape),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "forward_seconds": forward_times,
        "forward_seconds_mean": mean_forward,
        "forward_backward_seconds": forward_backward_times,
        "forward_backward_seconds_mean": mean_forward_backward,
        "steps_per_second": 1.0 / mean_forward_backward,
        "real_time_factor": mean_forward / processed_audio_seconds,
        "peak_allocated_vram_bytes": {
            "forward": forward_peak_allocated,
            "forward_backward": backward_peak_allocated,
        },
        "peak_reserved_vram_bytes": {
            "forward": forward_peak_reserved,
            "forward_backward": backward_peak_reserved,
        },
        "output_finite": bool(torch.isfinite(output).all().item()),
        "gradient_finite": bool(
            gradients and all(torch.isfinite(value).all().item() for value in gradients)
        ),
        "macs": None,
        "macs_note": "not measured; no estimate is reported",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
