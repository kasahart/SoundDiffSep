#!/usr/bin/env python3
"""Overfit one deterministic synthetic 2ch/1src batch as a learning smoke test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as functional

from sounddiffsep.models.deftmamba.complex_io import pack_complex_spectrogram
from sounddiffsep.models.deftmamba.model import DeFTMambaCore
from sounddiffsep.models.deftmamba.waveform_wrapper import DeFTMambaWaveformWrapper


def delayed(signal: torch.Tensor, samples: int) -> torch.Tensor:
    result = torch.zeros_like(signal)
    if samples == 0:
        result.copy_(signal)
    else:
        result[..., samples:] = signal[..., :-samples]
    return result


def synthetic_batch(
    *, batch_size: int, sample_rate: int, samples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    timeline = torch.arange(samples, dtype=torch.float32) / sample_rate
    target = (
        0.45 * torch.sin(2 * math.pi * 220.0 * timeline)
        + 0.25 * torch.sin(2 * math.pi * 610.0 * timeline + 0.3)
        + 0.20 * torch.sin(2 * math.pi * (130.0 * timeline + 900.0 * timeline.square()))
    )
    target = target.repeat(batch_size, 1)
    impulse_index = min(samples - 1, samples // 3)
    target[:, impulse_index] += 0.8
    noise0 = 0.025 * torch.randn(batch_size, samples, generator=generator)
    noise1 = 0.025 * torch.randn(batch_size, samples, generator=generator)
    channel0 = 0.90 * delayed(target, 2) + noise0
    channel1 = 0.65 * delayed(target, 5) + noise1
    return torch.stack((channel0, channel1), dim=1), target.unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The upstream-faithful learning smoke test requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    steps = args.steps if args.steps is not None else (12 if args.quick else 40)
    samples = 512 if args.quick else 2_048
    n_layers = 1 if args.quick else 4
    sample_rate = 16_000
    mixtures, target = synthetic_batch(
        batch_size=1, sample_rate=sample_rate, samples=samples, seed=args.seed
    )
    mixtures = mixtures.to(device)
    target = target.to(device)
    core = DeFTMambaCore(
        n_mics=2,
        n_srcs=1,
        n_layers=n_layers,
        emb_dim=64,
        n_heads=4,
        emb_kernel_size=4,
        emb_hop_size=1,
        hidden_dim=256,
        dropout=0.0,
    ).to(device)
    wrapper = DeFTMambaWaveformWrapper(
        core,
        n_mics=2,
        n_srcs=1,
        sample_rate=sample_rate,
        n_fft=64,
        win_length=64,
        hop_length=32,
        window="hamming",
        center=True,
        periodic=True,
        normalized=False,
    ).to(device)
    with torch.no_grad():
        input_ri = pack_complex_spectrogram(wrapper.waveform_to_spectrogram(mixtures))
        target_ri = pack_complex_spectrogram(wrapper.waveform_to_spectrogram(target))

    optimizer = torch.optim.Adam(core.parameters(), lr=1.0e-3)
    losses: list[float] = []
    gradients_finite = True
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        estimate_ri = core(input_ri)
        loss = functional.l1_loss(estimate_ri, target_ri)
        if not torch.isfinite(loss):
            raise RuntimeError("the one-batch loss became non-finite")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in core.parameters()
            if parameter.grad is not None
        ]
        gradients_finite = gradients_finite and bool(
            gradients and all(torch.isfinite(value).all().item() for value in gradients)
        )
        optimizer.step()
        losses.append(float(loss.item()))
    torch.cuda.synchronize(device)
    training_elapsed = time.perf_counter() - started
    with torch.no_grad():
        final_ri = core(input_ri)
        final_loss = float(functional.l1_loss(final_ri, target_ri).item())
        waveform_estimate = wrapper(mixtures)
    torch.cuda.synchronize(device)
    total_elapsed = time.perf_counter() - started
    initial_loss = losses[0]
    loss_ratio = final_loss / initial_loss
    success = final_loss < initial_loss * 0.8

    payload = {
        "purpose": "learning capability only; this is not the published training objective",
        "loss_name": "complex RI L1",
        "configuration": {
            "quick": args.quick,
            "steps": steps,
            "batch_size": 1,
            "samples": samples,
            "sample_rate": sample_rate,
            "n_mics": 2,
            "n_srcs": 1,
            "n_layers": n_layers,
            "emb_dim": 64,
            "dropout": 0.0,
            "n_fft": 64,
            "win_length": 64,
            "hop_length": 32,
        },
        "mixtures_shape": list(mixtures.shape),
        "target_shape": list(target.shape),
        "waveform_output_shape": list(waveform_estimate.shape),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_to_initial_ratio": loss_ratio,
        "loss_curve": losses + [final_loss],
        "success_criterion": "final_loss < 0.8 * initial_loss",
        "overfit_success": success,
        "output_finite": bool(torch.isfinite(waveform_estimate).all().item()),
        "gradient_finite": gradients_finite,
        "training_elapsed_seconds": training_elapsed,
        "total_elapsed_seconds_including_final_evaluation": total_elapsed,
        "training_steps_per_second": steps / training_elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not success or not payload["output_finite"] or not gradients_finite:
        raise SystemExit("one-batch overfit success criteria were not met")


if __name__ == "__main__":
    main()
