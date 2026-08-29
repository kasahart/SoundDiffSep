#!/usr/bin/env python3
"""Compare upstream_minfix and the independent SoundDiffSep implementation."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

from common import (
    DEFAULT_CHECKOUT,
    REPOSITORY_ROOT,
    UPSTREAM_COMMIT,
    apply_minimal_patch,
    prepare_checkout,
    write_json,
)


def load_upstream(source: Path):
    specification = importlib.util.spec_from_file_location(
        "deftmamba_upstream_minfix_comparison", source
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load upstream source: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def pair_state_dicts(upstream_model, local_model):
    upstream_items = list(upstream_model.state_dict().items())
    local_items = list(local_model.state_dict().items())
    if len(upstream_items) != len(local_items):
        raise AssertionError(
            f"state_dict entry count differs: {len(upstream_items)} vs {len(local_items)}"
        )

    local_state = OrderedDict()
    correspondence = []
    for (upstream_key, upstream_tensor), (local_key, local_tensor) in zip(
        upstream_items, local_items
    ):
        if upstream_tensor.shape != local_tensor.shape:
            raise AssertionError(
                "first state tensor shape mismatch: "
                f"{upstream_key} {tuple(upstream_tensor.shape)} != "
                f"{local_key} {tuple(local_tensor.shape)}"
            )
        local_state[local_key] = upstream_tensor.detach().clone()
        correspondence.append(
            {
                "upstream": upstream_key,
                "sounddiffsep": local_key,
                "shape": list(upstream_tensor.shape),
            }
        )
    local_model.load_state_dict(local_state, strict=True)
    return correspondence


def capture_upstream(model, model_input):
    import torch

    captured = OrderedDict()
    handles = []
    block = model.blocks[0]
    batch = model_input.shape[0]
    frequency_bins = model_input.shape[1]
    time_frames = model_input.shape[2]
    width = model.emb_dim

    def save_output(name):
        def hook(_module, _arguments, output):
            captured[name] = output.detach()

        return hook

    def save_input(name):
        def hook(_module, arguments):
            captured[name] = arguments[0].detach()

        return hook

    def save_residual_pair(prefix):
        def hook(_module, arguments, output):
            delta = output.detach()
            incoming = arguments[0].detach()
            captured[f"{prefix}_delta"] = delta
            captured[f"{prefix}_output"] = incoming + delta

        return hook

    def save_frequency_reconstruction(_module, _arguments, output):
        rebuilt = output.detach().reshape(batch, time_frames, width, frequency_bins)
        captured["block0.frequency.reconstructed"] = rebuilt.transpose(1, 2).contiguous()

    def save_time_reconstruction(_module, _arguments, output):
        rebuilt = output.detach().reshape(batch, frequency_bins, width, time_frames)
        captured["block0.time.reconstructed"] = rebuilt.permute(0, 2, 3, 1).contiguous()

    def save_time_norm(_module, arguments, output):
        captured["block0.frequency.output"] = arguments[0].detach()
        captured["block0.time.normalized"] = output.detach()

    handles.extend(
        [
            model.up_conv.register_forward_hook(save_output("encoder.output")),
            block.intra_norm.register_forward_hook(
                save_output("block0.frequency.normalized")
            ),
            block.intra_inv.register_forward_pre_hook(
                save_input("block0.frequency.unfolded")
            ),
            block.intra_inv.register_forward_hook(
                save_output("block0.frequency.gated")
            ),
            block.intra_att.register_forward_hook(
                save_residual_pair("block0.frequency.attention")
            ),
            block.intra_ffw.register_forward_hook(
                save_residual_pair("block0.frequency.mamba")
            ),
            block.intra_linear.register_forward_hook(save_frequency_reconstruction),
            block.inter_norm.register_forward_hook(save_time_norm),
            block.inter_inv.register_forward_pre_hook(
                save_input("block0.time.unfolded")
            ),
            block.inter_inv.register_forward_hook(save_output("block0.time.gated")),
            block.inter_att.register_forward_hook(
                save_residual_pair("block0.time.attention")
            ),
            block.inter_ffw.register_forward_hook(
                save_residual_pair("block0.time.mamba")
            ),
            block.inter_linear.register_forward_hook(save_time_reconstruction),
            model.down_conv.register_forward_pre_hook(save_input("block0.time.output")),
            model.down_conv.register_forward_pre_hook(save_input("output_head.input")),
            model.down_conv.register_forward_hook(
                save_output("output_head.output_channels_first")
            ),
        ]
    )
    try:
        with torch.no_grad():
            output = model(model_input)
    finally:
        for handle in handles:
            handle.remove()
    captured["output"] = output.detach()
    return output, captured


def tensor_metrics(reference, candidate) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional

    if reference.shape != candidate.shape:
        return {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "shape_equal": False,
        }
    reference_float = reference.float().reshape(-1)
    candidate_float = candidate.float().reshape(-1)
    difference = (reference_float - candidate_float).abs()
    denominator = torch.linalg.vector_norm(reference_float).clamp_min(1.0e-12)
    cosine = functional.cosine_similarity(
        reference_float.unsqueeze(0), candidate_float.unsqueeze(0), dim=1
    )
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_equal": True,
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "relative_l2_error": float(
            (torch.linalg.vector_norm(reference_float - candidate_float) / denominator).item()
        ),
        "cosine_similarity": float(cosine.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--n-mics", type=int, default=2)
    parser.add_argument("--n-srcs", type=int, default=1)
    parser.add_argument("--frequency-bins", type=int, default=12)
    parser.add_argument("--time-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from sounddiffsep.models.deftmamba.model import DeFTMambaCore

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream FlashAttention path")
    checkout = prepare_checkout(args.checkout)
    patch_status = apply_minimal_patch(checkout)
    upstream_module = load_upstream(checkout / "DeFTMamba.py")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    upstream_model = upstream_module.DeFTMamba(
        "separation",
        "unused",
        "unused",
        n_mics=args.n_mics,
        n_srcs=args.n_srcs,
        n_layers=1,
        emb_dim=64,
        hidden_dim=256,
        n_head=4,
        emb_ks=4,
        emb_hs=1,
        dropout=0.0,
    ).to(device)
    local_model = DeFTMambaCore(
        n_mics=args.n_mics,
        n_srcs=args.n_srcs,
        n_layers=1,
        emb_dim=64,
        hidden_dim=256,
        n_heads=4,
        emb_kernel_size=4,
        emb_hop_size=1,
        dropout=0.0,
    ).to(device)
    correspondence = pair_state_dicts(upstream_model, local_model)
    upstream_model.eval()
    local_model.eval()

    model_input = torch.randn(
        1,
        args.frequency_bins,
        args.time_frames,
        2 * args.n_mics,
        device=device,
        dtype=torch.float32,
    )
    upstream_output, upstream_trace = capture_upstream(upstream_model, model_input)
    with torch.no_grad():
        local_output, local_trace = local_model.forward_with_intermediates(model_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    comparison_order = list(upstream_trace)
    missing = [name for name in comparison_order if name not in local_trace]
    if missing:
        raise AssertionError(f"local trace is missing comparison points: {missing}")
    comparisons = OrderedDict(
        (name, tensor_metrics(upstream_trace[name], local_trace[name]))
        for name in comparison_order
    )
    first_mismatch = None
    for name, metrics in comparisons.items():
        if not metrics.get("shape_equal", False):
            first_mismatch = name
            break
        if not torch.allclose(
            upstream_trace[name], local_trace[name], atol=args.atol, rtol=args.rtol
        ):
            first_mismatch = name
            break

    upstream_parameters = list(upstream_model.parameters())
    local_parameters = list(local_model.parameters())
    payload = {
        "upstream_commit": UPSTREAM_COMMIT,
        "patch_status": patch_status,
        "configuration": {
            "n_mics": args.n_mics,
            "n_srcs": args.n_srcs,
            "n_layers": 1,
            "emb_dim": 64,
            "dropout": 0.0,
            "batch_size": 1,
            "dtype": "float32",
            "seed": args.seed,
        },
        "tolerance": {"absolute": args.atol, "relative": args.rtol},
        "parameter_count": {
            "upstream_minfix": sum(parameter.numel() for parameter in upstream_parameters),
            "sounddiffsep": sum(parameter.numel() for parameter in local_parameters),
        },
        "trainable_parameter_count": {
            "upstream_minfix": sum(
                parameter.numel() for parameter in upstream_parameters if parameter.requires_grad
            ),
            "sounddiffsep": sum(
                parameter.numel() for parameter in local_parameters if parameter.requires_grad
            ),
        },
        "state_dict_entries": len(correspondence),
        "state_dict_correspondence": correspondence,
        "comparisons": comparisons,
        "first_mismatch": first_mismatch,
        "all_comparisons_within_tolerance": first_mismatch is None,
        "final_output_metrics": tensor_metrics(upstream_output, local_output),
    }
    write_json(payload, args.output)


if __name__ == "__main__":
    main()
