"""Independent MIT implementation of the public DeFT-Mamba block structure.

Optional CUDA extensions are imported only when a block is instantiated.  This
keeps the normal :mod:`sounddiffsep` import path independent of Mamba and
FlashAttention.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, MutableMapping

import torch
from torch import Tensor, nn
import torch.nn.functional as torch_functional


_INSTALL_HINT = "pip install -e '.[deftmamba,dev]'"


def _load_flash_attention() -> Callable[..., Tensor]:
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The upstream-faithful DeFT-Mamba path requires FlashAttention. "
            f"Install the optional dependencies with: {_INSTALL_HINT}"
        ) from exc
    return flash_attn_func


def _load_mamba_runtime():
    try:
        from mamba_ssm.models.mixer_seq_simple import _init_weights
        from mamba_ssm.modules.block import Block
        from mamba_ssm.modules.mamba2_simple import Mamba2Simple
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The upstream-faithful DeFT-Mamba path requires mamba-ssm and its "
            f"CUDA extensions. Install them with: {_INSTALL_HINT}"
        ) from exc
    class FullSequenceMamba2Simple(Mamba2Simple):
        """Bridge the standard Block API to Mamba2Simple's full-sequence API."""

        def forward(self, sequence, inference_params=None, **kwargs):
            if inference_params is not None:
                raise ValueError("DeFT-Mamba does not expose cached Mamba inference")
            return super().forward(sequence, **kwargs)

    return Block, FullSequenceMamba2Simple, RMSNorm, _init_weights


class ChannelwiseLayerNorm2d(nn.Module):
    """Normalize each time-frequency cell across its feature channels."""

    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, width, 1, 1, dtype=torch.float32))
        self.offset = nn.Parameter(torch.zeros(1, width, 1, 1, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(
                "ChannelwiseLayerNorm2d expects [batch, channels, time, frequency], "
                f"got {tuple(features.shape)}"
            )
        mean = features.mean(dim=1, keepdim=True)
        variance = features.var(dim=1, unbiased=False, keepdim=True)
        standardized = (features - mean) / torch.sqrt(variance + self.epsilon)
        return standardized * self.scale + self.offset


class GatedContextProjection(nn.Module):
    """Reduce unfolded ``window * width`` channels back to ``width``."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.content = nn.Conv1d(input_channels, output_channels, 3, padding=1)
        self.control = nn.Conv1d(input_channels, output_channels, 3, padding=1)
        self.gate_activation = nn.Mish()

    def forward(self, windows: Tensor) -> Tensor:
        return self.content(windows) * self.gate_activation(self.control(windows))


class UpstreamFlashSelfAttention(nn.Module):
    """Public-code attention path, including its unconditional BF16 cast."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.dropout_probability = dropout
        expanded_width = width * heads

        # Registration order intentionally mirrors the public tensor layout so
        # reference weights can be paired by shape without sharing source code.
        self.query = nn.Linear(width, expanded_width, bias=False)
        self.value = nn.Linear(width, expanded_width, bias=False)
        self.key = nn.Linear(width, expanded_width, bias=False)
        if heads == 1:
            self.output = nn.Identity()
        else:
            self.output = nn.Sequential(
                nn.Linear(expanded_width, width),
                nn.Dropout(dropout),
            )
        self._attention_kernel = _load_flash_attention()

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"attention expects [batch, steps, width], got {tokens.shape}")
        if not tokens.is_cuda:
            raise RuntimeError(
                "The upstream-faithful attention path requires a CUDA tensor; "
                "no CPU attention replacement is used."
            )
        batch, steps, width = tokens.shape
        if width != self.width:
            raise ValueError(f"attention width must be {self.width}, got {width}")

        target_shape = (batch, steps, self.heads, self.width)
        query = self.query(tokens).reshape(target_shape).to(torch.bfloat16)
        key = self.key(tokens).reshape(target_shape).to(torch.bfloat16)
        value = self.value(tokens).reshape(target_shape).to(torch.bfloat16)
        attended = self._attention_kernel(
            query,
            key,
            value,
            # FlashAttention applies dropout_p regardless of module mode, so
            # gate it on self.training (the released code did not; see
            # docs/research/deftmamba_upstream_audit.md).
            dropout_p=self.dropout_probability if self.training else 0.0,
        )
        flattened = attended.reshape(batch, steps, self.heads * self.width)
        return self.output(flattened.to(torch.float32))


class BidirectionalMamba2(nn.Module):
    """One forward and one reversed Mamba2 pass, as exposed upstream."""

    def __init__(self, width: int) -> None:
        super().__init__()
        Block, Mamba2Simple, RMSNorm, initialize = _load_mamba_runtime()

        def make_block(layer_index: int):
            return Block(
                width,
                mixer_cls=partial(
                    Mamba2Simple,
                    layer_idx=layer_index,
                    d_state=16,
                    d_conv=4,
                    expand=4,
                    headdim=16,
                ),
                norm_cls=partial(RMSNorm, eps=1e-5),
                mlp_cls=nn.Identity,
                fused_add_norm=False,
            )

        self.forward_path = nn.ModuleList([make_block(0)])
        self.reverse_path = nn.ModuleList([make_block(0)])
        self.apply(partial(initialize, n_layer=1))
        self.fusion = nn.Linear(2 * width, 2 * width)

    @staticmethod
    def _scan(blocks: nn.ModuleList, sequence: Tensor) -> Tensor:
        carried_residual = None
        hidden = sequence
        for block in blocks:
            hidden, carried_residual = block(hidden, carried_residual)
        if carried_residual is not None:
            hidden = hidden + carried_residual
        return hidden

    def forward(self, tokens: Tensor) -> Tensor:
        forward_result = self._scan(self.forward_path, tokens.clone())
        reversed_result = self._scan(self.reverse_path, torch.flip(tokens, dims=(1,)))
        reversed_result = torch.flip(reversed_result, dims=(1,))
        return self.fusion(torch.cat((forward_result, reversed_result), dim=-1))


class MambaFeedForward(nn.Module):
    """Bidirectional Mamba2 followed by the public-code projection."""

    def __init__(self, width: int, hidden_width: int, dropout: float) -> None:
        super().__init__()
        expected_input = 2 * width
        if hidden_width // 2 != expected_input:
            raise ValueError(
                "The released projection requires hidden_dim // 2 == 2 * emb_dim; "
                f"got hidden_dim={hidden_width}, emb_dim={width}."
            )
        self.bidirectional_scan = BidirectionalMamba2(width)
        self.activation = nn.Mish(inplace=True)
        self.pre_projection_dropout = nn.Dropout(dropout)
        self.reduction = nn.Linear(hidden_width // 2, width)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.bidirectional_scan(tokens)
        tokens = self.activation(tokens)
        tokens = self.pre_projection_dropout(tokens)
        tokens = self.reduction(tokens)
        return self.output_dropout(tokens)


class DirectionalHybridMamba(nn.Module):
    """Frequency- or time-axis GCB, attention, Mamba2, and reconstruction."""

    def __init__(
        self,
        *,
        axis: str,
        width: int,
        window_size: int,
        hop_size: int,
        hidden_width: int,
        heads: int,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        if axis not in {"frequency", "time"}:
            raise ValueError(f"unsupported axis: {axis}")
        self.axis = axis
        self.width = width
        self.window_size = window_size
        self.hop_size = hop_size

        unfolded_channels = width * window_size
        self.normalization = ChannelwiseLayerNorm2d(width, epsilon)
        self.local_projection = GatedContextProjection(unfolded_channels, width)
        self.attention_norm = nn.LayerNorm(width)
        self.attention = UpstreamFlashSelfAttention(width, heads, dropout)
        self.mamba_norm = nn.LayerNorm(width)
        self.mamba_projection = MambaFeedForward(width, hidden_width, dropout)
        self.reconstruction = nn.ConvTranspose1d(
            width,
            width,
            window_size,
            stride=hop_size,
        )

    def forward(
        self,
        features: Tensor,
        trace: MutableMapping[str, Tensor] | None = None,
        trace_prefix: str = "",
    ) -> Tensor:
        batch, channels, time_steps, frequency_bins = features.shape
        if channels != self.width:
            raise ValueError(
                f"{self.axis} block expected {self.width} channels, got {channels}"
            )

        normalized = self.normalization(features)
        if self.axis == "frequency":
            sequences = normalized.transpose(1, 2).contiguous().reshape(
                batch * time_steps, channels, frequency_bins
            )
            axis_length = frequency_bins
            parallel_items = time_steps
        else:
            sequences = normalized.permute(0, 3, 1, 2).contiguous().reshape(
                batch * frequency_bins, channels, time_steps
            )
            axis_length = time_steps
            parallel_items = frequency_bins

        unfolded = torch_functional.unfold(
            sequences[..., None],
            (self.window_size, 1),
            stride=(self.hop_size, 1),
        )
        expected_unfolded_channels = channels * self.window_size
        if unfolded.shape[1] != expected_unfolded_channels:
            raise RuntimeError(
                f"{self.axis} unfold produced {unfolded.shape[1]} channels; "
                f"expected emb_dim * emb_kernel_size = {expected_unfolded_channels}"
            )

        local_features = self.local_projection(unfolded)
        if local_features.shape[1] != channels:
            raise RuntimeError(
                f"{self.axis} gated projection must reduce to {channels} channels, "
                f"got {local_features.shape[1]}"
            )
        tokens = local_features.transpose(1, 2)
        attention_delta = self.attention(self.attention_norm(tokens))
        attention_output = tokens + attention_delta
        mamba_delta = self.mamba_projection(self.mamba_norm(attention_output))
        mamba_output = attention_output + mamba_delta
        reconstructed = self.reconstruction(mamba_output.transpose(1, 2))

        if reconstructed.shape[-1] != axis_length:
            raise RuntimeError(
                f"{self.axis} ConvTranspose1d reconstructed length "
                f"{reconstructed.shape[-1]}, expected {axis_length}"
            )
        if self.axis == "frequency":
            rebuilt = reconstructed.reshape(
                batch, parallel_items, channels, axis_length
            ).transpose(1, 2).contiguous()
        else:
            rebuilt = reconstructed.reshape(
                batch, parallel_items, channels, axis_length
            ).permute(0, 2, 3, 1).contiguous()
        output = rebuilt + features

        if trace is not None:
            trace[f"{trace_prefix}.normalized"] = normalized
            trace[f"{trace_prefix}.unfolded"] = unfolded
            trace[f"{trace_prefix}.gated"] = local_features
            trace[f"{trace_prefix}.attention_delta"] = attention_delta
            trace[f"{trace_prefix}.attention_output"] = attention_output
            trace[f"{trace_prefix}.mamba_delta"] = mamba_delta
            trace[f"{trace_prefix}.mamba_output"] = mamba_output
            trace[f"{trace_prefix}.reconstructed"] = rebuilt
            trace[f"{trace_prefix}.output"] = output
        return output


class DeFTMambaBlock(nn.Module):
    """Frequency processing followed by time processing."""

    def __init__(
        self,
        *,
        width: int,
        window_size: int,
        hop_size: int,
        hidden_width: int,
        heads: int,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        common = dict(
            width=width,
            window_size=window_size,
            hop_size=hop_size,
            hidden_width=hidden_width,
            heads=heads,
            dropout=dropout,
            epsilon=epsilon,
        )
        self.frequency_path = DirectionalHybridMamba(axis="frequency", **common)
        self.time_path = DirectionalHybridMamba(axis="time", **common)
        self.window_size = window_size
        self.hop_size = hop_size

    def _padded_length(self, length: int) -> int:
        if length < self.window_size:
            raise ValueError(
                f"axis length {length} must be >= emb_kernel_size {self.window_size}"
            )
        return (
            math.ceil((length - self.window_size) / self.hop_size) * self.hop_size
            + self.window_size
        )

    def forward(
        self,
        features: Tensor,
        trace: MutableMapping[str, Tensor] | None = None,
        trace_prefix: str = "block",
    ) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"DeFT-Mamba block expects four dimensions, got {features.shape}")
        original_time = features.shape[2]
        original_frequency = features.shape[3]
        padded_time = self._padded_length(original_time)
        padded_frequency = self._padded_length(original_frequency)
        features = torch_functional.pad(
            features,
            (0, padded_frequency - original_frequency, 0, padded_time - original_time),
        )
        features = self.frequency_path(
            features,
            trace=trace,
            trace_prefix=f"{trace_prefix}.frequency",
        )
        features = self.time_path(
            features,
            trace=trace,
            trace_prefix=f"{trace_prefix}.time",
        )
        # emb_hop_size > 1 can pad either axis; return the input grid.
        return features[:, :, :original_time, :original_frequency]
