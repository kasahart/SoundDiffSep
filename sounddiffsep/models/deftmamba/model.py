"""Spectral DeFT-Mamba core for the validated 2-microphone/1-source setup."""

from __future__ import annotations

from collections import OrderedDict

from torch import Tensor, nn

from .blocks import DeFTMambaBlock


class DeFTMambaCore(nn.Module):
    """Direct complex spectral mapping with ``[B,F,T,2M] -> [B,F,T,2S]``.

    Only ``n_mics=2`` and ``n_srcs=1`` are validated by this baseline.  The
    arguments remain explicit to make the tensor contract auditable; other
    values are not advertised as supported configurations.
    """

    def __init__(
        self,
        *,
        n_mics: int = 2,
        n_srcs: int = 1,
        n_layers: int = 4,
        emb_dim: int = 64,
        n_heads: int = 4,
        emb_kernel_size: int = 4,
        emb_hop_size: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if n_mics <= 0 or n_srcs <= 0:
            raise ValueError("n_mics and n_srcs must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.n_mics = n_mics
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.emb_dim = emb_dim

        self.input_projection = nn.Conv2d(
            2 * n_mics,
            emb_dim,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.input_normalization = nn.GroupNorm(1, emb_dim, eps=eps)
        self.feature_blocks = nn.ModuleList(
            [
                DeFTMambaBlock(
                    width=emb_dim,
                    window_size=emb_kernel_size,
                    hop_size=emb_hop_size,
                    hidden_width=hidden_dim,
                    heads=n_heads,
                    dropout=dropout,
                    epsilon=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_projection = nn.Conv2d(
            emb_dim,
            2 * n_srcs,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

    def _forward_impl(
        self, spectrum: Tensor, collect_intermediates: bool
    ) -> tuple[Tensor, OrderedDict[str, Tensor]]:
        expected_features = 2 * self.n_mics
        if spectrum.ndim != 4:
            raise ValueError(
                "DeFTMambaCore expects [batch, frequency, time, RI channels], "
                f"got {tuple(spectrum.shape)}"
            )
        if spectrum.shape[-1] != expected_features:
            raise ValueError(
                f"last dimension must be 2 * n_mics = {expected_features}, "
                f"got {spectrum.shape[-1]}"
            )

        trace: OrderedDict[str, Tensor] = OrderedDict()
        channels_first = spectrum.permute(0, 3, 2, 1).contiguous()
        encoded = self.input_normalization(self.input_projection(channels_first))
        if collect_intermediates:
            trace["encoder.output"] = encoded

        features = encoded
        for index, block in enumerate(self.feature_blocks):
            features = block(
                features,
                trace=trace if collect_intermediates else None,
                trace_prefix=f"block{index}",
            )
        projected = self.output_projection(features)
        output = projected.permute(0, 3, 2, 1).contiguous()
        if collect_intermediates:
            trace["output_head.input"] = features
            trace["output_head.output_channels_first"] = projected
            trace["output"] = output
        return output, trace

    def forward(self, spectrum: Tensor) -> Tensor:
        output, _ = self._forward_impl(spectrum, collect_intermediates=False)
        return output

    def forward_with_intermediates(
        self, spectrum: Tensor
    ) -> tuple[Tensor, OrderedDict[str, Tensor]]:
        """Return named comparison points for the GPL reference harness."""

        return self._forward_impl(spectrum, collect_intermediates=True)


def create_official_2ch_1src_core(**overrides) -> DeFTMambaCore:
    """Construct the single configuration formally validated in this PR."""

    settings = dict(
        n_mics=2,
        n_srcs=1,
        n_layers=4,
        emb_dim=64,
        n_heads=4,
        emb_kernel_size=4,
        emb_hop_size=1,
        hidden_dim=256,
        dropout=0.1,
        eps=1.0e-5,
    )
    settings.update(overrides)
    return DeFTMambaCore(**settings)
