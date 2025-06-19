from asteroid.masknn.convolutional import BaseDCUMaskNet
from asteroid import complex_nn
from asteroid.models.dcunet import BaseDCUNet
from asteroid.masknn._dcunet_architectures import make_unet_encoder_decoder_args
from asteroid.utils.torch_utils import script_if_tracing, pad_x_to_y
from asteroid_filterbanks.transforms import from_torch_complex

import torch
from torch import nn
import numpy as np
from typing import Optional


class TwoChDCUMaskNet(BaseDCUMaskNet):
    _architectures = {
        "DCUNet-16": make_unet_encoder_decoder_args(
            # Encoders:
            # (in_chan, out_chan, kernel_size, stride, padding)
            (
                (2, 32, (7, 5), (2, 2), "auto"),
                (32, 32, (7, 5), (2, 1), "auto"),
                (32, 64, (7, 5), (2, 2), "auto"),
                (64, 64, (5, 3), (2, 1), "auto"),
                (64, 64, (5, 3), (2, 2), "auto"),
                (64, 64, (5, 3), (2, 1), "auto"),
                (64, 64, (5, 3), (2, 2), "auto"),
                (64, 64, (5, 3), (2, 1), "auto"),
            ),
            (
                (64, 64, (5, 3), (2, 1), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 2), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 1), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 2), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 1), "auto", (0, 0)),
                (128, 64, (7, 5), (2, 2), "auto", (0, 0)),
                (96, 64, (7, 5), (2, 1), "auto", (0, 0)),
                (96, 1, (7, 5), (2, 2), "auto", (0, 0)),
            ),
        ),
        "DCUNet-20": make_unet_encoder_decoder_args(
            # Encoders:
            # (in_chan, out_chan, kernel_size, stride, padding)
            (
                (2, 32, (7, 1), (1, 1), "auto"),
                (32, 32, (1, 7), (1, 1), "auto"),
                (32, 64, (7, 5), (2, 2), "auto"),
                (64, 64, (7, 5), (2, 1), "auto"),
                (64, 64, (5, 3), (2, 2), "auto"),
                (64, 64, (5, 3), (2, 1), "auto"),
                (64, 64, (5, 3), (2, 2), "auto"),
                (64, 64, (5, 3), (2, 1), "auto"),
                (64, 64, (5, 3), (2, 2), "auto"),
                (64, 90, (5, 3), (2, 1), "auto"),
            ),
            (
                (90, 64, (5, 3), (2, 1), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 2), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 1), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 2), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 1), "auto", (0, 0)),
                (128, 64, (5, 3), (2, 2), "auto", (0, 0)),
                (128, 64, (7, 5), (2, 1), "auto", (0, 0)),
                (128, 64, (7, 5), (2, 2), "auto", (0, 0)),
                (96, 64, (1, 7), (1, 1), "auto", (0, 0)),
                (96, 1, (7, 1), (1, 1), "auto", (0, 0)),
            ),
        ),
        "Large-DCUNet-20": make_unet_encoder_decoder_args(
            # Encoders:
            # (in_chan, out_chan, kernel_size, stride, padding)
            (
                (2, 45, (7, 1), (1, 1), "auto"),
                (45, 45, (1, 7), (1, 1), "auto"),
                (45, 90, (7, 5), (2, 2), "auto"),
                (90, 90, (7, 5), (2, 1), "auto"),
                (90, 90, (5, 3), (2, 2), "auto"),
                (90, 90, (5, 3), (2, 1), "auto"),
                (90, 90, (5, 3), (2, 2), "auto"),
                (90, 90, (5, 3), (2, 1), "auto"),
                (90, 90, (5, 3), (2, 2), "auto"),
                (90, 128, (5, 3), (2, 1), "auto"),
            ),
            # Decoders:
            # (in_chan, out_chan, kernel_size, stride, padding, output_padding)
            (
                (128, 90, (5, 3), (2, 1), "auto", (0, 0)),
                (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
                (180, 90, (5, 3), (2, 1), "auto", (0, 0)),
                (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
                (180, 90, (5, 3), (2, 1), "auto", (0, 0)),
                (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
                (180, 90, (7, 5), (2, 1), "auto", (0, 0)),
                (180, 90, (7, 5), (2, 2), "auto", (0, 0)),
                (135, 90, (1, 7), (1, 1), "auto", (0, 0)),
                (135, 1, (7, 1), (1, 1), "auto", (0, 0)),
            ),
        ),
    }

    def __init__(self, encoders, decoders, fix_length_mode=None, **kwargs):
        self.fix_length_mode = fix_length_mode
        self.encoders_stride_product = np.prod(
            [enc_stride for _, _, _, enc_stride, _ in encoders], axis=0
        )

        # Avoid circual import
        from asteroid.masknn.convolutional import (
            DCUNetComplexDecoderBlock,
            DCUNetComplexEncoderBlock,
        )

        super().__init__(
            encoders=[DCUNetComplexEncoderBlock(*args) for args in encoders],
            decoders=[DCUNetComplexDecoderBlock(*args) for args in decoders[:-1]],
            output_layer=complex_nn.ComplexConvTranspose2d(*decoders[-1]),
            **kwargs,
        )

    def forward(self, x):
        fixed_x = self.fix_input_dims(x)
        out = super(BaseDCUMaskNet, self).forward(fixed_x)
        out = self.fix_output_dims(out, x)
        return out

    def fix_input_dims(self, x):
        return _fix_dcu_input_dims(
            self.fix_length_mode, x, torch.from_numpy(self.encoders_stride_product)
        )

    def fix_output_dims(self, out, x):
        return _fix_dcu_output_dims(self.fix_length_mode, out, x)


@script_if_tracing
def _fix_dcu_input_dims(fix_length_mode: Optional[str], x, encoders_stride_product):
    """Pad or trim `x` to a length compatible with DCUNet."""
    freq_prod = int(encoders_stride_product[0])
    time_prod = int(encoders_stride_product[1])
    if (x.shape[2] - 1) % freq_prod:
        raise TypeError(
            f"Input shape must be [batch, ch, freq + 1, time + 1] with freq divisible by "
            f"{freq_prod}, got {x.shape} instead"
        )
    time_remainder = (x.shape[3] - 1) % time_prod
    if time_remainder:
        if fix_length_mode is None:
            raise TypeError(
                f"Input shape must be [batch, ch, freq + 1, time + 1] with time divisible by "
                f"{time_prod}, got {x.shape} instead. Set the 'fix_length_mode' argument "
                f"in 'DCUNet' to 'pad' or 'trim' to fix shapes automatically."
            )
        elif fix_length_mode == "pad":
            pad_shape = [0, time_prod - time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        elif fix_length_mode == "trim":
            pad_shape = [0, -time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        else:
            raise ValueError(f"Unknown fix_length mode '{fix_length_mode}'")
    return x


@script_if_tracing
def _fix_dcu_output_dims(fix_length_mode: Optional[str], out, x):
    """Fix shape of `out` to the original shape of `x`."""
    return pad_x_to_y(out, x)


class TwoChDCUNet(BaseDCUNet):
    """DCUNet as proposed in [1].

    Args:
        architecture (str): The architecture to use, any of
            "DCUNet-10", "DCUNet-16", "DCUNet-20", "Large-DCUNet-20".
        stft_n_filters (int) Number of filters for the STFT.
        stft_kernel_size (int): STFT frame length to use.
        stft_stride (int, optional): STFT hop length to use.
        sample_rate (float): Sampling rate of the model.
        masknet_kwargs (optional): Passed to :class:`DCUMaskNet`

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
          Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    masknet_class = TwoChDCUMaskNet

    def apply_masks(self, tf_rep, est_masks):
        masked_tf_rep = est_masks * tf_rep[:, 0, None]
        return from_torch_complex(masked_tf_rep)
