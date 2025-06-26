import numpy as np
from typing import Dict, List, NoReturn, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from .resunet import ResUNet30_Base


class ResUNet30_Base_Unconditional(ResUNet30_Base):
    """
    無条件ResUNet30ベースモデル
    """
    def __init__(self, input_channels, output_channels):
        super(ResUNet30_Base_Unconditional, self).__init__(input_channels, output_channels)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real[:,:, 0].reshape(shape)
        out_imag = out_imag[:,:, 0].reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform
    

    def forward(self, mixtures):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        encoder_block1_film_dict=dict(conv_block1=dict(beta1=0,beta2=0))
        decoder_block1_film_dict=dict(beta1=0, conv_block2=dict(beta1=0,beta2=0))

        # UNet
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, encoder_block1_film_dict)  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool,encoder_block1_film_dict)  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool, encoder_block1_film_dict)  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool, encoder_block1_film_dict)  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool, encoder_block1_film_dict)  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool, encoder_block1_film_dict)  # x6_pool: (bs, 384, T / 32, F / 64)
        x_center, _ = self.conv_block7a(x6_pool, encoder_block1_film_dict)  # (bs, 384, T / 32, F / 64)
        x7 = self.decoder_block1(x_center, x6, decoder_block1_film_dict)  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, decoder_block1_film_dict)  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, decoder_block1_film_dict)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, decoder_block1_film_dict)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, decoder_block1_film_dict)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, decoder_block1_film_dict)  # (bs, 32, T, F)

        x = self.after_conv(x12)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        return separated_audio
    

class ResUNet30_Unconditional(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResUNet30_Unconditional, self).__init__()

        self.base = ResUNet30_Base_Unconditional(
            input_channels=input_channels, 
            output_channels=output_channels,
        )
        

    def forward(self, mixtures):

        separated_audio = self.base(
            mixtures=mixtures, 
        )

        return separated_audio

    @torch.no_grad()
    def chunk_inference(self, mixtures):
        chunk_config = {
                    'NL': 1.0,
                    'NC': 3.0,
                    'NR': 1.0,
                    'RATE': 32000
                }

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]
        
        out_np = np.zeros([1, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

            chunk_out = self.base(
                mixtures=chunk_in, 
            )
            
            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out = self.base(
                    mixtures=chunk_in, 
                )

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np
