from typing import Any, Callable, Dict
import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class AudioSep_Unconditional(pl.LightningModule):
    """
    無条件音源分離用のAudioSepモデル
    テキスト条件付けを完全に除去したバージョン
    """

    def __init__(
        self,
        ss_model: nn.Module = None,
        loss_function=None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func=None,
    ):
        r"""Pytorch Lightning wrapper for unconditional audio separation.

        Args:
            ss_model: nn.Module, separation model
            loss_function: function or object
            learning_rate: float
            lr_lambda_func: function for learning rate scheduling
        """

        super().__init__()
        self.ss_model = ss_model
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

    def forward(self, x):
        """Forward pass for inference"""
        # 軸の入れ替え
        # x = x.permute(0, 2, 1)  # [batch_size, channels, samples] -> [batch_size, samples, channels]
        y = self.ss_model(x)

        return y  # .permute(0, 2, 1)  # [batch_size, samples, channels] -> [batch_size, channels, samples]

    def common_step(self, batch, batch_nb, train=True):
        mixed, noise_clean, tgt = batch
        _mixed = torch.stack([mixed, noise_clean], dim=1)
        est_targets = self(_mixed)
        loss = self.loss_function(est_targets, tgt.unsqueeze(1))
        return loss

    def training_step(self, batch, batch_idx):
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        loss = self.common_step(batch, batch_idx, train=True)
        # loss = loss.mean()

        # ログ出力
        self.log_dict({"loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # # 訓練ステップと同様の処理
        # random.seed(batch_idx)

        self.ss_model.eval()
        with torch.no_grad():
            loss = self.common_step(batch, batch_idx, train=False)
            # loss = loss.mean()

        self.log_dict({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        r"""Configure optimizer."""

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError(
                f"Optimizer type '{self.optimizer_type}' is not implemented"
            )

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

        return output_dict

    def separate_audio(self, mixture_waveform):
        """
        音声分離の推論用メソッド

        Args:
            mixture_waveform: 混合音声 [channels, samples] or [batch_size, channels, samples]

        Returns:
            separated_waveform: 分離された音声
        """
        self.eval()
        with torch.no_grad():
            if mixture_waveform.dim() == 2:
                mixture_waveform = mixture_waveform.unsqueeze(0)  # バッチ次元を追加
            input_dict = {"mixture": mixture_waveform}
            output_dict = self.forward(input_dict)
            separated = output_dict["waveform"]
            if separated.size(0) == 1:
                separated = separated.squeeze(0)  # バッチ次元を除去
            return separated


def get_model_class(model_type):
    """モデルクラスの取得"""
    if model_type == "ResUNet30_Unconditional":
        from .resunet_unconditional import ResUNet30_Unconditional

        return ResUNet30_Unconditional
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented")
