"""
統一モデルローダー - ResUNet30_UnconditionalとDCUNetの統一インターフェース
"""
import numpy as np
import torch
import yaml
from typing import Union
from pathlib import Path

def easy_load_model(model_type: str, device: str = "cpu") -> torch.nn.Module:
    """モデルの簡易読み込み

    Args:
        model_type: モデルタイプ ("resunet" or "DCUNet-16", "DCUNet-16-4096", "DCUNet-20", "Large-DCUNet-20")
        device: 実行デバイス ("cpu" or "cuda")

    Returns:
        torch.nn.Module: 読み込まれたモデル
    """
    # このファイルのパスを取得
    current_file_path = Path(__file__).resolve()
    project_dir = current_file_path.parent.parent

    if model_type == "resunet":
        config_path = project_dir / "ResUnet/stereo_unconditional.yaml"
        checkpoint_path = project_dir / "exp/checkpoints/ResUnet_TwoNoise_ClossTalk_add_EQ2/step=360000.ckpt"
        return load_model("resunet", config_path, checkpoint_path, device)
    elif "DCUNet" in model_type:
        config = project_dir / "DCUNet"
        checkpoint = project_dir / "exp/checkpoints"
        if model_type == "DCUNet-16":
            config_path = config / "conf_finetuning.yaml"
            checkpoint_path = checkpoint / "epoch=66-step=670000.ckpt"
        elif model_type == "DCUNet-16-4096":
            config_path = config / "conf_crosstalk_add_EQ_4096.yaml"
            checkpoint_path = checkpoint / "epoch=21-step=110000.ckpt"
        elif model_type == "DCUNet-20":
            config_path = config / "conf_finetuning_crosstalk_add_EQ_DCUNet20.yaml"
            checkpoint_path = checkpoint / "DCUNet-20_TwoNoise_CrossTalk_add_EQ2/epoch=15-step=80000.ckpt"
        elif model_type == "Large-DCUNet-20":
            config_path = config / "conf_finetuning_crosstalk_add_EQ_Large-DCUNet.yaml"
            checkpoint_path = checkpoint / "Large-DCUNet-20_TwoNoise_CrossTalk_add_EQ2/epoch=25-step=130000.ckpt"
        return load_model("DCUNet", config_path, checkpoint_path, device)
    
    raise ValueError(f"Unsupported model type: {model_type}")


def load_model(model_type: str, config_path: str, checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """モデルの読み込み
    
    Args:
        model_type: モデルタイプ ("resunet" or "dcunet")
        config_path: 設定ファイルのパス
        checkpoint_path: チェックポイントファイルのパス
        device: 実行デバイス ("cpu" or "cuda")
        
    Returns:
        torch.nn.Module: 読み込まれたモデル
        
    Raises:
        ValueError: サポートされていないモデルタイプの場合
        FileNotFoundError: ファイルが見つからない場合
    """
    model_type = model_type.lower()
    
    # モデルタイプの確認を先に行う
    if model_type not in ["resunet", "dcunet"]:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'resunet', 'dcunet'")
    
    # ファイル存在確認
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if model_type == "resunet":
        return _load_resunet(config_path, checkpoint_path, device)
    elif model_type == "dcunet":
        return _load_dcunet(config_path, checkpoint_path, device)
    else:
        # この行には到達しないはずだが、念のため
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'resunet', 'dcunet'")


def _load_resunet(config_path: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    """ResUNet30_Unconditionalの読み込み処理
    
    Args:
        config_path: 設定ファイルのパス
        checkpoint_path: チェックポイントファイルのパス
        device: 実行デバイス
        
    Returns:
        torch.nn.Module: ResUNet30_Unconditionalモデル
    """
    from sounddiffsep.models.resunet.resunet_unconditional import ResUNet30_Unconditional
    
    # 設定ファイル読み込み
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    
    # モデル初期化
    model = ResUNet30_Unconditional(
        input_channels=conf["model"]["input_channels"],
        output_channels=conf["model"]["output_channels"]
    )
    
    # チェックポイント読み込み
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    
    # state_dictの構造を確認して適切にロード
    if "state_dict" in state_dict:
        checkpoint_state_dict = state_dict["state_dict"]
    else:
        checkpoint_state_dict = state_dict
    
    # デバッグ用：最初の数個のキーを確認
    sample_keys = list(checkpoint_state_dict.keys())[:3]
    print(f"チェックポイントの最初の数個のキー: {sample_keys}")
    
    # キー名の調整（ss_model.プレフィックスを除去）
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith("ss_model."):
            new_key = key[len("ss_model."):]  # ss_model.を除去
            adjusted_state_dict[new_key] = value
        else:
            adjusted_state_dict[key] = value
    
    print(f"調整後のキー数: {len(adjusted_state_dict)}")
    model.load_state_dict(adjusted_state_dict)
    
    # デバイスに移動して評価モードに設定
    model = model.to(device)
    model.eval()
    
    return model


def _load_dcunet(config_path: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    """DCUNetの読み込み処理
    
    Args:
        config_path: 設定ファイルのパス
        checkpoint_path: チェックポイントファイルのパス
        device: 実行デバイス
        
    Returns:
        torch.nn.Module: DCUNetモデル
    """
    from sounddiffsep.models.dcunet import TwoChDCUNet
    
    # 設定ファイル読み込み
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    
    # DCUNetモデル初期化
    model = TwoChDCUNet(
        **conf["filterbank"],
        **conf["masknet"],
        sample_rate=conf["data"]["sample_rate"]
    )
    
    # チェックポイント読み込み
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    
    # state_dictの構造を確認して適切にロード
    if "state_dict" in state_dict:
        checkpoint_state_dict = state_dict["state_dict"]
    else:
        checkpoint_state_dict = state_dict
    
    # デバッグ用：最初の数個のキーを確認
    sample_keys = list(checkpoint_state_dict.keys())[:3]
    print(f"DCUNet チェックポイントの最初の数個のキー: {sample_keys}")
    
    # キー名の調整（必要に応じて）
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        # DCUNetの場合、通常はプレフィックス調整は不要だが、念のため確認
        if key.startswith("model."):
            new_key = key[len("model."):]
            adjusted_state_dict[new_key] = value
        else:
            adjusted_state_dict[key] = value
    
    print(f"DCUNet 調整後のキー数: {len(adjusted_state_dict)}")
    model.load_state_dict(adjusted_state_dict)
    
    # デバイスに移動して評価モードに設定
    model = model.to(device)
    model.eval()
    
    return model


def preprocess(mixed_signal: np.ndarray, noise_signal: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """前処理（両モデル共通）
    
    Args:
        mixed_signal: 混合信号 (channels, samples)
        noise_signal: ノイズ信号 (channels, samples)  
        device: 実行デバイス
        
    Returns:
        torch.Tensor: 前処理されたテンソル (batch, channels, samples)
    """
    # numpy配列をfloat32に変換
    mixed_signal = mixed_signal.astype(np.float32)
    noise_signal = noise_signal.astype(np.float32)
    
    # テンソルに変換してスタック
    input_tensor = torch.stack([
        torch.from_numpy(mixed_signal), 
        torch.from_numpy(noise_signal)
    ], dim=0).unsqueeze(0)  # (1, 2, channels, samples)
    
    return input_tensor.to(device)


def postprocess(output_tensor: torch.Tensor) -> np.ndarray:
    """後処理（両モデル共通）
    
    Args:
        output_tensor: モデルの出力テンソル
        
    Returns:
        np.ndarray: 後処理された音声信号
    """
    return output_tensor.squeeze().detach().cpu().numpy()


def load_config_from_file(config_file_path: str) -> dict:
    """設定ファイルから設定を読み込む
    
    Args:
        config_file_path: 設定ファイルのパス
        
    Returns:
        dict: 設定辞書
    """
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def separate_audio(mixed_signal: np.ndarray, noise_signal: np.ndarray, 
                  model: torch.nn.Module, device: str = "cpu") -> dict:
    """音源分離を実行する便利関数
    
    Args:
        mixed_signal: 混合信号
        noise_signal: ノイズ信号
        model: 分離モデル
        device: 実行デバイス
        
    Returns:
        dict: 分離結果 {"estimated_target": np.ndarray, "estimated_noise": np.ndarray}
    """
    
    mix_norm = np.max(np.abs(mixed_signal))/0.9
    noise_norm = np.max(np.abs(noise_signal))/0.9

    mix_normalized = mixed_signal / mix_norm
    noise_normalized = noise_signal / noise_norm

    # 前処理
    input_tensor = preprocess(mix_normalized, noise_normalized, device)

    # ResUNetの場合、4D -> 3Dに変換（mixturesだけを使用）
    # input_tensor shape: (1, 2, 2, samples) -> mixtures: (1, 2, samples)
    if input_tensor.shape[1] == 2 and input_tensor.shape[2] == 2:
        # 混合信号を取得: input_tensor[0, 0] = (2, samples)
        mixtures = input_tensor[0, 0]  # (2, samples)
        mixtures = mixtures.unsqueeze(0)  # (1, 2, samples)
        model_input = mixtures
    else:
        model_input = input_tensor
    
    # 推論実行
    with torch.no_grad():
        output_tensor = model.to(device)(model_input)
    
    # 後処理
    estimated_target_mono = postprocess(output_tensor)
    
    # ResUNetの出力は1チャンネルなので、入力と同じチャンネル数に変換
    if estimated_target_mono.ndim == 1:  # (samples,)
        if mixed_signal.shape[0] == 2:  # ステレオ入力の場合
            estimated_target = np.stack([estimated_target_mono, estimated_target_mono], axis=0)
        else:
            estimated_target = estimated_target_mono.reshape(1, -1)
    else:  # 既に適切な形状
        estimated_target = estimated_target_mono

    estimated_target *= mix_norm
    estimated_noise = mixed_signal - estimated_target
    
    return {
        "estimated_target": estimated_target,
        "estimated_noise": estimated_noise
    }
