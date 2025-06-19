import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# from asteroid.models import SuDORMRFImprovedNet
# from asteroid.engine.optimizers import make_optimizer
from schedulefree import RAdamScheduleFree

from torchmetrics.audio.snr import (
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
)

from sounddiffsep.data_utils.mix_dataset import AudioMixCrosstalkTwoNoiseEQ as Dataset
from sounddiffsep.system.sep_system import TwoChSepSystem
from models.dcunet import TwoChDCUNet

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
torch.set_float32_matmul_precision("high")
# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir",
    default="exp/finetuning",
    help="Full path to save best validation model",
)

def loss_fn(pred, tgt):
    loss_value = -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()
    return loss_value


def main(conf):
    print("=== ファインチューニング開始 ===")
    print(f"実験ディレクトリ: {conf['main_args']['exp_dir']}")
    
    print("\n--- データセットの初期化 ---")
    print(f"訓練用音声ディレクトリ: {conf['data']['train']['audio_dir']}")
    print(f"検証用音声ディレクトリ: {conf['data']['valid']['audio_dir']}")
    print(f"RIRディレクトリ: {conf['data']['train']['rir_dir']}")
    print(f"セグメント長: {conf['training']['segment']}")
    print(f"データ拡張確率: {conf['training']['augmentation_p']}")
    
    train_set = Dataset(
        audio_dir=conf["data"]["train"]["audio_dir"],
        rir_dir=conf["data"]["train"]["rir_dir"],
        dset="train",
        p=conf["training"]["augmentation_p"],
        segment=conf["training"]["segment"])
    print(f"訓練データセットサイズ: {len(train_set)}")
    
    val_set = Dataset(
        audio_dir=conf["data"]["valid"]["audio_dir"],
        rir_dir=conf["data"]["valid"]["rir_dir"],
        dset="eval",
        segment=conf["training"]["segment"])
    print(f"検証データセットサイズ: {len(val_set)}")
    
    print("\n--- データローダーの作成 ---")
    print(f"バッチサイズ: {conf['training']['batch_size']}")
    print(f"ワーカー数: {conf['training']['num_workers']}")
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    print(f"訓練バッチ数: {len(train_loader)}")
    print(f"検証バッチ数: {len(val_loader)}")

    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    print("\n--- モデルの初期化 ---")
    print(f"サンプルレート: {conf['data']['sample_rate']}")
    print(f"音源数: {conf['data']['n_src']}")
    print(f"フィルターバンク設定: {conf['filterbank']}")
    print(f"マスクネット設定: {conf['masknet']}")
    
    model = TwoChDCUNet(
        **conf["filterbank"],
        **conf["masknet"],
        sample_rate=conf["data"]["sample_rate"],
    )
    print(f"モデル初期化完了: {type(model).__name__}")

    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n--- 設定ファイルの保存 ---")
    conf_path = os.path.join(exp_dir, "conf.yml")
    print(f"設定ファイル保存先: {conf_path}")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)
    print("設定ファイル保存完了")

    # Define Loss function.
    print("\n--- システムの初期化 ---")
    print(f"学習率: {conf['optim']['lr']}")
    print(f"ベースモデルチェックポイント: {conf['base']['ckpt']}")
    
    system = TwoChSepSystem(
        model=model,
        loss_func=loss_fn,
        optimizer=None,  # TwoChSSSystemのconfigure_optimizersで定義
        train_loader=train_loader,
        val_loader=val_loader,
        config=conf,
        lr=conf["optim"]["lr"],
    )
    
    print("ベースモデルの重みを読み込み中...")
    state_dict = torch.load(conf["base"]["ckpt"], weights_only=True)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    print("ベースモデルの重み読み込み完了")

    # Define callbacks
    print("\n--- コールバックの設定 ---")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    print(f"チェックポイント保存先: {checkpoint_dir}")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    print("ModelCheckpointコールバック追加")
    
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)
        )
        print("EarlyStoppingコールバック追加")
    else:
        print("EarlyStopping無効")

    print("\n--- トレーナーの初期化 ---")
    print(f"最大エポック数: {conf['training']['epochs']}")
    print(f"勾配クリッピング: {conf['training']['gradient_clipping']}")
    print(f"GPU利用可能: {torch.cuda.is_available()}")
    print(f"訓練バッチ制限: {conf['training']['limit_train_batches']}")
    
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        gradient_clip_val=conf["training"]["gradient_clipping"],
        benchmark=True,
        limit_train_batches=conf["training"][
            "limit_train_batches"
        ],  # Useful for fast experiment
        # limit_val_batches=10,  # Useful for fast experiment
    )
    print("トレーナー初期化完了")

    #チューニング前のval_lossを確認
    print("\n--- ファインチューニング前の検証 ---")
    pre_val_results = trainer.validate(system, val_loader)
    print(f"ファインチューニング前の検証損失: {pre_val_results[0]['val_loss']:.4f}")
    
    # トレーニング
    print("\n--- ファインチューニング開始 ---")
    trainer.fit(system)
    print("ファインチューニング完了")

    print("\n--- ベストモデルの保存 ---")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    best_k_path = os.path.join(exp_dir, "best_k_models.json")
    print(f"ベストkモデル情報保存先: {best_k_path}")
    with open(best_k_path, "w") as f:
        json.dump(best_k, f, indent=0)
    print(f"ベストモデル一覧: {best_k}")

    print(f"ベストモデルパス: {checkpoint.best_model_path}")
    state_dict = torch.load(checkpoint.best_model_path, weights_only=True)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()
    print("ベストモデルをCPUに移動")

    to_save = system.model.serialize()
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    torch.save(to_save, best_model_path)
    print(f"ベストモデル保存完了: {best_model_path}")
    
    print("\n=== ファインチューニング処理完了 ===")


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    print("=== ファインチューニングスクリプト開始 ===")
    config_file = "DCUNet/conf_finetuning_crosstalk_add_EQ_Large-DCUNet.yml"
    print(f"設定ファイル読み込み: {config_file}")
    
    with open(config_file) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    
    print("\n--- 設定内容 ---")
    pprint(arg_dic)

    print(f"\n--- 乱数シード設定 ---")
    seed = 114514
    print(f"シード値: {seed}")
    pl.seed_everything(seed)
    
    main(arg_dic)
