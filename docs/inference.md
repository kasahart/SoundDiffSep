# 推論の約束（`sounddiffsep.inference`）

```python
from sounddiffsep.inference import load
sep = load("model.pt", "cuda:0")            # model.pt.preprocess.json を読む（cfg= で dict を直接渡してもよい）
s_hat, n_hat = sep.separate(close, far)     # 元のレベル、n_hat = close - s_hat
```

入力は生の 2 ch（`[T]` か `[B, T]`、numpy か torch）。前処理はすべて sidecar
`<ckpt>.preprocess.json` に書き、コード側に機種や実験ごとの分岐を置かない。

```json
{"model": "resunet30", "input_order": ["close", "far"],
 "norm": {"type": "close_peak", "peak": 0.5}}
```

| キー | 値 |
|---|---|
| `model` | `resunet30`（`{"state_dict": ...}`、`ss_model.` 接頭辞は外す）／ `deftmamba`（`checkpoint_schema: relative_mixture_training_v1`）／ `dcunet`（`config`: 学習 yaml。asteroid が要る。ここでは未検証） |
| `input_order` | `["close", "far"]`（設計どおり）か `["far", "close"]`（本番 DeFT の入れ替え） |
| `norm.type` / `norm.peak` | 下表 |
| `tf32`（任意） | `true` で TF32 の matmul/cuDNN（プロセス全体に効く）。DeFT 本番 worker の再現用 |

## 1. チャンネルの役割

- ch0 = close（機械の近く）、ch1 = far。
- 学習の約束（DeFT-Mamba relative-mixture）：目的音 = ch0 で ch1 より **6 dB 以上**大きい音源
  （背景音に対する相対値）。背景音どうしの ch0/ch1 差は **±1 dB**（spread 2 dB）以内。
- どのモデルも「最初の入力チャンネルにおける目的音の像」を出す（ResUNet30 は ch0 にマスクをかける）。
- `["close", "far"]`：`separate()` が `(s_hat, n_hat)` を返す。`n_hat = close - s_hat`。
- `["far", "close"]`（本番の入れ替え）：出力は **far マイクでの像**（far で優勢な音 = 本番では
  far 側の雑音推定 N̂_f）。これは学習の約束の外の使い方で、`n_hat` の意味も変わるため
  `separate()` は拒否し、`far_image(close, far)` で返す（残差が要るなら `far - far_image`）。

## 2. 正規化（モデルごと、sidecar に書く）

ゲインは `.separate()` に渡した入力から 1 ファイルごとに計算し、入力に掛け、出力を最初の入力
チャンネルのゲインで戻す（実装は `sounddiffsep/normalization.py`）。

| `norm.type` | ゲイン | 使うモデル |
|---|---|---|
| `independent_peak` | チャンネルごとに peak → p | DeFT-Mamba の学習（p = 0.9）、旧 SoundDiffSep モデル |
| `common_close_anchor` | close の peak → p、同じ係数を両 ch に | DeFT-Mamba 本番 worker（p = 0.9） |
| `joint_peak` | max(peak(close), peak(far)) → p | ResUNet30 初期値 sep_init_step10000（p = 0.9） |
| `close_peak` | close の peak → p、同じ係数を両 ch に | close peak で学習した微調整版（学習は close peak U(0.1, 0.9)、推論は p = 0.5） |
| `none` | 1 | — |

WLS-F などで加工した close を入れるが、ゲインは生 close から取りたい場合は、自分でスケールして
`none` を使う。

## 3. 推論時の dropout は切る

`load()` は `eval()` に加え、attention モジュールの `dropout_probability` を 0 にする。
DeFT-Mamba の attention は以前 `eval()` でも dropout が掛かっていた（FlashAttention に
`dropout_p` をそのまま渡していた）。現在は `self.training` のときだけ渡す。

## 4. 再現の確認

- ResUNet30（`close_peak 0.5`）：既存の推論スクリプトの出力と max|diff| 5.3e-7（float32 の丸め程度）。
- DeFT-Mamba（relative_mixture_training_v1、`["far", "close"]`、`common_close_anchor 0.9`）：別実装の推論 worker と
  `tf32: true` で完全一致、`tf32: false`（FP32）では max|diff| 2e-4 程度。参照側の worker は
  `torch.set_float32_matmul_precision("high")` が効いた状態で動いていたため、`tf32` キーで合わせられるようにしている。
