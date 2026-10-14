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
| `model` | `resunet30`（`{"state_dict": ...}`、`ss_model.` 接頭辞は外す）／ `dcunet`（`config`: 学習 yaml。asteroid が要る。ここでは未検証）／ `register_model(name, builder)` で登録した名前 |
| `input_order` | `["close", "far"]`（設計どおり）か `["far", "close"]`（入れ替え） |
| `norm.type` / `norm.peak` | 下表 |
| `tf32`（任意） | `true` で TF32 の matmul/cuDNN（プロセス全体に効く）。TF32 で動く既存パイプラインの再現用 |

## 1. チャンネルの役割

- ch0 = close（機械の近く）、ch1 = far。
- どのモデルも「最初の入力チャンネルにおける目的音の像」を出す（ResUNet30 は ch0 にマスクをかける）。
- `["close", "far"]`：`separate()` が `(s_hat, n_hat)` を返す。`n_hat = close - s_hat`。
- `["far", "close"]`（入れ替え）：出力は **far マイクでの像**（far で優勢な音、たとえば
  far 側の雑音推定 N̂_f）。これは学習の約束の外の使い方で、`n_hat` の意味も変わるため
  `separate()` は拒否し、`far_image(close, far)` で返す（残差が要るなら `far - far_image`）。

## 2. 正規化（モデルごと、sidecar に書く）

ゲインは `.separate()` に渡した入力から 1 ファイルごとに計算し、入力に掛け、出力を最初の入力
チャンネルのゲインで戻す（実装は `sounddiffsep/normalization.py`）。

| `norm.type` | ゲイン | 使うモデル |
|---|---|---|
| `independent_peak` | チャンネルごとに peak → p | チャンネルごとの peak で学習したモデル（p = 0.9）、旧 SoundDiffSep モデル |
| `common_close_anchor` | close の peak → p、同じ係数を両 ch に | close を基準にした推論（p = 0.9） |
| `joint_peak` | max(peak(close), peak(far)) → p | ResUNet30 の初期 checkpoint（p = 0.9） |
| `close_peak` | close の peak → p、同じ係数を両 ch に | close peak で学習した微調整版（学習は close peak U(0.1, 0.9)、推論は p = 0.5） |
| `none` | 1 | — |

前処理で加工した close を入れるが、ゲインは生 close から取りたい場合は、自分でスケールして
`none` を使う。

## 3. 推論時の dropout は切る

`load()` は `eval()` に加え、`dropout_probability` を持つモジュール（`eval()` でも dropout を
掛ける attention 実装がある）の値を 0 にする。

## 4. 再現の確認

- ResUNet30（`close_peak 0.5`）：既存の推論スクリプトの出力と max|diff| 5.3e-7（float32 の丸め程度）。
