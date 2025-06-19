# AudioSetMixCrosstalkTwoNoiseEQ データ作成手順

## 概要

`AudioSetMixCrosstalkTwoNoiseEQ`は音声データセットクラスで、以下の特徴を持つデータを生成します：

- **ターゲット音声**: RIR（Room Impulse Response）を適用してクロストークを模擬
- **ノイズ**: 2つの異なるノイズ音声にそれぞれ異なるRIRを適用
- **EQ処理**: ノイズ音声にバンドストップフィルタによるEQ処理を適用
- **混合**: ターゲット+ノイズ、ノイズ+ターゲット（逆方向）の2つの混合音声を生成

## クラス階層

```text
AudioSetMix (基底クラス)
└── AudioSetMixTwoNoise (2つのノイズ対応)
    └── AudioSetMixCrosstalkTwoNoise (ターゲットにRIR適用)
        └── AudioSetMixCrosstalkTwoNoiseEQ (EQ処理追加)
```

## データ作成の詳細手順

### 1. 初期化パラメータ

```python
def __init__(self, dset="", sr=32000, test_snr=-5, p=1.0, segment=10):
```

- `dset`: "train" または "eval"
- `sr`: サンプリングレート（32kHz）
- `test_snr`: 評価時のSNR値
- `p`: EQ処理の適用確率
- `segment`: 音声セグメントの長さ（秒）

### 2. EQフィルタの設定

```python
self.eq = BandStopFilter(
    min_center_freq=200,
    max_center_freq=10000,
    min_bandwidth_fraction=0.25,
    max_bandwidth_fraction=1,
    zero_phase=True,
    p=self.p,
)
```

- バンドストップフィルタを使用
- 中心周波数: 200Hz〜10kHz
- 帯域幅: 25%〜100%

### 3. データ生成プロセス（`__getitem__`メソッド）

#### 3.1 音声とRIRの選択

1. **ターゲット音声の選択**

   ```python
   tgt_name = self.data_names[idx]
   ```

2. **ノイズ音声の選択（2つ）**

   ```python
   noise_name1, rir_name1 = self.get_noise_and_rir_names(idx, tgt_name, second=False)
   noise_name2, rir_name2 = self.get_noise_and_rir_names(idx, tgt_name, second=True)
   ```

3. **ターゲット用RIRの選択**
   - 訓練時: ランダム選択
   - 評価時: 固定パターン

#### 3.2 SNR設定

- **訓練時**:
  - メインSNR: -10dB〜+10dBのランダム値
  - ノイズミックスSNR: |メインSNR| + 5 + 0〜25dBのランダム値（最小5dB、最大30dB）

- **評価時**:
  - メインSNR: 固定値（test_snr）
  - ノイズミックスSNR: |test_snr| + 5dB

#### 3.3 音声データの読み込み

```python
tgt_raw = self.load_audio(tgt_name)        # ターゲット音声
noise_clean1 = self.load_audio(noise_name1) # ノイズ1
noise_clean2 = self.load_audio(noise_name2) # ノイズ2
```

#### 3.4 RIRの適用

```python
tgt_reverb = self.apply_rir_to_noise(tgt_raw, rir_tgt)      # ターゲットにRIR適用
noise1_reverb = self.apply_rir_to_noise(noise_clean1, rir1) # ノイズ1にRIR適用
noise2_reverb = self.apply_rir_to_noise(noise_clean2, rir2) # ノイズ2にRIR適用
```

#### 3.5 ノイズの合成

```python
noise_reverb = noise1_reverb + noise2_reverb  # リバーブ後のノイズ2つを合算
noise_clean = noise_clean1 + noise_clean2     # リバーブ前のノイズ2つを合算
```

#### 3.6 音声の混合

1. **ターゲット + ノイズの混合**

   ```python
   tgt_mixed = self.mix_audio(tgt_raw, noise_reverb, snr)
   ```

2. **ノイズ + ターゲットの混合**

   ```python
   noise_mixed = self.mix_audio(noise_clean, tgt_reverb, noise_mix_snr)
   ```

#### 3.7 EQ処理の適用

訓練時のみ、ノイズ混合音声にバンドストップフィルタを適用：

```python
if self.dset == "train":
    noise_mixed = torch.tensor(
        self.eq(noise_mixed.numpy(), sample_rate=self.sr)
    )
```

#### 3.8 正規化

```python
tgt_final, noise_final, mixed_final = self.normalize_audio(
    tgt_raw, noise_mixed, tgt_mixed
)
```

各音声の最大振幅を0.9に正規化

### 4. 出力データ

最終的に以下の3つの音声データが返される：

1. **mixed_final**: メイン混合音声（リバーブ前ターゲット + リバーブ後2ノイズ合成）
2. **noise_final**: ノイズ混合音声（リバーブ前2ノイズ合成 + リバーブ後ターゲット + EQ処理）
3. **tgt_final**: クリーンターゲット音声（リバーブ前）

## データパスの設定

### 必要なディレクトリ構造

```text
/workspaces/2chssDNN/data/
├── audioset/
│   ├── train.pkl  # 訓練データのメタ情報
│   └── eval.pkl   # 評価データのメタ情報
└── RVAE-EM-rirs/
    ├── train/     # 訓練用RIRファイル (.wav)
    └── test/      # 評価用RIRファイル (.wav)
```

### 音声ファイルパス

- AudioSetの音声ファイルパスは`train.pkl`/`eval.pkl`に記録
- パス内の"clapsep-only-audio"は"2chssDNN"に置換される

## 使用例

```python
# 訓練用データセット
train_dataset = AudioSetMixCrosstalkTwoNoiseEQ(
    dset="train", 
    sr=32000, 
    test_snr=-10, 
    p=1.0,  # EQ適用確率100%
    segment=3
)

# 評価用データセット
eval_dataset = AudioSetMixCrosstalkTwoNoiseEQ(
    dset="eval", 
    sr=32000, 
    test_snr=-10, 
    segment=3
)

# データの取得
mixed, noise_mixed, target = train_dataset[0]
```

## 特徴

1. **クロストーク模擬**: ターゲット音声にもRIRを適用することで、実際の録音環境でのクロストークを模擬
2. **複数ノイズ対応**: 2つの独立したノイズ源を使用
3. **EQ処理**: 実際の音響機器の特性を模擬するバンドストップフィルタ
4. **双方向混合**: ターゲット→ノイズとノイズ→ターゲットの両方向の音響結合を考慮
5. **動的SNR**: 訓練時にランダムなSNRを使用して汎化性能を向上

この手順により、実際の音響環境により近いデータセットが生成され、音源分離モデルの性能向上が期待できます。
