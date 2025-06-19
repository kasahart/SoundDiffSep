# AudioMixCrosstalkTwoNoiseEQ データ作成手順

## 概要

`AudioMixCrosstalkTwoNoiseEQ`は音声データセットクラスで、以下の特徴を持つデータを生成します：

- **ターゲット音声**: RIR（Room Impulse Response）を適用してクロストークを模擬
- **ノイズ**: 2つの異なるノイズ音声にそれぞれ異なるRIRを適用
- **EQ処理**: ノイズ音声にバンドストップフィルタによるEQ処理を適用
- **混合**: ターゲット+ノイズ、ノイズ+ターゲット（逆方向）の2つの混合音声を生成

## クラス階層

```text
torch.utils.data.Dataset
└── AudioMixCrosstalkTwoNoiseEQ (独立実装)
```

## データ作成の詳細手順

### 1. 初期化パラメータ

```python
def __init__(self, audio_dir, rir_dir, dset="train", sr=32000, test_snr=-5, p=1.0, segment=10):
```

- `audio_dir`: 音声ファイルが格納されているディレクトリパス
- `rir_dir`: RIRファイルが格納されているディレクトリパス
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
   target_name = self.audio_names[idx]
   ```

2. **ノイズ音声の選択（2つ）**

   ```python
   # 訓練時はランダム選択
   noise_name1 = self._choose_different_noise(target_name)
   noise_name2 = self._choose_different_noise(target_name)
   
   # 評価時は固定設定を使用
   noise_name1 = self.noise_names_1[idx]
   noise_name2 = self.noise_names_2[idx]
   ```

3. **RIRの選択**
   ```python
   # 訓練時: ランダム選択
   rir_name1 = random.choice(self.rir_names)
   rir_name2 = random.choice(self.rir_names)
   rir_name_tgt = random.choice(self.rir_names)
   
   # 評価時: 固定パターン
   rir_name1 = self.rir_names_1[idx]
   rir_name2 = self.rir_names_2[idx]
   rir_name_tgt = self.rir_names_tgt[idx]
   ```

#### 3.2 SNR設定

- **訓練時**:
  - メインSNR: -10dB〜+10dBのランダム値
  - ノイズミックスSNR: |メインSNR| + 5 + 0〜25dBのランダム値（最小5dB、最大30dB）

- **評価時**:
  - メインSNR: 固定値（test_snr）
  - ノイズミックスSNR: |test_snr| + 5dB

#### 3.3 音声データの読み込み

```python
target_raw = self.load_audio(target_name)        # ターゲット音声
noise_clean1 = self.load_audio(noise_name1)     # ノイズ1
noise_clean2 = self.load_audio(noise_name2)     # ノイズ2
```

#### 3.4 RIRの適用

```python
target_reverb = self.apply_rir(target_raw, rir_target)  # ターゲットにRIR適用
noise1_reverb = self.apply_rir(noise_clean1, rir1)     # ノイズ1にRIR適用
noise2_reverb = self.apply_rir(noise_clean2, rir2)     # ノイズ2にRIR適用
```

#### 3.5 ノイズの合成

```python
noise_reverb_combined = noise1_reverb + noise2_reverb  # リバーブ後のノイズ2つを合算
noise_clean_combined = noise_clean1 + noise_clean2     # リバーブ前のノイズ2つを合算
```

#### 3.6 音声の混合

1. **ターゲット + ノイズの混合**

   ```python
   target_mixed = self.mix_audio(target_raw, noise_reverb_combined, main_snr)
   ```

2. **ノイズ + ターゲットの混合**

   ```python
   noise_mixed = self.mix_audio(noise_clean_combined, target_reverb, noise_mix_snr)
   ```

#### 3.7 EQ処理の適用

訓練時のみ、ノイズ混合音声にバンドストップフィルタを適用：

```python
if self.dset == "train" and self.eq is not None:
    noise_mixed = torch.tensor(
        self.eq(noise_mixed.numpy(), sample_rate=self.sr)
    )
```

#### 3.8 正規化

```python
target_final, noise_final, mixed_final = self.normalize_audio(
    target_raw, noise_mixed, target_mixed
)
```

各音声の最大振幅を0.9に正規化

### 4. 出力データ

最終的に以下の3つの音声データが返される：

1. **mixed_final**: メイン混合音声（リバーブ前ターゲット + リバーブ後2ノイズ合成）
2. **noise_final**: ノイズ混合音声（リバーブ前2ノイズ合成 + リバーブ後ターゲット + EQ処理）
3. **target_final**: クリーンターゲット音声（リバーブ前）

## データパスの設定

### 必要なディレクトリ構造

```text
/workspaces/SoundDiffSep/data/
├── audio/         # 音声ファイル (.wav, .mp3, .flac)
└── rir/           # RIRファイル (.wav)
```

### 音声ファイル

- 音声ファイルは`audio_dir`で指定されたディレクトリから再帰的に検索
- 対応フォーマット: `.wav`, `.mp3`, `.flac`

### RIRファイル

- RIRファイルは`rir_dir`で指定されたディレクトリから再帰的に検索
- 対応フォーマット: `.wav`

## 使用例

```python
# 訓練用データセット
train_dataset = AudioMixCrosstalkTwoNoiseEQ(
    audio_dir="/workspaces/SoundDiffSep/data/audio",
    rir_dir="/workspaces/SoundDiffSep/data/rir",
    dset="train", 
    sr=32000, 
    test_snr=-10, 
    p=1.0,  # EQ適用確率100%
    segment=3
)

# 評価用データセット
eval_dataset = AudioMixCrosstalkTwoNoiseEQ(
    audio_dir="/workspaces/SoundDiffSep/data/audio",
    rir_dir="/workspaces/SoundDiffSep/data/rir",
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
