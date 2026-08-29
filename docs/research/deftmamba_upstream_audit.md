# DeFT-Mamba 固定 upstream 監査

## 1. 目的と結論

この文書は、DeFT-Mamba 公式公開リポジトリを固定 commit
`34e2c9a59fd4e3b0135b55dac9be3d972602bfd8` で監査した記録である。
対象は公開物そのものとし、論文記述、公開コード、実行に必要な最小 patch、
SoundDiffSep 側で採用する 2 microphone / 1 source 契約を混同しない。

監査の結論は次のとおりである。

- 無修正の `DeFTMamba.py` は構文エラーのため parse、import、forward できない。
- 構文エラーの次に、2 箇所の tab/space 混在、公開 tree に存在しない
  `models.arch.mamba_ssm.modules.block`、GatedConv の channel 不整合がある。
- 論文は 6 blocks、`D -> DS` の object grouping、audio/class decoders、
  source counting、PIT、2-stage training、separation refinement tuning を記述する。
  公開 `DeFTMamba.py` は既定 4 blocks と `D -> 2S` の direct complex spectral
  mapping だけを実装し、classification と training 系は公開していない。
- 最小 patch 適用後の実装を本プロジェクトでは **`upstream_minfix`** と呼ぶ。
  「修正版公式実装」または「正しい公式モデル」とは呼ばない。
- `upstream_minfix` は CUDA 上で compile、import、`n_mics=2, n_srcs=1` の
  class 生成、full forward/backward、finite output/gradient、state_dict
  serialization を確認した。
- 同じ実行条件、同一重み、同一入力で SoundDiffSep 独立実装と比較し、
  encoder、frequency/time block、output head を含む 22 比較点の maximum/mean
  absolute error と relative L2 error はすべて 0、最初の不一致はなかった。
- 論文性能の再現、公式 training loss の再現、分離性能の主張はこの監査からは行わない。

## 2. 一次資料と固定条件

| 項目 | 固定値・参照先 |
| --- | --- |
| 公式 repository | [donghoney0416/DeFTMamba](https://github.com/donghoney0416/DeFTMamba) |
| 固定 commit | [`34e2c9a59fd4e3b0135b55dac9be3d972602bfd8`](https://github.com/donghoney0416/DeFTMamba/tree/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8) |
| commit metadata | 2026-01-26T15:29:58+09:00, `Add GNU General Public License v2` |
| 論文 | [arXiv:2409.12413v1](https://arxiv.org/abs/2409.12413), 2024-09-19 |
| 論文 method | [Section II](https://arxiv.org/html/2409.12413#S2) |
| 論文 experiment | [Section III](https://arxiv.org/html/2409.12413#S3) |
| 公開 architecture figure | [固定 commit の `src/fig/Framework.png`](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/src/fig/Framework.png) |
| upstream license | GNU General Public License Version 2 (`LICENSE`) |

upstream は SoundDiffSep package 内へ vendor しない。`.cache/upstream/DeFTMamba`
の外部 checkout と、GPL-2.0 コードへ適用する
[`minimal_runtime_fixes.patch`](../../third_party_patches/deftmamba/minimal_runtime_fixes.patch)
を使う。SoundDiffSep 側の実装は別に記述された独立実装であり、MIT license を
GPL へ変更しない。

## 3. 公開 file inventory

固定 commit の `git ls-tree -r --name-only HEAD` は 91 files だった。repository
root の実行・設定ファイルは次の 5 files で、残りは demo 用 audio 40 files と
figure/image 46 files である。

| 公開物 | 内容 | 監査結果 |
| --- | --- | --- |
| `DeFTMamba.py` | spectral core model | 存在するが、無修正では構文解析不可 |
| `AudioDataset2.py` | gpuRIR を使う simulation data generator | 存在。単独 compile は成功するが、hard-coded 外部 data path と未定義の実行環境を必要とする |
| `scene_generator.cfg` | 16 kHz、4 s、room/SNR 等の generator 設定 | 存在 |
| `index.html` | demo page | 存在。demo audio/figure を参照する |
| `LICENSE` | license text | GPL Version 2 |
| `src/audio/**` | mixture、ground truth、baseline、proposal の demo WAV | 40 files。checkpoint ではない |
| `src/fig/**` | architecture、dataset、result table、spectrogram images | 46 files |

固定 tree には次がない。

- README または model usage documentation
- `requirements.txt`、`environment.yml`、`pyproject.toml`、package version pin
- training script、optimizer loop、loss implementation、PIT implementation
- inference script、waveform wrapper、model 用 STFT/ISTFT implementation
- checkpoint または learned weights
- class decoder implementation
- classification-based source counting implementation
- separation refinement tuning implementation
- 公開コードが import する `models/arch/mamba_ssm/modules/block.py`

`AudioDataset2.py` には STFT/ISTFT があるが、これは diffuse noise の空間相関を
合成するための SciPy 処理である。Hann window、`nfft=256`、75% overlap を使い、
論文の separation model 前後の Hamming STFT/ISTFT 実装ではない。

## 4. 無修正 upstream の実行記録

### 4.1 unmodified git archive の結果

[`run_raw_upstream.py`](../../tools/deftmamba_reference/run_raw_upstream.py) は、dirty
になり得る cache checkout を直接検査せず、固定 commit から `git archive` で
一時 tree を作り、その無修正 tree に対して検査する。runner 自体の return code
は 0 だが、これは 3 件の failure を JSON として正常に記録できたことを表す。
各 child command の return code は 1 である。
完全な command/stdout/stderr は
[`raw_upstream_audit.json`](evidence/raw_upstream_audit.json) に保存した。
公開記録では実行 interpreter のローカルパスだけを `<runtime-python>` に匿名化している。
引数、return code、stdout、stderr は変更していない。

#### `<runtime-python> -m py_compile DeFTMamba.py`

```console
$ <runtime-python> -m py_compile DeFTMamba.py
  File "DeFTMamba.py", line 23
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func*
                                                                     ^
SyntaxError: invalid syntax
```

`stdout` は空、`stderr` は上記、return code は 1 だった。

#### `<runtime-python> -c "import DeFTMamba"`

```console
$ <runtime-python> -c "import DeFTMamba"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "./DeFTMamba.py", line 23
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func*
                                                                     ^
SyntaxError: invalid syntax
```

`stdout` は空、`stderr` は上記、return code は 1 だった。

#### `<runtime-python> -m compileall .`

```console
$ <runtime-python> -m compileall .
Listing '.'...
Compiling './AudioDataset2.py'...
Compiling './DeFTMamba.py'...
***   File "./DeFTMamba.py", line 23
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func*
                                                                     ^
SyntaxError: invalid syntax

Listing './src'...
Listing './src/audio'...
Listing './src/audio/clip1'...
Listing './src/audio/clip2'...
Listing './src/audio/clip3'...
Listing './src/audio/clip4'...
Listing './src/fig'...
Listing './src/fig/clip1'...
Listing './src/fig/clip2'...
Listing './src/fig/clip3'...
Listing './src/fig/clip4'...
```

この command は return code 1 だった。出力から `AudioDataset2.py` は compile
対象になり、failure は `DeFTMamba.py` の line 23 で発生したことが分かる。

### 4.2 最初の syntax error の後ろにある blocker

無修正 file を書き換えず、source text を memory 上で段階的に補正して
`compile()` / `exec()` した結果は次の順だった。

1. line 23 の末尾 `*` だけを除くと、line 291 で
   `TabError: inconsistent use of tabs and spaces in indentation`。
2. line 291 の tab だけを空白へ直すと、line 308 で同じ `TabError`。
3. 2 箇所の indentation を直すと parse は成功する。
4. parse 後、現行 optional dependency を導入した環境でも line 19 の
   `from models.arch.mamba_ssm.modules.block import Block` が
   `ModuleNotFoundError: No module named 'models'` になる。固定 tree に
   `models/` は存在しない。
5. import path を公開 `mamba_ssm.modules.block` へ向けると、
   `n_mics=2, n_srcs=1` の class 自体は生成でき、encoder input channels は 4、
   output head channels は 2 になる。
6. その状態の最初の frequency GCB で、unfold output `[10, 256, 9]` に対し
   Conv1d weight `[64, 64, 3]` が適用され、次で停止した。

```text
RuntimeError: Given groups=1, weight of size [64, 64, 3],
expected input[10, 256, 9] to have 64 channels, but got 256 channels instead
```

7. GatedConv input を 256 channels に合わせるだけでは、Conv output 64 channels
   と raw unfold 256 channels の加算が次で停止する。

```text
RuntimeError: The size of tensor a (64) must match the size of tensor b (256)
at non-singleton dimension 1
```

したがって、無修正 upstream は「dependency を入れれば動く」状態ではない。
syntax、indentation、import、tensor shape の独立した blocker がある。

## 5. Blocker classification

| 分類 | blocker | 無修正での影響 | 対応方針 |
| --- | --- | --- | --- |
| 1. syntax blocker | `flash_attn_func*` | file 全体が parse 不可 | 余分な `*` だけを除く |
| 2. import blocker | `models.arch.mamba_ssm.modules.block` | 公開 tree に module がなく import 不可 | 公開 package の `mamba_ssm.modules.block.Block` を使う |
| 3. dependency blocker | requirements/version pin がない | 著者の dependency 組合せを再構築不能 | optional dependency として分離する |
| 3. dependency blocker | tested `Block.forward` は `inference_params` を mixer へ渡すが、tested `Mamba2Simple.forward` は受け取らない | full sequence Mamba 呼び出し前に API mismatch となる | `None` のみ受け、元の full-sequence forward へ委譲する parameter-free adapter を追加 |
| 4. indentation blocker | lines 291, 308 の tab | line 23 修正後も parse 不可 | tab を周囲と同じ 8 spaces にする |
| 5. tensor-shape blocker | unfold は `C*G`、GatedConv は `C` input として生成 | `G=4` で 256 channels を 64-channel Conv に入力 | Conv input を `emb_dim*emb_ks` にする |
| 5. tensor-shape blocker | GatedConv の `D` output と raw unfold の `GD` を加算 | `D=64,G=4` で 64 + 256 channels | 成立しない GCB 内加算を除く。block 外側の residual は維持 |
| 6. behavioral change | 意図的な architecture 置換はなし。ただし standard `Block` と未公開 local `Block` の同一性は確定不能 | 公開 tree だけでは著者依存を実行不能 | Attention/Mamba/normalization/activation/decoder/dropout の公開演算は維持しつつ、この runtime 仮定を明示する |

`AudioDataset2.py` 内にも tab があるが、記録した無修正 file の `py_compile` は
成功した。したがって今回の model forward 用 patch には
含めない。

## 6. `upstream_minfix` の全修正

patch は GPL-2.0 upstream checkout だけへ適用する。各修正の影響を以下に示す。

| 修正 | 分類 | 修正前の error | 修正内容と必要性 | tensor/state shape への影響 | 数値挙動への影響 | 対応確認 |
| --- | --- | --- | --- | --- | --- | --- |
| FlashAttention import の `*` を削除 | syntax | line 23 `SyntaxError` | 有効な import 文にする | なし | 演算 graph に到達する前のためなし | raw `py_compile` failure と minfix `py_compile` success |
| `Block` import path を変更 | import | `No module named 'models'` | 固定 tree にない local path から、導入済み `mamba_ssm.modules.block` へ変更 | 公開 package の `Block` が同じ parameter layout ならなし | 著者の未公開 local `Block` と公開 package `Block` の数値同一性は **公開情報から確定できない** | minfix import、class construction |
| `FullSequenceMamba2Simple` adapter を追加し forward/backward Mamba で使用 | dependency/API | tested `Block` は `inference_params=None` を渡すが、`Mamba2Simple.forward(self,u,seq_idx=None)` は受理しない | `None` の場合だけ引数を受けて元の `Mamba2Simple.forward(u, ...)` へ委譲。non-`None` は明示 error | parameter を追加せず、state_dict key/shape を変えない | full-sequence、`inference_params=None` では元 call への委譲なので演算を変えない。streaming/incremental path は対象外 | signature inspection、CUDA forward/backward、serialization、22-point reference equivalence |
| GatedConv input を `emb_dim*emb_ks` に変更 | tensor shape | Conv weight `[64,64,3]` が 256-channel unfold input を拒否 | 論文の `GD -> D` 2-branch Conv と一致させる | 各 GatedConv weight は `[D,D,3]` から `[D,DG,3]`。`D=64,G=4` では `[64,64,3] -> [64,256,3]`。bias/output は D のまま | raw primary path は forward 不可能なため raw output との数値差は定義不能。parameter count/initialization draw は変わる | exact pre-fix RuntimeError、minfix class/state shape、reference comparison test |
| frequency/time GCB の `+ unfolded` を除去 | tensor shape | D-channel GatedConv output と GD-channel unfold を加算不能 | 論文 Fig.1 の gated `GD -> D` output をそのまま attention へ渡す。ConvTranspose 後の direction-level residual は残す | GCB output を D に保つ。state parameter shape への追加影響なし | `G=4` raw path は未定義なので数値比較不能。仮に `G=1` とした未検証 path では挙動変更になるため、正式対応を主張しない | exact pre-fix RuntimeError、CUDA forward/backward、frequency/time GCB を含む 22-point reference equivalence |
| lines 291, 308 の tab を spaces に変更 | indentation | 2 件の `TabError` | 周囲 block と同じ indentation にする | なし | なし | staged in-memory compile、minfix `compileall` |

この patch は次を修正していない。

- `n_layers=4` という公開 code default
- paper の `D -> DS` object grouping の欠落
- classification decoder、source counting、SRT の欠落
- Attention の BF16 強制 cast
- eval 時も `flash_attn_func` へ同じ `dropout_p` を渡す挙動
- `bidirectional=False` path
- Mamba-FFN の paper figure と code の channel-width 差
- output head `D -> 2*n_srcs`
- direct complex spectral mapping

これらを patch で変えると、forward 成立のための最小 runtime fix ではなく
architecture/behavior change になるためである。

## 7. `upstream_minfix` の現在の実行状態

### 7.1 実行条件

reference path は CUDA、Mamba2、FlashAttention を必要とする。これらは optional
dependency として通常 import から分離している。upstream authors が使用した正確な
dependency 組合せは **公開情報から確定できない**。

### 7.2 成功した確認

patch 適用後の外部 checkout に対して、次は成功した。

```console
$ python -m py_compile DeFTMamba.py
# return code 0

$ python -c "import DeFTMamba; print('import_ok')"
import_ok

$ python -m compileall -q .
# return code 0
```

次の class construction も成功した。

```text
n_mics=2
n_srcs=1
n_layers=1
up_conv.in_channels=4
down_conv.out_channels=2
parameter_count=631042
```

この compile/import/class construction に加え、以下の CUDA full forward/backward
と reference equivalence まで完了している。

### 7.3 `upstream_minfix` 2ch/1src CUDA forward/backward

[`run_minfixed_upstream.py`](../../tools/deftmamba_reference/run_minfixed_upstream.py)
を `n_mics=2, n_srcs=1` で実行した machine-readable result は
[`upstream_minfix_2ch_1src.json`](evidence/upstream_minfix_2ch_1src.json) である。
これは公式 FlashAttention と Mamba2 を使う `upstream_minfix` path であり、CPU
fallback、SDPA、GRU/LSTM/Transformer/Identity 置換ではない。

| 項目 | 実測結果 |
| --- | --- |
| config | `n_layers=1`, `emb_dim=64`, `n_heads=4`, `emb_ks=4`, `emb_hs=1`, `dropout=0`, float32 CUDA |
| core input | `[1,12,10,4]` |
| core output | `[1,12,10,2]`（期待 shape と一致） |
| output | finite |
| input gradient | finite |
| parameter gradients | 84 tensors に存在し、すべて finite |
| parameters / trainable | 631,042 / 631,042 |
| state_dict save/load 後の maximum absolute output error | 0.0 |
| forward + backward | 12.029364488 s |
| peak allocated CUDA memory | 323,375,616 bytes |

時間と memory はこの runner の単一実測で、定常状態 benchmark の代表値とは
扱わない。raw upstream は Section 4 の構文・shape blockers により同じ測定を
定義できないため、raw と minfix の数値差は報告しない。

### 7.4 SoundDiffSep 独立実装との 22-point equivalence

[`compare_with_sounddiffsep.py`](../../tools/deftmamba_reference/compare_with_sounddiffsep.py)
は `batch_size=1`, float32 CUDA, seed 20260829, `n_mics=2`, `n_srcs=1`,
`n_layers=1`, `emb_dim=64`, `dropout=0` で、同じ input と対応付けた同じ weights
を使った。結果は
[`reference_equivalence.json`](evidence/reference_equivalence.json) に保存した。

| 比較 | 結果 |
| --- | --- |
| parameters / trainable | upstream_minfix と SoundDiffSep がともに 631,042 / 631,042 |
| state_dict correspondence | 84 entries。各対応に共通 tensor shape を記録 |
| intermediate/final comparison points | 22 |
| shape comparison | 22/22 一致 |
| maximum absolute error | 22/22 で 0.0 |
| mean absolute error | 22/22 で 0.0 |
| relative L2 error | 22/22 で 0.0 |
| cosine similarity | 0.9999998807907104–1.0000001192092896 |
| first mismatch | `null` |
| final output | shape `[1,12,10,2]`、maximum/mean/relative error 0.0、cosine 1.0 |

22 points は encoder output、frequency block の normalized/unfolded/gated/
attention delta/output/Mamba delta/output/reconstructed/final、time block の同じ
9 points、output-head input/channels-first output、最終 `[B,F,T,2]` output である。
誤差指標が全点で厳密に 0 である一方、cosine の 1 前後の微小な表示差は
cosine reduction の float32 rounding であり、tensor element の不一致ではない。
設定した tolerance は absolute `1e-5`、relative `1e-4` だが、結果を合わせるために
tolerance を広げてはいない。

### 7.5 4 秒 waveform path の関連 CUDA evidence

SoundDiffSep wrapper の主設定 (`n_layers=4`) と paper-depth 補助設定
(`n_layers=6`) でも `[1,2,64000] -> [1,1,64000]` の forward/backward、finite
output/gradient、serialization error 0.0 を確認した。これは raw upstream の
waveform wrapper ではなく、この PR の独立実装を含む end-to-end smoke である。

| depth | parameters | forward (s) | forward+backward (s) | peak allocated / reserved bytes | 証跡 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 | 2,513,218 | 4.750980141 | 12.810026917 | 9,200,450,560 / 9,799,991,296 | [`smoke_4layer_4s.json`](evidence/smoke_4layer_4s.json) |
| 6 | 3,768,002 | 4.774435010 | 12.876496830 | 13,317,577,728 / 14,006,878,208 | [`smoke_6layer_4s.json`](evidence/smoke_6layer_4s.json) |

両 smoke は 16 kHz、4 s、Hamming、`n_fft=win_length=512`、`hop_length=256`、
batch 1、float32、CUDA、`dropout=0.1` である。serialization 比較では、後述の
公開 code の eval-time attention dropout を保存するため seed 20260830 を replay
している。cold/startup を含むこの値を、別の warmup/反復 benchmark 値と混同しない。
RNG reset なしの同一入力連続 eval は 4/6 層とも exactly equal ではなく、maximum
absolute difference はそれぞれ 0.00655575 / 0.00839784 だった。
定常 benchmark は
[`benchmark_4layer_4s.json`](evidence/benchmark_4layer_4s.json) と
[`benchmark_6layer_4s.json`](evidence/benchmark_6layer_4s.json)、学習可能性だけを
確認する quick one-batch overfit は
[`one_batch_overfit_quick.json`](evidence/one_batch_overfit_quick.json) に分離している。
後者は公開 training objective の再現ではない。

## 8. 公開 code から読める tensor contract

公開 `DeFTMamba.forward` は waveform を受け取らない。

```text
input:  [B, F, T, 2*M]
permute -> [B, 2*M, T, F]
encoder -> [B, D, T, F]
blocks  -> [B, D, T, F]
head    -> [B, 2*S, T, F]
permute -> [B, F, T, 2*S]
```

したがって、今回の意図的な設定変更は次の 2 点である。

```text
n_mics: 4 (upstream default) -> 2
n_srcs: 4 (upstream default) -> 1

core input:  [B, F, T, 4]
core output: [B, F, T, 2]
```

output は `down_conv(batch)` そのものであり、input spectrum との乗算はない。
公開 code に次の処理は存在しない。

```python
estimated_spec = mask * input_spec
```

よって公開 code が示す separation branch は direct complex spectral mapping である。
ただし、公開 code 自身には RI pack/unpack と ISTFT がない。

## 9. Architecture の静的監査

### 9.1 Encoder と output head

- encoder は kernel `(3,3)`、padding `(1,1)`、bias 有効の
  `Conv2d(2*n_mics, emb_dim)` と `GroupNorm(1, emb_dim)`。
- output head は同じ kernel/padding の `Conv2d(emb_dim, 2*n_srcs)`。
- 論文が明示するのは `2M -> D` encoder と、object ごとの `D -> 2` audio
  decoder までである。kernel、bias、GroupNorm groups、epsilon の実験値は
  **公開情報から確定できない**。

### 9.2 Frequency / time block

各 `DeFTANblock` は frequency、time の順で次を行う。

1. `LayerNormalization4D`
2. unfold (`emb_ks=4`, `emb_hs=1` が code default)
3. 2-branch GatedConv と Mish gate
4. pre-norm multi-head self-attention
5. pre-norm bidirectional Mamba-FFN
6. `ConvTranspose1d`
7. direction-level residual

padding 後の frequency length を `Q`、time length を `T`、embedding channel を
`C=D`、kernel を `G` とすると、frequency unfold は
`[B*T, D*G, (Q-G)/hop+1]` になる。time unfold も同様に `[B*Q,D*G,...]`
である。公開 code の GatedConv input 宣言が D だったことが shape blocker の
直接原因である。

### 9.3 Attention

論文 Section II-B は positional embedding を使わない MHSA と
FlashAttention-2 + linear projection を記述し、公開 code も Q/K/V linear、
FlashAttention、output linear を使う。この大枠は一致する。

公開 code 固有の挙動は次のとおりである。

- `dim_head = dim` とし、Q/K/V は各 `D -> D*n_heads`。
- Q/K/V を FlashAttention 前に常に BF16 へ cast する。
- `dropout_p=self.dropout` を `self.training` と無関係に渡す。
- output は FP32 へ戻して output projection へ渡す。
- `self.scale` は保存するが、call argument としては使わない。

[FlashAttention 公式 interface](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention)
は evaluation 時に `dropout_p=0.0` を指定するよう記述している。一方、公開 code
default は 0.1 で、`model.eval()` は plain float の `self.dropout` を変えない。
したがって dropout > 0 の再現経路では、eval 時も attention dropout を渡す
公開挙動を保存する。論文が意図した eval semantics は
**公開情報から確定できない**ため、最小 patch では修正しない。

### 9.4 Mamba-FFN

論文 Fig.1 は `Bi-Mamba2` を示す。公開 code も default
`bidirectional=True` で forward と reversed sequence の 2 paths を使うため、
主経路の bidirectionality は一致する。

ただし次の差・制約がある。

- tested `Mamba2Simple` は input と同じ D channels を返す。公開 `BiMamba` は
  forward/backward を concatenate して 2D とし、`FC(2D,2D)` を適用する。
- MambaFFN の次段は `Linear(hidden_dim//2,D)`。defaults では
  `hidden_dim//2=128=2D` なので成立するが、`hidden_dim` と D を独立に変更すると
  shape が成立する保証はない。
- 論文 Fig.1 は Bi-Mamba2 後を 4D と図示するため、図の 4D と公開 code の
  external 2D は一致しない。Mamba2 の `expand=4` は tested package では内部幅で、
  module external output は D である。
- `bidirectional=False` では `self.backward_blocks` が定義されないまま forward
  で参照される。また `FC` は常に 2D input を要求する。その path は動作しない。
  主経路は true 固定なので、最小 patch では変更しない。

著者が使用した未公開 local `Block` と Mamba version の正確な tensor contract は
**公開情報から確定できない**。

### 9.5 未使用 constructor arguments

`audio_mode`、`class_mode`、`onoff_mode`、`n_cls` は object に保存されるが、
module construction と `forward` では参照されない。`att_dim` も演算に使われない。
`use_lora`/`loralib` は実際の layer 置換に使われない。したがって、これらの
引数名から classification、source counting、LoRA の公開実装があると推測しては
ならない。

## 10. 論文と公開 code の discrepancy table

| 項目 | 論文 | 固定 commit の公開 code | 判定 |
| --- | --- | --- | --- |
| blocks | Section III-B: `N_b=6`, `D=64` | defaults は `n_layers=4`, `emb_dim=64`。6 は引数指定可能 | default depth が不一致。主 config は code default 4、paper-depth 補助 config は 6 と分離する |
| input encoder | Section II-A/Fig.1: complex `2M -> D`、Conv2d + group normalization | `Conv2d(2*n_mics,D)` + `GroupNorm(1,D)` | 大枠一致 |
| GCB | Section II-B/Fig.1: unfold 後 `GD -> D` の並列 Conv1d と Mish gate | raw code は D-input Conv を GD tensor に適用し、D output と GD input を加算 | 明示的 shape discrepancy。raw forward blocker |
| Attention | positional embedding なし MHSA、FlashAttention-2 + linear projection | 同じ構成要素を持つ | 大枠一致。dtype/head width/dropout は paper に詳細なし |
| bidirectionality | Fig.1 に Bi-Mamba2 | code default も bidirectional | 主経路は一致。false path は壊れている |
| Mamba-FFN width | Fig.1 は Bi-Mamba2 後 4D | tested API と code の concatenate/FC は external 2D | 不一致。著者の未公開 dependency contract は確定不能 |
| object grouping | blocks 後 Conv2d `D -> DS`、`S x D x T x F` に reshape | `feature_fusion` は comment out。`D -> 2S` head のみ | 論文の object feature grouping は未公開 |
| audio decoder | object ごとに `D -> 2` RI、その後 waveform `S x N` | 全 source をまとめて `D -> 2S`、spectral tensor を返す。ISTFT なし | 公開 code は direct RI spectral mapping のみ |
| class decoder | parallel CRNN、output `S x C` | class decoder なし | 論文のみ |
| class count | 13 foreground + silence = 14 classes | `n_cls=2` default だが未使用 | 数値も不一致。branch 自体が未公開 |
| source counting | silence class argmax で inactive 判定、Table V SCA | 実装なし | 論文のみ |
| stage 1 loss | SA-SDR + cross entropy、S-track PIT、audio/class shared permutation | loss/trainer/PIT なし | 論文のみ |
| SRT | estimated source count を使い、10% SI-SDR + 90% SDR で stage 2 | 実装なし | 論文のみ |
| STFT | 16 kHz、Hamming 32 ms、hop 16 ms | model STFT/ISTFT なし。generator 内に別目的の Hann/256/75% overlap のみ | wrapper の 512-sample window/256-sample hop は paper-aligned。FFT size等は確定不能 |
| duration/data | 16 kHz、4 s、2–4 foreground sources | cfg は 16 kHz/4 s。ただし generator は `[2,3,4,5]` から選び、5 sources も生成 | source count 条件が不一致 |
| movement | 75% moving、speed 0–3 m/s | 25% を固定し、残りへ sinusoidal trajectory。明示的な 0–3 m/s sampling/clamp なし | 75% は対応するが speed bound の一致は確定不能 |
| microphone array | 4-channel tetrahedron、radius 4.2 cm | generator 本体は 0.042 m を hard-code。一方 cfg は `microphone_radius=0.1` で未使用 | 実処理は paper と一致、cfg と実処理は不一致 |
| modes | joint separation/classification/on-off behavior | `audio_mode`, `class_mode`, `onoff_mode` は保存のみ | mode switching 実装なし |
| released workflow | paper は training/evaluation results を報告 | train/inference/checkpoint/dependency manifest なし | paper result を公開 tree だけで再現不能 |

論文の根拠箇所は次である。

- [Section II-A](https://arxiv.org/html/2409.12413#S2.SS1): `2M -> D`, S object
  tracks、audio `D -> 2` RI、CRNN class decoder。
- [Section II-B](https://arxiv.org/html/2409.12413#S2.SS2): GCB `GD -> D`、
  FlashAttention-2、Mamba-FFN、`D -> DS` grouping。
- [Section III-A](https://arxiv.org/html/2409.12413#S3.SS1): 16 kHz、4 s、
  2–4 sources、movement、4-channel 4.2 cm array。
- [Section III-B](https://arxiv.org/html/2409.12413#S3.SS2): loss、PIT、SRT、
  Hamming 32/16 ms、`N_b=6`, `D=64`。

公開 code の主要根拠は次である。

- [constructor、encoder、head](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/DeFTMamba.py#L27-L55)
- [spectral forward](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/DeFTMamba.py#L66-L78)
- [Attention](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/DeFTMamba.py#L81-L116)
- [BiMamba / MambaFFN](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/DeFTMamba.py#L119-L179)
- [GatedConv](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/DeFTMamba.py#L196-L204)
- [unfold と block reconstruction](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/DeFTMamba.py#L248-L313)
- [data generator](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/AudioDataset2.py#L114-L216)
- [generator の noise STFT](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/AudioDataset2.py#L298-L363)
- [generator の保存処理](https://github.com/donghoney0416/DeFTMamba/blob/34e2c9a59fd4e3b0135b55dac9be3d972602bfd8/AudioDataset2.py#L428-L477)

## 11. RI packing と waveform target

論文は multichannel complex spectrogram の channel 数を `2M`、audio decoder
output を 2 RI channels と記す。公開 code は `[B,F,T,2M]` を受け取るが、
complex tensor からの packing を実装していない。そのため、次のどちらかを
upstream から一意には選べない。

```text
[mic0_real, mic0_imag, mic1_real, mic1_imag, ...]
[mic0_real, mic1_real, ..., mic0_imag, mic1_imag, ...]
```

同様に、`2S` output の source/RI ordering を unpack する公開処理もない。
SoundDiffSep 再現実装では channel-major interleaved RI、すなわち
`[mic0_real,mic0_imag,mic1_real,mic1_imag]` と、1 source output
`[source_real,source_imag]` を採用するが、これは **この再現実装で採用した規約**
であり、公式規約とは断言しない。

論文 Eq.(1) は `X_{m,s}` を m 番 microphone で捕捉した reverberant source と
定義する一方、Fig.1 の waveform output は microphone 軸のない `S x N` である。
generator は full reverberant source image と direct-path source image を生成し、
保存処理では `full - direct` と direct を別 directory へ保存する。しかし training
script がないため、実験で dry source、特定 microphone の source image、
full reverberant image、または別の target のどれを supervision に使ったかは
**公開情報から確定できない**。

## 12. 公開情報から確定できない事項

以下は推測で埋めない。

- **公開情報から確定できない**: input microphone/RI channel の正確な packing 順。
- **公開情報から確定できない**: output source/RI channel の正確な unpack 順。
- **公開情報から確定できない**: waveform target が dry source か、どの microphone
  の source image か、full/direct/reverberation-only のどれか。
- **公開情報から確定できない**: model STFT の `n_fft`、`center`、Hamming の
  `periodic`、`normalized`、onesided/padding/ISTFT details。32 ms window と
  16 ms hop、16 kHz から window/hop samples は 512/256 と計算できるが、FFT size
  が必ず 512 とは論文だけでは断言できない。
- **公開情報から確定できない**: authors が実際に使った Python、PyTorch、CUDA、
  `mamba_ssm`、`flash_attn`、`loralib` versions。
- **公開情報から確定できない**: 未公開
  `models.arch.mamba_ssm.modules.block.Block` と公開 package `Block` の完全な数値同一性。
- **公開情報から確定できない**: paper Fig.1 の Mamba-FFN 4D 表記と公開 code の
  external 2D のどちらが著者実験 checkpoint に対応したか。
- **公開情報から確定できない**: paper 実験時の attention dropout と eval behavior。
- **公開情報から確定できない**: paper の movement speed 0–3 m/s が公開 generator
  の sinusoidal trajectory で常に満たされるか。
- **公開情報から確定できない**: 公開されていない training batch construction、
  loss weights（stage 1 joint loss 内の相対 weight）、scheduler、augmentation、
  checkpoint selection、inference post-processing。

## 13. この監査から言える範囲

固定 commit の公開 code は、入力 channel を `2*n_mics`、出力 channel を
`2*n_srcs` として構築する。したがって `n_mics=2,n_srcs=1` の parameterized
class construction と `[B,F,T,4] -> [B,F,T,2]` という意図は確認できる。

無修正 upstream は構文・import・tensor-shape blockers のため full forward
できない。一方、明示した最小 patch 後の `upstream_minfix` は、参照実行条件で
2ch/1src core の forward/backward、finite output/gradient、serialization error
0.0 まで確認できた。さらに SoundDiffSep 独立実装は、同一重み・同一入力の
22 intermediate/final points で maximum/mean absolute error と relative L2 error
がすべて 0、first mismatch が `null` だった。

これらは、固定 source の欠陥、最小 patch の範囲、paper/code の差、および
2ch/1src baseline の実行・移植・学習可能性を検証する証拠である。論文の未公開
training/classification/source-counting/SRT 系を補完するものではなく、論文性能の
再現や実データ上の分離性能を示すものでもない。
