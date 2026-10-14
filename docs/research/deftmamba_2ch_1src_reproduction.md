# DeFT-Mamba 2ch/1src baseline 再現レポート

## 1. 結論と範囲

本変更は、固定 upstream commit
`34e2c9a59fd4e3b0135b55dac9be3d972602bfd8` の公開 separation branch を基準に、
次の二点を分けて確認するための experimental baseline である。

1. 実行不能な公開コードへ forward 成立に必要な最小 runtime fix を適用し、
   その演算を独立実装と照合できるか。
2. 2 microphone input / 1 source output の契約で forward、backward、保存・復元、
   および小さな固定合成 batch への学習が成立するか。

upstream default からの意図的な入出力変更は次の二点だけである。

| 項目 | upstream default | この baseline |
| --- | ---: | ---: |
| `n_mics` | 4 | **2** |
| `n_srcs` | 4 | **1** |

主設定は公開コード default と同じ 4 blocks である。6 blocks は論文記載の depth
だけを変えた補助設定であり、主結果や「論文再現」とは扱わない。

非目標は、complex/amplitude mask、mixture/reference-microphone への mask 乗算、
MIMO・4-source・可変 source 数、PIT、classification、source counting、SRT、
公開されていない公式 loss/training の推測実装、Mamba/Attention の置換、
既存データ正規化の変更、および既存モデルとの性能比較である。
本結果は論文性能または実データ分離性能を示さない。

## 2. 実装とライセンス境界

GPL-2.0 の upstream source は SoundDiffSep package へコピーしていない。
reference runner は無視対象の `.cache/upstream/DeFTMamba` に固定 commit を取得し、
GPL source にだけ
`minimal_runtime_fixes.patch`（GPL-2.0 のコード片を含むため、このリポジトリでは配布しない。内容は upstream の構文・import・形状の 6 点の実行上の修正で、表は監査文書を参照）
を適用する。この外部実行物を **`upstream_minfix`** と呼び、「正しい公式モデル」
とは呼ばない。raw runner は cache の patch 状態によらず固定 commit の
`git archive` を一時展開するため、無修正 source を監査できる。
checkout の expected/actual commit 一致は
[`upstream_checkout.json`](evidence/upstream_checkout.json) に保存した。

`sounddiffsep/models/deftmamba/` は、論文と公開コードから確認できる構造・tensor
contract を別の名前と構成で記述した MIT 側の独立実装である。正しさは、同一
input と upstream の state tensors を shape 確認付きで対応させた数値比較で検証する。
この分離により、SoundDiffSep 全体の MIT license は変更しない。

## 3. Tensor flow と RI 規約

waveform wrapper と core の流れは次のとおりである。

```text
waveform input                [B, 2, N]
  -> complex STFT             [B, 2, F, T]
  -> channel-major RI pack    [B, F, T, 4]
  -> DeFT-Mamba core          [B, F, T, 2]
  -> one-source RI unpack     [B, 1, F, T] complex
  -> ISTFT(length=N)          [B, 1, N]
```

2-microphone input の packing は次である。

```text
[mic0_real, mic0_imag, mic1_real, mic1_imag]
```

output は `[source_real, source_imag]` と解釈する。公開コードは complex tensor
からの pack/unpack を持たないため、これは **この再現実装で採用した規約**であり、
公式規約とは断言しない。round-trip、順序、gradient は unit test で確認した。

output head は公開コードと同じ `Conv2d(emb_dim, 2*n_srcs)`、今回なら
`Conv2d(64, 2)` である。core output を推定 spectrum そのものとして ISTFT し、
次の処理は一切行わない。

```python
estimated_spec = mask * input_spec  # 実装していない
```

固定 spectrum を返す dummy core で、異なる入力 waveform に対して同じ推定結果に
なることも検査しており、wrapper が暗黙に input spectrum を乗算しないことを確認した。

## 4. STFT と教師信号の契約

主設定
[`official_2ch_1src_16k.yaml`](../../configs/deftmamba/official_2ch_1src_16k.yaml)
は次を採用する。

| 項目 | 値 |
| --- | ---: |
| sample rate / duration | 16,000 Hz / 4 s（64,000 samples） |
| `n_fft` / `win_length` / `hop_length` | 512 / 512 / 256 |
| window | Hamming |
| `ri_packing` | `channel_major_interleaved`（この再現実装の規約） |
| `center` / `periodic` / `normalized` | `true` / `true` / `false` |
| `onesided` | `true` |

16 kHz、Hamming 32 ms、hop 16 ms は論文条件に合わせたため、window/hop は
512/256 samples となる。一方、model 用 STFT/ISTFT は upstream 公開コードに存在せず、
`n_fft=512`、`center=true`、periodic Hamming、非 normalized、padding/ISTFT details が
著者実験と同じかは **公開情報から確定できない**。wrapper の STFT 設定は変更可能だが、
正式に検証した baseline は上記だけである。

1-source output は dataset が渡す `[B,1,N]` 教師に対応する。dry source、特定 mic
位置の source image、full/direct reverberant image のどれを著者実験が使ったかは
**公開情報から確定できない**ため、モデル内部では意味を仮定しない。

## 5. Optional dependency と統合契約

通常の `import sounddiffsep` または `import sounddiffsep.models.deftmamba` では
`mamba_ssm` と `flash_attn` を import しない。名前付き experimental factory
`deftmamba_official_2ch_1src` で core を生成するときだけ遅延 import し、不足時は
導入方法を含む `ImportError` を返す。GRU、Identity、PyTorch SDPA などへの黙った
fallback はない。公式再現経路は CUDA + FlashAttention を必要とする。
upstream authors が使用した正確な dependency 組合せは公開情報から確定できない。

既存 `TwoChSepSystem.common_step` は `mixed` と `noise_clean/reference` を
`torch.stack(..., dim=1)` して `[B,2,N]` を作る。wrapper output `[B,1,N]` は
`tgt.unsqueeze(1)` の `[B,1,N]` と一致するため、既存 batch stack と loss の形を
変更せず接続できる。この契約は dedicated test でも `[B,2,511] -> [B,1,511]`
として確認した。既存 ResUNet/DCUNet の loader default は変更していない。
標準 `pip install -e .`（DeFT extras なし）、通常 import、ResUNet、DCUNet-16、
DCUNet-20、Large-DCUNet-20 の constructor も成功した。parameter counts と
非依存 import の確認結果は
[`integration_checks.json`](evidence/integration_checks.json) に保存した。既存 checkpoint
自体は変更しておらず、この検証では checkpoint load を実行していない。

## 6. Raw upstream と `upstream_minfix`

### 6.1 無修正 source

固定 commit の無修正 `git archive` では、`py_compile`、import、`compileall` の
3 child commands がすべて return code 1 だった。最初の blocker は line 23 の
`flash_attn_func*` による `SyntaxError` である。runner 自身が JSON を保存して
正常終了することと、child command の失敗を混同しない。完全な stdout/stderr は
[`raw_upstream_audit.json`](evidence/raw_upstream_audit.json) に保存した。

その後に確認された blocker と最小修正は、2 箇所の tab/space 混在、公開 tree に
ない `models.arch.mamba_ssm.modules.block` import、標準 `Block` と
`Mamba2Simple` の full-sequence API 差、unfold 後 `D*G` channels に対する
GatedConv の `D` input 宣言、および `D` output と `D*G` unfold の成立しない加算である。
Attention、Mamba2、normalization、activation、decoder、dropout の公開演算は別方式へ
置換していない。ただし、公開 tree にない local `Block` を tested standard `Block` で
補う runtime 仮定と著者依存の数値同一性は **公開情報から確定できない**。

### 6.2 2ch/1src minfix 実行

`n_layers=1, n_mics=2, n_srcs=1, F=12, T=10, dropout=0` で
`[1,12,10,4] -> [1,12,10,2]` の forward/backward が完了した。output、input
gradient、84 parameter gradients は finite、parameter/trainable parameter はともに
631,042、state reload 後の max absolute error は 0 だった。単発の
forward+backward は 12.0294 s、peak allocated は 323,375,616 bytes だった。
これは benchmark 値ではない。詳細は
[`upstream_minfix_2ch_1src.json`](evidence/upstream_minfix_2ch_1src.json) にある。

## 7. 独立実装との reference equivalence

比較条件は `B=1, F=12, T=10, n_layers=1, emb_dim=64, n_mics=2,
n_srcs=1, dropout=0, float32, seed=20260829, CUDA`、許容値は
`atol=1e-5, rtol=1e-4` である。

| 比較項目 | upstream_minfix | SoundDiffSep | 結果 |
| --- | ---: | ---: | --- |
| parameters | 631,042 | 631,042 | 一致 |
| trainable parameters | 631,042 | 631,042 | 一致 |
| state_dict entries | 84 | 84 | 全 shape 一致 |
| comparison points | 22 | 22 | 全点 tolerance 内 |
| first mismatch | — | — | `null` |

主要 intermediate shape は次である。

| 段階 | shape |
| --- | --- |
| encoder | `[1,64,10,12]` |
| frequency normalized/output/reconstructed | `[1,64,10,12]` |
| frequency unfold / gated / attention・Mamba tokens | `[10,256,9]` / `[10,64,9]` / `[10,9,64]` |
| time normalized/output/reconstructed | `[1,64,10,12]` |
| time unfold / gated / attention・Mamba tokens | `[12,256,7]` / `[12,64,7]` / `[12,7,64]` |
| output-head input / channels-first output | `[1,64,10,12]` / `[1,2,10,12]` |
| final output | `[1,12,10,2]` |

22 点すべてで maximum absolute error、mean absolute error、relative L2 error は
`0.0` だった。cosine similarity は float32 reduction の丸めを含め
`0.9999998807907104` から `1.0000001192092896`、final output は `1.0` である。
したがって最初の不一致層は存在しない。全 state key 対応、shape、各点の個別値は
[`reference_equivalence.json`](evidence/reference_equivalence.json) に保存した。

## 8. 4/6-layer waveform smoke

両設定を `B=1, float32, 16 kHz, 4 s` で実行し、いずれも
`[1,2,64000] -> [1,1,64000]`、finite output、finite input/parameter gradients、
state reload 後の seeded replay error 0 を確認した。

| depth | parameters | forward | forward+backward | peak allocated | peak reserved | evidence |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 2,513,218 | 4.7510 s | 12.8100 s | 8.569 GiB | 9.127 GiB | [`smoke_4layer_4s.json`](evidence/smoke_4layer_4s.json) |
| 6 | 3,768,002 | 4.7744 s | 12.8765 s | 12.403 GiB | 13.045 GiB | [`smoke_6layer_4s.json`](evidence/smoke_6layer_4s.json) |

この smoke timing は各一回で、depth 間の性能比較には用いない。公開 Attention は
`eval()` 時も `dropout_p=0.1` を渡すため、serialization comparison は同じ CUDA RNG
seed を replay した。この stochastic eval behavior は修正せず記録している。
RNG を reset しない同一入力の連続 eval は、4層で max/mean absolute difference
`0.00655575 / 0.00107540`、6層で `0.00839784 / 0.00144760` となり、いずれも
exactly equal ではなかった。smoke timing は process ごとの初回 Triton JIT を含む
cold-start diagnostic であり、性能比較には次節の warm-up 後 benchmark だけを使う。

## 9. One-batch learning smoke

外部データは使わず、複数正弦波、chirp、impulse を含む 1-source target と、
異なる delay（2/5 samples）、gain（0.90/0.65）、独立 noise を持つ 2-channel input
を生成した。quick 条件は `n_layers=1, samples=512, n_fft=64, hop=32,
dropout=0, 12 optimizer steps` である。loss は学習可能性確認用の
**complex RI L1** であり、公式 training loss ではない。

loss curve は非単調で、
`0.5140 -> 1.1531 -> 0.6925 -> 0.5173 -> 0.5754 -> 0.4596 ->
0.7021 -> 0.4974 -> 0.3952 -> 0.3887 -> 0.4323 -> 0.3703 -> 0.3002`
だった。initial 0.5139865、final 0.3001612、final/initial 0.5839864
（約 41.6% 低下）で、事前定義した `final < 0.8 * initial` を満たした。
output/gradients は finite、12 steps の training は 12.2209 s（0.9819 steps/s）、
最終評価込み 12.2766 s、peak allocated/reserved は
373,175,808 / 597,688,320 bytes だった。詳細は
[`one_batch_overfit_quick.json`](evidence/one_batch_overfit_quick.json) にある。

これは短い固定 batch に最適化できることだけを示し、主 4-layer/4 s 条件の訓練、
汎化、source separation quality、論文性能を示さない。

## 10. 4/6-layer benchmark

条件は `B=1, float32, 16 kHz, 4 s, CUDA`、warmup 1 回、計測 3 回である。
RTF は mean forward time / 4 s、steps/s は mean forward+backward の逆数である。

| depth | params | forward mean | fwd+bwd mean | steps/s | RTF | allocated fwd / fwd+bwd | reserved fwd / fwd+bwd | evidence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 2,513,218 | 0.041092 s | 0.125854 s | 7.9457 | 0.010273 | 0.630 / 8.340 GiB | 0.730 / 8.812 GiB | [`benchmark_4layer_4s.json`](evidence/benchmark_4layer_4s.json) |
| 6 | 3,768,002 | 0.060906 s | 0.190945 s | 5.2371 | 0.015227 | 0.635 / 12.171 GiB | 0.734 / 12.732 GiB | [`benchmark_6layer_4s.json`](evidence/benchmark_6layer_4s.json) |

forward-only phase 前に warm-up の gradient を解放し allocator cache を空にしたため、
表の forward allocated/reserved 値は training warm-up の保持分を含まない。両方で
output と gradient は finite だった。I/O と parameters は float32 だが、公開 Attention
どおり Q/K/V は内部で BF16 へ cast する。また dropout 0.1 の eval forward は
stochastic である。MACs は正確に測定していないため推定値を報告しない。結果はこの
短い反復回数に基づく diagnostic であり、環境をまたぐ性能比較には用いない。

## 11. 実行コマンド

evidence 生成時は出力先も指定した。

```bash
python -m compileall sounddiffsep
python -m pytest -q

python tools/deftmamba_reference/prepare_upstream.py \
  --output docs/research/evidence/upstream_checkout.json
python tools/deftmamba_reference/run_raw_upstream.py \
  --output docs/research/evidence/raw_upstream_audit.json
python tools/deftmamba_reference/run_minfixed_upstream.py \
  --n-mics 2 --n-srcs 1 \
  --output docs/research/evidence/upstream_minfix_2ch_1src.json
python tools/deftmamba_reference/compare_with_sounddiffsep.py \
  --n-mics 2 --n-srcs 1 \
  --output docs/research/evidence/reference_equivalence.json

python scripts/smoke_deftmamba_2ch_1src.py \
  --sample-rate 16000 --seconds 4 --batch-size 1 --n-layers 4 \
  --output docs/research/evidence/smoke_4layer_4s.json
python scripts/smoke_deftmamba_2ch_1src.py \
  --sample-rate 16000 --seconds 4 --batch-size 1 --n-layers 6 \
  --output docs/research/evidence/smoke_6layer_4s.json
python scripts/smoke_train_deftmamba_2ch_1src.py --quick \
  --output docs/research/evidence/one_batch_overfit_quick.json
python scripts/benchmark_deftmamba_2ch_1src.py \
  --sample-rate 16000 --seconds 4 --batch-size 1 --n-layers 4 \
  --output docs/research/evidence/benchmark_4layer_4s.json
python scripts/benchmark_deftmamba_2ch_1src.py \
  --sample-rate 16000 --seconds 4 --batch-size 1 --n-layers 6 \
  --output docs/research/evidence/benchmark_6layer_4s.json
```

通常 test は `pytest.ini` で external reference marker を除外し、外部 clone/CUDA
比較は明示的に分離する。

```bash
python -m pytest -q
python -m pytest -q -m deftmamba_reference
```

記録した実行では通常 suite が `36 passed, 1 deselected in 16.13s`、dedicated
reference suite が `1 passed, 36 deselected in 9.40s` だった。CUDA unavailable の場合、
CUDA-only tests は理由付き skip となる。

## 12. Failed/skipped items と既知の制約

- raw upstream の compile/import/compileall はすべて SyntaxError で失敗した。これは
  隠さず raw evidence として残している。
- CPU には公式 FlashAttention kernel がない。diagnostic attention fallback は作らず、
  CUDA unavailable または optional dependency 不足時の reference/core tests は skip する。
- eval 時の attention dropout は公開コードどおり残る。`dropout>0` の反復出力は
  stochastic であり、serialization 同一性には RNG replay が必要である。
- `deftmamba_official_2ch_1src` は dedicated experimental factory から生成する。
  checkpoint 前提の既存 `easy_load_model` registry へは追加していないため、そこから
  選択可能とは主張しない。生成した wrapper は `TwoChSepSystem` の stack 契約へ接続できる。
- `n_mics=2, n_srcs=1` だけを正式な対応・試験対象とする。constructor が受け取れる
  他の値を対応済みとは扱わない。
- 6-layer config は depth/計算量の補助確認であり、論文の object grouping、class decoder、
  loss、PIT、SRT、training schedule を再現しない。
- RI packing、model STFT の一部、waveform target の意味は upstream から確定できない。
- one-batch overfit は quick 設定の synthetic batch と非公式 complex RI L1 に限られる。
- 実データ評価、学習済み checkpoint、分離指標、入力/出力 scale 保存、長時間 chunk
  inference、16 kHz 以外、既存 DCUNet との同条件比較は未実施である。
- benchmark は 3 iterations の diagnostic であり、MACs は未計測である。
- 公開 tree にない local `Block` と、今回使用した公開 `mamba_ssm.Block` が著者環境と
  完全に同じだったかは確定できない。`upstream_minfix` は公開 path を実行可能にする
  最小解釈であって、著者の未公開 training system の復元ではない。

以上から言えるのは、固定公開コードの blocker を追跡可能な patch で解消した
`upstream_minfix` と独立 MIT 実装が、検証条件では同じ parameter shape・中間値・最終値を
生成し、2ch waveform から 1ch waveform まで finite に微分・保存でき、短い合成 batch
へ loss を低下させられた、という範囲までである。
