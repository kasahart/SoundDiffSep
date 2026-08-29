# DeFT-Mamba 2ch/1src baseline の次段階

この baseline は、固定公開コードを最小 runtime fix で実行できるか、および
2-channel input / 1-channel output が学習可能か、という二つだけを切り分けた。
次の項目は主要因子を同時に変えないため、今回の pull request には含めない。

1. 教師契約（dry source または明示した microphone の source image）を固定し、
   version 管理した実データで separation quality を評価する。
2. `n_fft`、`center`、Hamming の periodicity、normalization を明示し、STFT 条件を
   一因子ずつ変更する。
3. constructor が値を受け取れるだけで対応済みとせず、16 kHz 以外を個別検証する。
4. 公開コードに沿う direct RI mapping と、別実装の complex-mask path を比較する。
   reference baseline 自体へ mask を混ぜない。
5. architecture と dataset を固定し、complex RI L1、waveform L1、SNR 系 loss を
   一つずつ比較する。
6. 入出力 waveform scale と物理振幅の保存性を評価する。Pa、SPL、RMS、energy の
   暗黙な normalization は追加しない。
7. 長時間音声の chunk inference を別実装し、境界 artifact、overlap 方針、state、
   latency、memory をそれぞれ測定する。
8. data、STFT、target、loss、duration、sample rate、metric を同一にして既存
   DCUNet と比較する。

complex masking、MIMO output、source counting、classification、PIT、separation
refinement tuning なども、参照元と受け入れ条件を持つ別変更として
提案する。
