import os
import random
import scipy.signal

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from audiomentations import BandStopFilter


class AudioMixCrosstalkTwoNoiseEQ(torch.utils.data.Dataset):
    """
    AudioMixCrosstalkTwoNoiseEQ データセットクラス
    
    仕様書に基づいたシンプルな実装:
    - ターゲット音声にRIRを適用してクロストークを模擬
    - 2つの異なるノイズ音声にそれぞれ異なるRIRを適用
    - ノイズ音声にバンドストップフィルタによるEQ処理を適用
    - ターゲット+ノイズ、ノイズ+ターゲットの2つの混合音声を生成
    """
    
    def __init__(self, audio_dir, rir_dir, dset="train", sr=32000, test_snr=-5, p=1.0, segment=10):
        """
        Args:
            audio_dir: 音声ファイルが格納されているディレクトリパス
            rir_dir: RIRファイルが格納されているディレクトリパス
            dset: "train" または "eval"
            sr: サンプリングレート（32kHz）
            test_snr: 評価時のSNR値
            p: EQ処理の適用確率
            segment: 音声セグメントの長さ（秒）
        """
        assert dset in ["train", "eval"], "`dset` must be one of ['train', 'eval']"
        
        self.audio_dir = audio_dir
        self.rir_dir = rir_dir
        self.dset = dset
        self.sr = sr
        self.test_snr = test_snr
        self.segment = segment
        self.p = p
        
        # RIRファイルリストの取得
        self.rir_names = self._get_rir_names()
        
        # 音声ファイルリストの取得
        self.audio_names = self._get_audio_names()
        
        # EQフィルタの設定
        self.eq = BandStopFilter(
            min_center_freq=200,
            max_center_freq=10000,
            min_bandwidth_fraction=0.25,
            max_bandwidth_fraction=1,
            zero_phase=True,
            p=self.p,
        )

        
        # 評価時の固定設定
        if dset == "eval":
            self._setup_eval_fixed_settings()
    
    def _get_rir_names(self):
        """RIRファイルリストの取得"""
        rir_names = []
        for root, dirs, files in os.walk(self.rir_dir):
            for file in files:
                if file.endswith(".wav"):
                    rir_names.append(os.path.join(root, file))
        return rir_names
    
    def _get_audio_names(self):
        """音声ファイルリストの取得"""
        audio_names = []
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_names.append(os.path.join(root, file))
        return audio_names
    
    def _setup_eval_fixed_settings(self):
        """評価時の固定設定"""
        # 評価時は再現性のためにノイズとRIRを固定
        random.seed(42)  # 固定シード
        
        # 1つ目のノイズとRIR
        self.noise_names_1 = [self._choose_different_noise(tgt) for tgt in self.audio_names]
        self.rir_names_1 = random.choices(self.rir_names, k=len(self.audio_names))
        
        # 2つ目のノイズとRIR
        self.noise_names_2 = [self._choose_different_noise(tgt) for tgt in self.audio_names]
        self.rir_names_2 = random.choices(self.rir_names, k=len(self.audio_names))
        
        # ターゲット用RIR
        self.rir_names_tgt = random.choices(self.rir_names, k=len(self.audio_names))
        
        random.seed()  # シードをリセット
    
    def _choose_different_noise(self, target_name):
        """ターゲットと異なるノイズを選択"""
        while True:
            noise_name = random.choice(self.audio_names)
            if noise_name != target_name:
                return noise_name
    
    def __len__(self):
        return len(self.audio_names)
    
    def load_wav(self, path):
        """音声ファイルの読み込みとセグメント処理"""
        max_length = int(self.sr * self.segment)
        wav = librosa.core.load(path, sr=self.sr)[0]
        
        if len(wav) > max_length:
            # 最大振幅を中心とした切り出し
            max_amp_idx = np.argmax(np.abs(wav))
            start = max(0, max_amp_idx - max_length // 2)
            start = min(start, len(wav) - max_length)
            wav = wav[start:start + max_length]
        elif len(wav) < max_length:
            # ゼロパディング
            wav = np.pad(wav, (0, max_length - len(wav)), "constant")
        
        return wav
    
    def load_audio(self, path):
        """音声ファイルの読み込み"""
        return torch.tensor(self.load_wav(path))
    
    def load_rir(self, rir_path):
        """RIRファイルの読み込み"""
        return librosa.core.load(rir_path, sr=self.sr)[0]
    
    def apply_rir(self, audio, rir):
        """音声にRIRを適用"""
        convolved = scipy.signal.fftconvolve(audio.numpy(), rir, mode="full")
        convolved = convolved[:int(self.sr * self.segment)]
        return torch.tensor(convolved.astype(np.float32))
    
    def mix_audio(self, target, noise, snr_db):
        """指定SNRで音声を混合"""
        # 次元を合わせる
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)
        if snr_db.dim() == 0:
            snr_db = snr_db.unsqueeze(0)
            
        return torchaudio.functional.add_noise(target, noise, snr=snr_db)
    
    def normalize_audio(self, target, noise_mixed, target_mixed):
        """音声の正規化（最大振幅を0.9に）"""
        # target_mixedを基準に正規化
        max_value = torch.max(torch.abs(target_mixed))
        target_normalized = target * 0.9 / max_value
        target_mixed_normalized = target_mixed * 0.9 / max_value
        
        # noise_mixedは独立して正規化
        noise_max_value = torch.max(torch.abs(noise_mixed))
        noise_mixed_normalized = noise_mixed * 0.9 / noise_max_value
        
        return target_normalized, noise_mixed_normalized, target_mixed_normalized
    
    def __getitem__(self, idx):
        """データアイテムの取得"""
        # 1. 音声とRIRの選択
        target_name = self.audio_names[idx]
        
        if self.dset == "train":
            # 訓練時はランダム選択
            noise_name1 = self._choose_different_noise(target_name)
            noise_name2 = self._choose_different_noise(target_name)
            rir_name1 = random.choice(self.rir_names)
            rir_name2 = random.choice(self.rir_names)
            rir_name_tgt = random.choice(self.rir_names)
        else:
            # 評価時は固定設定を使用
            noise_name1 = self.noise_names_1[idx]
            noise_name2 = self.noise_names_2[idx]
            rir_name1 = self.rir_names_1[idx]
            rir_name2 = self.rir_names_2[idx]
            rir_name_tgt = self.rir_names_tgt[idx]
        
        # 2. SNR設定
        if self.dset == "train":
            # 訓練時: -10dB〜+10dBのランダム値
            main_snr = torch.tensor((torch.rand(1) * 20 - 10).item())
            # ノイズミックスSNR: |メインSNR| + 5 + 0〜25dBのランダム値
            noise_mix_snr = torch.tensor(abs(main_snr.item()) + 5 + torch.rand(1).item() * 25)
        else:
            # 評価時: 固定値
            main_snr = torch.tensor(self.test_snr)
            noise_mix_snr = torch.tensor(abs(self.test_snr) + 5)
        
        # 3. 音声データの読み込み
        target_raw = self.load_audio(target_name)
        noise_clean1 = self.load_audio(noise_name1)
        noise_clean2 = self.load_audio(noise_name2)
        
        # 4. RIRの読み込み
        rir_target = self.load_rir(rir_name_tgt)
        rir1 = self.load_rir(rir_name1)
        rir2 = self.load_rir(rir_name2)
        
        # 5. RIRの適用
        target_reverb = self.apply_rir(target_raw, rir_target)  # ターゲットにRIR適用
        noise1_reverb = self.apply_rir(noise_clean1, rir1)     # ノイズ1にRIR適用
        noise2_reverb = self.apply_rir(noise_clean2, rir2)     # ノイズ2にRIR適用
        
        # 6. ノイズの合成
        noise_reverb_combined = noise1_reverb + noise2_reverb  # リバーブ後のノイズ2つを合算
        noise_clean_combined = noise_clean1 + noise_clean2     # リバーブ前のノイズ2つを合算
        
        # 7. 音声の混合
        # ターゲット + ノイズの混合
        target_mixed = self.mix_audio(target_raw, noise_reverb_combined, main_snr)
        
        # ノイズ + ターゲットの混合
        noise_mixed = self.mix_audio(noise_clean_combined, target_reverb, noise_mix_snr)
        
        # 8. EQ処理の適用（訓練時のみ）
        if self.dset == "train" and self.eq is not None:
            noise_mixed = torch.tensor(
                self.eq(noise_mixed.numpy(), sample_rate=self.sr)
            )
        
        # 9. 正規化
        target_final, noise_final, mixed_final = self.normalize_audio(
            target_raw, noise_mixed, target_mixed
        )
        
        return (
            mixed_final.squeeze(),  # メイン混合音声（リバーブ前ターゲット + リバーブ後2ノイズ合成）
            noise_final.squeeze(),  # ノイズ混合音声（リバーブ前2ノイズ合成 + リバーブ後ターゲット + EQ処理）
            target_final.squeeze(), # クリーンターゲット音声（リバーブ前）
        )


if __name__ == "__main__":
    # 使用例
    audio_dir = "/workspaces/SoundDiffSep/data/audio"  # 音声ファイルディレクトリ
    rir_dir = "/workspaces/SoundDiffSep/data/rir"    # RIRファイルディレクトリ（サンプル用）
    
    dataset = AudioMixCrosstalkTwoNoiseEQ(
        audio_dir=audio_dir,
        rir_dir=rir_dir,
        dset="train", 
        sr=32000, 
        test_snr=-10, 
        p=1.0,  # EQ適用確率100%
        segment=3
    )
    
    # データの取得
    mixed, noise, target = dataset[0]
    print(f"Dataset length: {len(dataset)}")
    print(f"Mixed shape: {mixed.shape}")
    print(f"Noise shape: {noise.shape}")
    print(f"Target shape: {target.shape}")
    
    # 音声ファイルの保存（テスト用）
    sf.write("mixed.wav", mixed.numpy(), dataset.sr, subtype="PCM_16")
    sf.write("noise.wav", noise.numpy(), dataset.sr, subtype="PCM_16")
    sf.write("target.wav", target.numpy(), dataset.sr, subtype="PCM_16")
