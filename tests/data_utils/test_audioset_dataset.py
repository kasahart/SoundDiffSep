#!/usr/bin/env python3
"""
AudioSetMixCrosstalkTwoNoiseEQ データセットクラスの単体テスト
"""

import os
import numpy as np
import torch
import soundfile as sf
from sounddiffsep.data_utils.audioset_dataset import AudioSetMixCrosstalkTwoNoiseEQ


def create_test_audio_files():
    """テスト用の音声ファイルを生成"""
    # テスト用ディレクトリの作成
    test_dir = "/tmp/test_audio"
    os.makedirs(test_dir, exist_ok=True)
    
    # サンプル音声データを生成（1秒間、16kHz）
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 異なる周波数のサイン波を生成
    audio_files = []
    frequencies = [440, 880, 1320, 660, 220]  # A4, A5, E6, E5, A3
    
    for i, freq in enumerate(frequencies):
        # サイン波生成
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # ファイル保存
        filename = f"{test_dir}/test_audio_{i}_{freq}Hz.wav"
        sf.write(filename, signal, sr)
        audio_files.append(filename)
        print(f"Created: {filename}")
    
    return test_dir, audio_files

def test_dataset_basic():
    """基本的なデータセット機能のテスト"""
    print("=== Basic Dataset Test ===")
    
    # テスト用音声ファイルを作成
    test_dir, audio_files = create_test_audio_files()
    

    # データセット作成
    dataset = AudioSetMixCrosstalkTwoNoiseEQ(
        audio_dir=test_dir,
        rir_dir=test_dir,  # 同じディレクトリをRIRとしても使用
        dset="train",
        sr=16000,
        segment=1,  # 1秒
        p=0.0  # EQ処理を無効化
    )
    
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should contain audio files"
    
    # データ取得テスト
    mixed, noise, target = dataset[0]
    print(f"Mixed shape: {mixed.shape}")
    print(f"Noise shape: {noise.shape}")
    print(f"Target shape: {target.shape}")
    
    # 形状確認
    expected_length = 16000  # 1秒 × 16kHz
    assert mixed.shape[0] == expected_length, f"Expected {expected_length}, got {mixed.shape[0]}"
    assert noise.shape[0] == expected_length, f"Expected {expected_length}, got {noise.shape[0]}"
    assert target.shape[0] == expected_length, f"Expected {expected_length}, got {target.shape[0]}"
    
    print("✓ Basic dataset test passed!")
    
    # テストファイルをクリーンアップ
    cleanup_test_files()



def test_eval_mode():
    """評価モードのテスト"""
    print("=== Eval Mode Test ===")
    
    test_dir, audio_files = create_test_audio_files()
    
        
        # 評価モードでデータセット作成
    dataset = AudioSetMixCrosstalkTwoNoiseEQ(
        audio_dir=test_dir,
        rir_dir=test_dir,
        dset="eval",
        sr=16000,
        test_snr=-5,
        segment=1,
        p=0.0
    )
    
    print(f"Eval dataset length: {len(dataset)}")
    
    # 再現性テスト - 同じインデックスで同じ結果が得られるか
    mixed1, noise1, target1 = dataset[0]
    mixed2, noise2, target2 = dataset[0]
    
    # 同じ結果が得られることを確認
    assert torch.allclose(mixed1, mixed2), "Eval mode should be deterministic"
    assert torch.allclose(noise1, noise2), "Eval mode should be deterministic"
    assert torch.allclose(target1, target2), "Eval mode should be deterministic"
    
    print("✓ Eval mode test passed!")
    
    # テストファイルをクリーンアップ
    cleanup_test_files()



def test_different_parameters():
    """異なるパラメータでのテスト"""
    print("=== Different Parameters Test ===")
    
    test_dir, audio_files = create_test_audio_files()

    # 異なるセグメント長でテスト
    for segment in [0.5, 1.0, 2.0]:
        dataset = AudioSetMixCrosstalkTwoNoiseEQ(
            audio_dir=test_dir,
            rir_dir=test_dir,
            dset="train",
            sr=16000,
            segment=segment,
            p=0.0
        )
        
        if len(dataset) > 0:
            mixed, noise, target = dataset[0]
            expected_length = int(16000 * segment)
            assert mixed.shape[0] == expected_length, f"Segment {segment}s: expected {expected_length}, got {mixed.shape[0]}"
            print(f"✓ Segment {segment}s test passed")
    
    print("✓ Different parameters test completed!")
    
    # テストファイルをクリーンアップ
    cleanup_test_files()
        

def cleanup_test_files():
    """テストファイルのクリーンアップ"""
    test_dir = "/tmp/test_audio"
    if os.path.exists(test_dir):
        try:
            import shutil
            shutil.rmtree(test_dir)
            print(f"Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up test directory {test_dir}: {e}")
    else:
        print(f"Test directory {test_dir} does not exist, no cleanup needed")

if __name__ == "__main__":
    try:
        # 全てのテストを実行
        test_dataset_basic()
        test_eval_mode()
        test_different_parameters()
        print("\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        # 最終的なクリーンアップを確実に実行
        cleanup_test_files()
        print("Final cleanup completed.")

