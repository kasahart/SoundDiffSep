import numpy as np
from scipy import signal

class OverlapAdd:
    """
    重複加算法を使ってマルチチャネル信号をセグメントに分割、バッチとして処理し、再構成するクラス
    
    Args:
        window_size (int): セグメントの窓サイズ
        hop_size (int, optional): セグメント間のホップサイズ。指定しない場合はwindow_size // 2
        window (str, optional): 窓関数の種類（'hann', 'hamming'など）
    """
    def __init__(self, window_size, hop_size=None, window='hann'):
        self.window_size = window_size
        self.hop_size = hop_size if hop_size is not None else window_size // 2
        
        # 窓関数の作成
        if window:
            self.window = signal.get_window(window, window_size)
            self.use_window = True
        else:
            self.window = np.ones(window_size)
            self.use_window = False
    
    def segment_signal(self, x):
        """
        マルチチャネル信号をセグメントに分割する
        
        Args:
            x (numpy.ndarray): 入力信号、形状は(n_channels, n_samples)
            
        Returns:
            numpy.ndarray: セグメント化された信号、形状は(n_segments, n_channels, window_size)
        """
        n_channels, n_samples = x.shape
        
        # パディングを追加
        pad_size = self.window_size
        x_padded = np.pad(x, ((0, 0), (0, pad_size)))
        
        # セグメントの数を計算
        n_segments = 1 + (n_samples - self.window_size) // self.hop_size + 1
        
        # セグメントを格納する配列を初期化 (n_segments, n_channels, window_size)
        segments = np.zeros((n_segments, n_channels, self.window_size))
        
        # 信号をセグメントに分割
        for i in range(n_segments):
            start = i * self.hop_size
            segments[i] = x_padded[:, start:start + self.window_size]
            
        return segments
    
    def process_signal(self, x, processing_func):
        """
        マルチチャネル信号を処理する
        
        Args:
            x (numpy.ndarray): 入力信号、形状は(n_channels, n_samples)
            processing_func (callable): バッチ処理を行う関数。
                                      (n_segments, n_channels, window_size)の配列を入力とし、
                                      同じ形状の処理済み配列を返すこと
            
        Returns:
            numpy.ndarray: 処理された信号、形状は(n_channels, n_samples)
        """
        n_channels, n_samples = x.shape
        
        # 信号をセグメントに分割 -> (n_segments, n_channels, window_size)
        segments = self.segment_signal(x)
        
        # すべてのセグメントをバッチとして一度に処理
        processed_segments = processing_func(segments)
        
        # 窓関数を適用 (各チャネルの各セグメントに)
        if self.use_window:
            # window形状を(1, 1, window_size)に拡張してブロードキャスト
            processed_segments = processed_segments * self.window.reshape(1, 1, -1)
        else:
            processed_segments = processed_segments / (self.window_size / self.hop_size)
        
        # 重複加算による再構成
        output = np.zeros((n_channels, n_samples + self.window_size))
        
        for i in range(processed_segments.shape[0]):
            start = i * self.hop_size
            # 各チャネル、各セグメントを適切な位置に加算
            output[:, start:start + self.window_size] += processed_segments[i]
        
        # 正規化係数を計算（窓関数の重なりによる影響を補正）
        if self.use_window:
            # 窓関数の重ね合わせを計算
            window_sum = np.zeros((1, n_samples + self.window_size))
            for i in range(segments.shape[0]):
                start = i * self.hop_size
                window_sum[0, start:start + self.window_size] += self.window
            
            # 0除算を避ける
            mask = window_sum > 1e-10
            output = np.where(mask, output / window_sum, output)
        
        # 元の長さに戻す
        return output[:, :n_samples]
