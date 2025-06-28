"""
信号处理模块
负责对声波信号进行高通滤波、连续小波变换和统计特征提取
"""

import numpy as np
import pywt
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Tuple, Dict, List, Any
import yaml


class SignalProcessor:
    """信号处理器类"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化信号处理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.filter_config = self.config['processing']['highpass_filter']
        self.feature_config = self.config['features']
        
    def apply_highpass_filter(self, waveform: np.ndarray) -> np.ndarray:
        """
        应用4阶巴特沃斯高通滤波器
        
        Args:
            waveform: 输入波形 (1024,)
            
        Returns:
            filtered_waveform: 滤波后的波形 (1024,)
        """
        # 设计滤波器
        nyquist = self.filter_config['sampling_rate'] / 2
        normalized_cutoff = self.filter_config['cutoff_freq'] / nyquist
        
        b, a = signal.butter(
            self.filter_config['filter_order'], 
            normalized_cutoff, 
            btype='high'
        )
        
        # 应用滤波器
        filtered_waveform = signal.filtfilt(b, a, waveform)
        
        return filtered_waveform
    
    def extract_cwt_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        提取连续小波变换特征
        
        Args:
            waveform: 输入波形 (1024,)
            
        Returns:
            cwt_coeffs: CWT系数 (scales, 1024)
        """
        # 确保波形是float类型
        waveform = waveform.astype(np.float64)
        
        # 生成尺度
        scales = np.arange(1, self.feature_config['cwt']['scales'] + 1, dtype=np.float64)
        
        # 计算CWT，不使用sampling_period参数避免类型问题
        cwt_coeffs, _ = pywt.cwt(
            waveform, 
            scales, 
            self.feature_config['cwt']['wavelet']
        )
        
        # 取绝对值并归一化
        cwt_coeffs = np.abs(cwt_coeffs)
        
        # 对数变换以增强对比度
        cwt_coeffs = np.log1p(cwt_coeffs)
        
        return cwt_coeffs
    
    def extract_statistical_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        提取统计特征
        
        Args:
            waveform: 输入波形 (1024,)
            
        Returns:
            features: 统计特征向量 (n_features,)
        """
        features = []
        
        # 确保波形是float类型
        waveform = waveform.astype(np.float64)
        
        # 时间轴
        dt = float(self.feature_config['cwt']['sampling_period'])
        time_axis = np.arange(len(waveform), dtype=np.float64) * dt
        
        # 频域分析
        fft = np.fft.fft(waveform)
        freqs = np.fft.fftfreq(len(waveform), dt)
        magnitude = np.abs(fft)
        
        # 只考虑正频率部分
        positive_freq_mask = freqs > 0
        freqs_pos = freqs[positive_freq_mask]
        magnitude_pos = magnitude[positive_freq_mask]
        
        for feature_name in self.feature_config['statistical_features']:
            if feature_name == 'max_abs':
                # 最大绝对幅值
                feature_value = np.max(np.abs(waveform))
                
            elif feature_name == 'rms':
                # 均方根幅值
                feature_value = np.sqrt(np.mean(waveform**2))
                
            elif feature_name == 'peak_time':
                # 峰值到达时间
                peak_idx = np.argmax(np.abs(waveform))
                feature_value = time_axis[peak_idx]
                
            elif feature_name == 'skewness':
                # 偏度
                feature_value = skew(waveform)
                
            elif feature_name == 'kurtosis':
                # 峰度
                feature_value = kurtosis(waveform)
                
            elif feature_name == 'dominant_freq':
                # 主频（最大幅值对应的频率）
                dominant_idx = np.argmax(magnitude_pos)
                feature_value = freqs_pos[dominant_idx]
                
            elif feature_name == 'spectral_centroid':
                # 频谱重心
                feature_value = np.sum(freqs_pos * magnitude_pos) / np.sum(magnitude_pos)
                
            elif feature_name == 'spectral_bandwidth':
                # 频谱带宽
                centroid = np.sum(freqs_pos * magnitude_pos) / np.sum(magnitude_pos)
                feature_value = np.sqrt(np.sum(((freqs_pos - centroid)**2) * magnitude_pos) / np.sum(magnitude_pos))
                
            else:
                feature_value = 0.0
                
            features.append(feature_value)
        
        return np.array(features)
    
    def process_waveform(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        处理单个波形
        
        Args:
            waveform: 输入波形 (1024,)
            
        Returns:
            filtered_waveform: 滤波后的波形 (1024,)
            cwt_features: CWT特征 (scales, 1024)
            stat_features: 统计特征 (n_features,)
        """
        # 高通滤波
        filtered_waveform = self.apply_highpass_filter(waveform)
        
        # 提取CWT特征
        cwt_features = self.extract_cwt_features(filtered_waveform)
        
        # 提取统计特征
        stat_features = self.extract_statistical_features(filtered_waveform)
        
        return filtered_waveform, cwt_features, stat_features
    
    def process_batch_waveforms(self, waveforms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量处理波形
        
        Args:
            waveforms: 输入波形批次 (batch_size, 1024)
            
        Returns:
            filtered_waveforms: 滤波后的波形批次 (batch_size, 1024)
            cwt_features_batch: CWT特征批次 (batch_size, scales, 1024)
            stat_features_batch: 统计特征批次 (batch_size, n_features)
        """
        batch_size = waveforms.shape[0]
        
        # 初始化输出数组
        filtered_waveforms = np.zeros_like(waveforms)
        cwt_features_batch = np.zeros((batch_size, self.feature_config['cwt']['scales'], waveforms.shape[1]))
        stat_features_batch = np.zeros((batch_size, len(self.feature_config['statistical_features'])))
        
        # 逐个处理波形
        for i in range(batch_size):
            filtered_waveform, cwt_features, stat_features = self.process_waveform(waveforms[i])
            
            filtered_waveforms[i] = filtered_waveform
            cwt_features_batch[i] = cwt_features
            stat_features_batch[i] = stat_features
        
        return filtered_waveforms, cwt_features_batch, stat_features_batch


def test_signal_processor():
    """测试信号处理器"""
    print("测试信号处理器...")
    
    processor = SignalProcessor()
    
    # 生成测试信号
    dt = 1e-5
    t = np.arange(1024) * dt
    # 混合频率信号：低频噪声 + 高频信号
    test_signal = (
        0.5 * np.sin(2 * np.pi * 500 * t) +  # 低频成分（应被滤除）
        1.0 * np.sin(2 * np.pi * 5000 * t) +  # 高频成分
        0.1 * np.random.randn(1024)  # 噪声
    )
    
    # 处理信号
    filtered_signal, cwt_features, stat_features = processor.process_waveform(test_signal)
    
    print(f"原始信号形状: {test_signal.shape}")
    print(f"滤波后信号形状: {filtered_signal.shape}")
    print(f"CWT特征形状: {cwt_features.shape}")
    print(f"统计特征形状: {stat_features.shape}")
    print(f"统计特征值: {stat_features}")
    
    # 测试批量处理
    batch_waveforms = np.random.randn(10, 1024)
    filtered_batch, cwt_batch, stat_batch = processor.process_batch_waveforms(batch_waveforms)
    
    print(f"批量处理 - 输入形状: {batch_waveforms.shape}")
    print(f"批量处理 - CWT特征形状: {cwt_batch.shape}")
    print(f"批量处理 - 统计特征形状: {stat_batch.shape}")
    
    print("信号处理器测试通过！")


if __name__ == "__main__":
    test_signal_processor() 