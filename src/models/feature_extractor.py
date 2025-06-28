"""
特征提取器模块
负责对CWT特征和统计特征进行标准化和预处理
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Any
import pickle
import os


class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self, normalization_method: str = 'standard'):
        """
        初始化特征提取器
        
        Args:
            normalization_method: 归一化方法 ('standard' 或 'minmax')
        """
        self.normalization_method = normalization_method
        self.cwt_scaler = None
        self.stat_scaler = None
        self.is_fitted = False
        
    def fit(self, cwt_features: np.ndarray, stat_features: np.ndarray):
        """
        拟合特征缩放器
        
        Args:
            cwt_features: CWT特征 (n_samples, scales, time_points)
            stat_features: 统计特征 (n_samples, n_features)
        """
        print("拟合特征缩放器...")
        
        # 对CWT特征进行归一化 - 将所有样本展平
        cwt_flat = cwt_features.reshape(-1, cwt_features.shape[-1])  # (n_samples * scales, time_points)
        
        if self.normalization_method == 'standard':
            self.cwt_scaler = StandardScaler()
            self.stat_scaler = StandardScaler()
        else:  # minmax
            self.cwt_scaler = MinMaxScaler()
            self.stat_scaler = MinMaxScaler()
        
        # 拟合CWT特征缩放器
        self.cwt_scaler.fit(cwt_flat)
        
        # 拟合统计特征缩放器
        self.stat_scaler.fit(stat_features)
        
        self.is_fitted = True
        print("特征缩放器拟合完成")
        
    def transform(self, cwt_features: np.ndarray, stat_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换特征
        
        Args:
            cwt_features: CWT特征 (n_samples, scales, time_points)
            stat_features: 统计特征 (n_samples, n_features)
            
        Returns:
            normalized_cwt: 归一化CWT特征 (n_samples, scales, time_points)
            normalized_stat: 归一化统计特征 (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("特征提取器尚未拟合，请先调用fit()方法")
        
        # 归一化CWT特征
        original_shape = cwt_features.shape
        cwt_flat = cwt_features.reshape(-1, original_shape[-1])
        normalized_cwt_flat = self.cwt_scaler.transform(cwt_flat)
        normalized_cwt = normalized_cwt_flat.reshape(original_shape)
        
        # 归一化统计特征
        normalized_stat = self.stat_scaler.transform(stat_features)
        
        return normalized_cwt, normalized_stat
    
    def fit_transform(self, cwt_features: np.ndarray, stat_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并转换特征
        
        Args:
            cwt_features: CWT特征 (n_samples, scales, time_points)
            stat_features: 统计特征 (n_samples, n_features)
            
        Returns:
            normalized_cwt: 归一化CWT特征 (n_samples, scales, time_points)
            normalized_stat: 归一化统计特征 (n_samples, n_features)
        """
        self.fit(cwt_features, stat_features)
        return self.transform(cwt_features, stat_features)
    
    def to_tensors(self, cwt_features: np.ndarray, stat_features: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将numpy数组转换为PyTorch张量
        
        Args:
            cwt_features: CWT特征 (n_samples, scales, time_points)
            stat_features: 统计特征 (n_samples, n_features)
            labels: 标签 (n_samples,)
            
        Returns:
            cwt_tensor: CWT特征张量 (n_samples, 1, scales, time_points)
            stat_tensor: 统计特征张量 (n_samples, n_features)
            label_tensor: 标签张量 (n_samples,)
        """
        # CWT特征添加通道维度
        cwt_tensor = torch.FloatTensor(cwt_features).unsqueeze(1)  # 添加通道维度
        
        # 统计特征
        stat_tensor = torch.FloatTensor(stat_features)
        
        # 标签
        label_tensor = torch.FloatTensor(labels)
        
        return cwt_tensor, stat_tensor, label_tensor
    
    def save(self, save_path: str):
        """
        保存特征提取器
        
        Args:
            save_path: 保存路径
        """
        extractor_data = {
            'normalization_method': self.normalization_method,
            'cwt_scaler': self.cwt_scaler,
            'stat_scaler': self.stat_scaler,
            'is_fitted': self.is_fitted
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(extractor_data, f)
        
        print(f"特征提取器保存到: {save_path}")
    
    def load(self, load_path: str):
        """
        加载特征提取器
        
        Args:
            load_path: 加载路径
        """
        with open(load_path, 'rb') as f:
            extractor_data = pickle.load(f)
        
        self.normalization_method = extractor_data['normalization_method']
        self.cwt_scaler = extractor_data['cwt_scaler']
        self.stat_scaler = extractor_data['stat_scaler']
        self.is_fitted = extractor_data['is_fitted']
        
        print(f"特征提取器从 {load_path} 加载完成")


def prepare_dataset_from_processed_data(processed_data_path: str) -> Dict[str, Any]:
    """
    从预处理数据准备训练数据集
    
    Args:
        processed_data_path: 预处理数据路径
        
    Returns:
        dataset: 包含训练数据的字典
    """
    print(f"从 {processed_data_path} 加载预处理数据...")
    
    with open(processed_data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    # 合并所有方位的数据
    all_cwt_features = []
    all_stat_features = []
    all_labels = []
    all_sides = []
    
    for side, data in processed_data.items():
        all_cwt_features.append(data['cwt_features'])
        all_stat_features.append(data['stat_features'])
        all_labels.append(data['labels'])
        all_sides.extend([side] * len(data['labels']))
    
    # 合并数据
    cwt_features = np.vstack(all_cwt_features)
    stat_features = np.vstack(all_stat_features)
    labels = np.concatenate(all_labels)
    
    print(f"合并后的数据形状:")
    print(f"  CWT特征: {cwt_features.shape}")
    print(f"  统计特征: {stat_features.shape}")
    print(f"  标签: {labels.shape}")
    print(f"  标签范围: {labels.min():.3f} - {labels.max():.3f}")
    
    dataset = {
        'cwt_features': cwt_features,
        'stat_features': stat_features,
        'labels': labels,
        'sides': all_sides
    }
    
    return dataset


def test_feature_extractor():
    """测试特征提取器"""
    print("测试特征提取器...")
    
    # 生成测试数据
    n_samples = 100
    n_scales = 64
    n_time = 1024
    n_stat_features = 8
    
    cwt_features = np.random.randn(n_samples, n_scales, n_time)
    stat_features = np.random.randn(n_samples, n_stat_features)
    labels = np.random.rand(n_samples)
    
    # 测试特征提取器
    extractor = FeatureExtractor('standard')
    
    # 拟合和转换
    norm_cwt, norm_stat = extractor.fit_transform(cwt_features, stat_features)
    
    print(f"原始CWT特征形状: {cwt_features.shape}")
    print(f"归一化CWT特征形状: {norm_cwt.shape}")
    print(f"原始统计特征形状: {stat_features.shape}")
    print(f"归一化统计特征形状: {norm_stat.shape}")
    
    # 转换为张量
    cwt_tensor, stat_tensor, label_tensor = extractor.to_tensors(norm_cwt, norm_stat, labels)
    
    print(f"CWT张量形状: {cwt_tensor.shape}")
    print(f"统计特征张量形状: {stat_tensor.shape}")
    print(f"标签张量形状: {label_tensor.shape}")
    
    print("特征提取器测试通过！")


if __name__ == "__main__":
    test_feature_extractor() 