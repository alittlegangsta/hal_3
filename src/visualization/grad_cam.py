"""
Grad-CAM可视化模块
实现Grad-CAM技术，用于分析CNN模型对CWT时频特征的关注区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, List, Dict, Any
import pickle
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.cnn_model import create_model
from src.models.feature_extractor import FeatureExtractor


class GradCAM:
    """Grad-CAM可视化器"""
    
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        """
        初始化Grad-CAM
        
        Args:
            model: 已训练的模型
            target_layer_name: 目标卷积层名称
        """
        self.model = model
        self.model.eval()
        
        # 设置目标层
        if target_layer_name is None:
            # 默认使用最后一个卷积层
            self.target_layer = self.model.cnn_branch.conv_layers[-1]
        else:
            self.target_layer = dict(self.model.named_modules())[target_layer_name]
        
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """保存前向传播的激活值"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0]
    
    def generate_cam(self, cwt_features: torch.Tensor, stat_features: torch.Tensor, 
                    class_idx: int = 0) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            cwt_features: CWT特征 (1, 1, scales, time_points)
            stat_features: 统计特征 (1, n_features)
            class_idx: 类别索引（回归任务通常为0）
            
        Returns:
            cam: Grad-CAM热力图 (scales, time_points)
        """
        # 确保输入需要梯度
        cwt_features.requires_grad_(True)
        
        # 前向传播
        output = self.model(cwt_features, stat_features)
        
        # 清除之前的梯度
        self.model.zero_grad()
        
        # 反向传播
        if output.dim() > 1:
            output = output[0, class_idx]
        else:
            output = output[0]
        
        output.backward()
        
        # 获取梯度和激活值
        gradients = self.gradients[0]  # (channels, height, width)
        activations = self.activations[0]  # (channels, height, width)
        
        # 计算权重（全局平均池化）
        weights = torch.mean(gradients, dim=(1, 2))  # (channels,)
        
        # 生成CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # (height, width)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU激活
        cam = F.relu(cam)
        
        # 归一化到[0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, cwt_features: torch.Tensor, stat_features: torch.Tensor,
                     original_cwt: np.ndarray, label: float, save_path: str = None) -> plt.Figure:
        """
        可视化Grad-CAM结果
        
        Args:
            cwt_features: CWT特征张量 (1, 1, scales, time_points)
            stat_features: 统计特征张量 (1, n_features)
            original_cwt: 原始CWT特征 (scales, time_points)
            label: 真实标签
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        # 生成CAM
        cam = self.generate_cam(cwt_features, stat_features)
        
        # 获取模型预测
        with torch.no_grad():
            prediction = self.model(cwt_features, stat_features).item()
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 时间轴（毫秒）
        time_axis = np.arange(original_cwt.shape[1]) * 1e-5 * 1000
        scale_axis = np.arange(original_cwt.shape[0])
        
        # 1. 原始CWT特征
        im1 = axes[0].imshow(original_cwt, aspect='auto', cmap='viridis',
                           extent=[time_axis[0], time_axis[-1], scale_axis[-1], scale_axis[0]])
        axes[0].set_title('原始CWT特征')
        axes[0].set_xlabel('时间 (ms)')
        axes[0].set_ylabel('尺度')
        plt.colorbar(im1, ax=axes[0])
        
        # 2. Grad-CAM热力图
        im2 = axes[1].imshow(cam, aspect='auto', cmap='jet', alpha=0.8,
                           extent=[time_axis[0], time_axis[-1], scale_axis[-1], scale_axis[0]])
        axes[1].set_title('Grad-CAM热力图')
        axes[1].set_xlabel('时间 (ms)')
        axes[1].set_ylabel('尺度')
        plt.colorbar(im2, ax=axes[1])
        
        # 3. 叠加显示
        axes[2].imshow(original_cwt, aspect='auto', cmap='gray', alpha=0.7,
                      extent=[time_axis[0], time_axis[-1], scale_axis[-1], scale_axis[0]])
        im3 = axes[2].imshow(cam, aspect='auto', cmap='jet', alpha=0.5,
                           extent=[time_axis[0], time_axis[-1], scale_axis[-1], scale_axis[0]])
        axes[2].set_title(f'叠加显示\n真实标签: {label:.3f}, 预测值: {prediction:.3f}')
        axes[2].set_xlabel('时间 (ms)')
        axes[2].set_ylabel('尺度')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grad-CAM可视化保存到: {save_path}")
        
        return fig


def analyze_high_channeling_samples():
    """分析高窜槽样本的Grad-CAM"""
    print("=" * 70)
    print("分析高窜槽样本的Grad-CAM")
    print("=" * 70)
    
    # 加载模型
    model = create_model()
    model.load_state_dict(torch.load('data/processed/best_model.pth', map_location='cpu'))
    model.eval()
    
    # 加载特征提取器
    feature_extractor = FeatureExtractor()
    feature_extractor.load('data/processed/feature_extractor.pkl')
    
    # 加载预处理数据
    with open('data/processed/small_sample_processed.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model)
    
    # 分析各个方位的高窜槽样本
    os.makedirs('data/results', exist_ok=True)
    
    for side in ['A', 'B']:
        print(f"\n分析方位 {side} 的高窜槽样本...")
        
        side_data = processed_data[side]
        labels = side_data['labels']
        
        # 找到高窜槽样本（标签 > 0.5）
        high_channeling_indices = np.where(labels > 0.5)[0]
        
        if len(high_channeling_indices) == 0:
            print(f"方位 {side} 没有高窜槽样本")
            continue
        
        # 选择几个最高窜槽比例的样本
        sorted_indices = high_channeling_indices[np.argsort(labels[high_channeling_indices])[::-1]]
        top_samples = sorted_indices[:min(3, len(sorted_indices))]
        
        for i, sample_idx in enumerate(top_samples):
            print(f"  处理样本 {i+1}: 索引 {sample_idx}, 标签 {labels[sample_idx]:.3f}")
            
            # 获取样本数据
            original_cwt = side_data['cwt_features'][sample_idx]  # (scales, time_points)
            stat_features_raw = side_data['stat_features'][sample_idx]  # (n_features,)
            
            # 归一化特征
            norm_cwt, norm_stat = feature_extractor.transform(
                original_cwt[np.newaxis, :, :], 
                stat_features_raw[np.newaxis, :]
            )
            
            # 转换为张量
            cwt_tensor, stat_tensor, _ = feature_extractor.to_tensors(
                norm_cwt, norm_stat, np.array([labels[sample_idx]])
            )
            
            # 生成Grad-CAM可视化
            save_path = f'data/results/gradcam_side_{side}_sample_{i+1}_label_{labels[sample_idx]:.3f}.png'
            fig = grad_cam.visualize_cam(
                cwt_tensor, stat_tensor, original_cwt, labels[sample_idx], save_path
            )
            plt.close(fig)
    
    print("\nGrad-CAM分析完成！")
    print("结果保存在 data/results/ 目录下")


if __name__ == "__main__":
    analyze_high_channeling_samples() 