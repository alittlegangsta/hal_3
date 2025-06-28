#!/usr/bin/env python3
"""
Grad-CAM可视化脚本
用于生成声波测井数据的Grad-CAM热力图，分析模型关注的时频特征
"""

import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.cnn_model import HybridCNNModel
from src.visualization.grad_cam import GradCAM
from src.utils.device_utils import get_device

def load_model_and_data():
    """加载训练好的模型和数据"""
    print("Loading model and data...")
    
    # 加载预处理数据
    with open('data/processed/small_sample_processed.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # 创建模型
    model = HybridCNNModel(config_path='configs/config.yaml')
    
    # 使用统一的设备选择函数
    device = get_device('configs/config.yaml')
    
    # 加载训练好的权重
    if device.type == 'cuda':
        model.load_state_dict(torch.load('data/processed/best_model.pth'))
    else:
        model.load_state_dict(torch.load('data/processed/best_model.pth', map_location=device))
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    
    return model, processed_data, device

def generate_gradcam_visualization(azimuth='A', sample_idx=None, output_dir='data/results'):
    """
    生成Grad-CAM可视化
    
    Args:
        azimuth: 方位 ('A'-'H')
        sample_idx: 样本索引（如果为None，自动选择高窜槽样本）
        output_dir: 输出目录
    """
    # 加载模型和数据
    model, processed_data, device = load_model_and_data()
    
    # 获取指定方位的数据
    if azimuth not in processed_data:
        print(f"Error: Azimuth {azimuth} not found in processed data")
        return
    
    azimuth_data = processed_data[azimuth]
    cwt_features_all = azimuth_data['cwt_features']
    stat_features_all = azimuth_data['stat_features']
    labels_all = azimuth_data['labels']
    
    # 选择样本
    if sample_idx is None:
        # 自动选择高窜槽样本
        high_label_indices = [i for i, label in enumerate(labels_all) if label > 0.5]
        if high_label_indices:
            sample_idx = high_label_indices[0]
        else:
            # 如果没有高窜槽样本，选择标签最高的样本
            sample_idx = np.argmax(labels_all)
    
    if sample_idx >= len(cwt_features_all):
        print(f"Error: Sample index {sample_idx} out of range (max: {len(cwt_features_all)-1})")
        return
    
    # 获取样本数据
    cwt_features = cwt_features_all[sample_idx]
    statistical_features = stat_features_all[sample_idx]
    sample_label = labels_all[sample_idx]
    
    print(f"Analyzing sample {sample_idx} from azimuth {azimuth}")
    print(f"Channeling ratio: {sample_label:.3f}")
    
    # 转换为tensor
    cwt_tensor = torch.FloatTensor(cwt_features).unsqueeze(0).unsqueeze(0).to(device)
    stat_tensor = torch.FloatTensor(statistical_features).unsqueeze(0).to(device)
    
    # 模型预测
    with torch.no_grad():
        prediction = model(cwt_tensor, stat_tensor)
        pred_value = prediction.item()
    
    print(f"Model prediction: {pred_value:.3f}")
    
    # 生成Grad-CAM
    print("Generating Grad-CAM...")
    grad_cam = GradCAM(model, target_layer_name='cnn_branch.conv_layers.2')
    cam = grad_cam.generate_cam(cwt_tensor, stat_tensor, class_idx=0)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Grad-CAM Analysis - Azimuth {azimuth}, Sample {sample_idx}\n'
                 f'Label: {sample_label:.3f}, Prediction: {pred_value:.3f}', fontsize=14)
    
    # 原始CWT特征
    im1 = axes[0, 0].imshow(cwt_features, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Original CWT Features')
    axes[0, 0].set_ylabel('Frequency (scales)')
    axes[0, 0].set_xlabel('Time samples')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Grad-CAM热力图
    im2 = axes[0, 1].imshow(cam, aspect='auto', cmap='jet', alpha=0.8)
    axes[0, 1].set_title('Grad-CAM Attention Heatmap')
    axes[0, 1].set_ylabel('Frequency (scales)')
    axes[0, 1].set_xlabel('Time samples')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 叠加图
    axes[1, 0].imshow(cwt_features, aspect='auto', cmap='gray', alpha=0.7)
    im3 = axes[1, 0].imshow(cam, aspect='auto', cmap='jet', alpha=0.5)
    axes[1, 0].set_title('CWT + Grad-CAM Overlay')
    axes[1, 0].set_ylabel('Frequency (scales)')
    axes[1, 0].set_xlabel('Time samples')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 统计特征
    feature_names = ['Mean', 'Std', 'Max', 'Min', 'Energy', 'RMS', 'Skewness', 'Kurtosis']
    bars = axes[1, 1].bar(range(len(statistical_features)), statistical_features)
    axes[1, 1].set_title('Statistical Features')
    axes[1, 1].set_xticks(range(len(feature_names)))
    axes[1, 1].set_xticklabels(feature_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 为统计特征添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{statistical_features[i]:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/gradcam_azimuth_{azimuth}_sample_{sample_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM visualization saved to: {save_path}")
    
    # 打印关注区域分析
    max_pos = np.unravel_index(np.argmax(cam), cam.shape)
    print(f"Peak attention at: Frequency scale {max_pos[0]}, Time sample {max_pos[1]}")
    print(f"Attention intensity range: {cam.min():.3f} - {cam.max():.3f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualization for acoustic logging data')
    parser.add_argument('--azimuth', '-a', default='A', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       help='Azimuth direction (default: A)')
    parser.add_argument('--sample', '-s', type=int, default=None,
                       help='Sample index (default: auto-select high channeling sample)')
    parser.add_argument('--output', '-o', default='data/results',
                       help='Output directory (default: data/results)')
    
    args = parser.parse_args()
    
    print("=== Grad-CAM Visualization for Acoustic Logging ===")
    generate_gradcam_visualization(args.azimuth, args.sample, args.output)
    print("=== Visualization completed! ===")

if __name__ == "__main__":
    main() 