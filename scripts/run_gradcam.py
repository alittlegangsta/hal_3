#!/usr/bin/env python3
"""
Grad-CAM可视化运行脚本
用于调试和测试Grad-CAM可视化功能
"""

import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.cnn_model import HybridCNNModel
from src.models.feature_extractor import FeatureExtractor
from src.visualization.grad_cam import GradCAM
from src.utils.config import load_config
from src.utils.device_utils import get_device

def load_model_and_data():
    """加载训练好的模型和测试数据"""
    print("Loading model and data...")
    
    # 加载配置
    config = load_config('configs/config.yaml')
    
    # 加载特征提取器
    with open('data/processed/feature_extractor.pkl', 'rb') as f:
        feature_extractor = pickle.load(f)
    
    # 加载预处理数据
    with open('data/processed/small_sample_processed.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # 加载标签
    with open('data/processed/all_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    
    # 加载高窜槽样本索引
    with open('data/processed/high_channeling_indices.pkl', 'rb') as f:
        high_indices = pickle.load(f)
    
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
    print(f"Processed data shape: {len(processed_data)}")
    print(f"High channeling samples: {len(high_indices)}")
    
    return model, processed_data, labels, high_indices, feature_extractor, device

def test_gradcam_on_samples(model, processed_data, labels, high_indices, feature_extractor, device, num_samples=5):
    """测试Grad-CAM在几个样本上的表现"""
    print(f"\nTesting Grad-CAM on {num_samples} high channeling samples...")
    
    # 创建结果目录
    os.makedirs('data/results/gradcam_debug', exist_ok=True)
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model, target_layer_name='cnn_branch.conv_layers.2')
    
    # 选择几个高窜槽样本进行测试 - 从方位A开始
    azimuth = 'A'  # 选择方位A
    azimuth_data = processed_data[azimuth]
    azimuth_high_indices = high_indices[azimuth]
    
    # 从预处理数据中直接获取特征
    cwt_features_all = azimuth_data['cwt_features']
    stat_features_all = azimuth_data['stat_features']
    labels_all = azimuth_data['labels']
    
    # 过滤出在数据范围内的高窜槽索引
    data_size = len(cwt_features_all)
    valid_indices = [idx for idx in azimuth_high_indices if idx < data_size]
    
    if len(valid_indices) == 0:
        # 如果没有有效的高窜槽索引，从标签中找高窜槽样本
        high_label_indices = [i for i, label in enumerate(labels_all) if label > 0.5]
        test_indices = high_label_indices[:num_samples] if len(high_label_indices) >= num_samples else high_label_indices
        print(f"No valid high indices in range. Using samples with high labels: {test_indices}")
    else:
        test_indices = valid_indices[:num_samples] if len(valid_indices) >= num_samples else valid_indices
        print(f"Using valid high channeling indices: {test_indices}")
    
    for i, idx in enumerate(test_indices):
        try:
            print(f"\nProcessing sample {i+1}/{len(test_indices)} (azimuth: {azimuth}, index: {idx})")
            
            # 获取样本特征
            cwt_features = cwt_features_all[idx]
            statistical_features = stat_features_all[idx]
            sample_label = labels_all[idx]
            
            print(f"Sample label (channeling ratio): {sample_label:.3f}")
            
            print(f"CWT features shape: {cwt_features.shape}")
            print(f"Statistical features shape: {statistical_features.shape}")
            
            # 转换为tensor，CWT需要添加通道维度
            cwt_tensor = torch.FloatTensor(cwt_features).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 64, 1024)
            stat_tensor = torch.FloatTensor(statistical_features).unsqueeze(0).to(device)
            
            # 模型预测
            with torch.no_grad():
                prediction = model(cwt_tensor, stat_tensor)
                pred_value = prediction.item()
            
            print(f"Model prediction: {pred_value:.3f}")
            
            # 生成Grad-CAM
            print("Generating Grad-CAM...")
            cam = grad_cam.generate_cam(cwt_tensor, stat_tensor, class_idx=0)
            
            print(f"CAM shape: {cam.shape}")
            print(f"CAM min/max: {cam.min():.3f}/{cam.max():.3f}")
            
            # 可视化并保存结果
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Sample {idx} - Label: {sample_label:.3f}, Prediction: {pred_value:.3f}', fontsize=14)
            
            # 原始CWT特征
            im1 = axes[0, 0].imshow(cwt_features, aspect='auto', cmap='viridis')
            axes[0, 0].set_title('Original CWT Features')
            axes[0, 0].set_ylabel('Frequency (scales)')
            axes[0, 0].set_xlabel('Time samples')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Grad-CAM热力图
            im2 = axes[0, 1].imshow(cam, aspect='auto', cmap='jet', alpha=0.8)
            axes[0, 1].set_title('Grad-CAM Heatmap')
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
            axes[1, 1].bar(range(len(statistical_features)), statistical_features)
            axes[1, 1].set_title('Statistical Features')
            axes[1, 1].set_xticks(range(len(feature_names)))
            axes[1, 1].set_xticklabels(feature_names, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = f'data/results/gradcam_debug/sample_{idx}_gradcam.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization to: {save_path}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nGrad-CAM debugging completed. Results saved in data/results/gradcam_debug/")

def analyze_gradcam_patterns(model, processed_data, labels, high_indices, feature_extractor, device):
    """分析Grad-CAM模式，统计关注区域"""
    print("\nAnalyzing Grad-CAM attention patterns...")
    
    grad_cam = GradCAM(model, target_layer_name='cnn_branch.conv_layers.2')
    
    all_cams = []
    all_labels = []
    
    # 处理更多样本进行统计分析 - 使用方位A的样本
    azimuth = 'A'
    azimuth_data = processed_data[azimuth]
    azimuth_high_indices = high_indices[azimuth]
    
    # 从预处理数据中直接获取特征
    cwt_features_all = azimuth_data['cwt_features']
    stat_features_all = azimuth_data['stat_features']
    labels_all = azimuth_data['labels']
    
    # 过滤出在数据范围内的高窜槽索引
    data_size = len(cwt_features_all)
    valid_indices = [idx for idx in azimuth_high_indices if idx < data_size]
    
    if len(valid_indices) == 0:
        # 如果没有有效的高窜槽索引，从标签中找高窜槽样本
        high_label_indices = [i for i, label in enumerate(labels_all) if label > 0.5]
        test_indices = high_label_indices[:20] if len(high_label_indices) >= 20 else high_label_indices
    else:
        test_indices = valid_indices[:20] if len(valid_indices) >= 20 else valid_indices
    
    for idx in test_indices:
        try:
            # 获取样本特征
            cwt_features = cwt_features_all[idx]
            statistical_features = stat_features_all[idx]
            sample_label = labels_all[idx]
            
            # 转换为tensor，CWT需要添加通道维度
            cwt_tensor = torch.FloatTensor(cwt_features).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 64, 1024)
            stat_tensor = torch.FloatTensor(statistical_features).unsqueeze(0).to(device)
            
            # 生成Grad-CAM
            cam = grad_cam.generate_cam(cwt_tensor, stat_tensor, class_idx=0)
            
            all_cams.append(cam)
            all_labels.append(sample_label)
            
        except Exception as e:
            print(f"Error processing sample {idx} for pattern analysis: {str(e)}")
            continue
    
    if all_cams:
        # 计算平均CAM
        mean_cam = np.mean(all_cams, axis=0)
        std_cam = np.std(all_cams, axis=0)
        
        # 保存统计结果
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 平均关注图
        im1 = axes[0].imshow(mean_cam, aspect='auto', cmap='jet')
        axes[0].set_title('Average Grad-CAM Pattern')
        axes[0].set_ylabel('Frequency (scales)')
        axes[0].set_xlabel('Time samples')
        plt.colorbar(im1, ax=axes[0])
        
        # 标准差图
        im2 = axes[1].imshow(std_cam, aspect='auto', cmap='viridis')
        axes[1].set_title('Grad-CAM Standard Deviation')
        axes[1].set_ylabel('Frequency (scales)')
        axes[1].set_xlabel('Time samples')
        plt.colorbar(im2, ax=axes[1])
        
        # 关注强度统计
        cam_max_positions = []
        for cam in all_cams:
            max_pos = np.unravel_index(np.argmax(cam), cam.shape)
            cam_max_positions.append(max_pos)
        
        freq_positions = [pos[0] for pos in cam_max_positions]
        time_positions = [pos[1] for pos in cam_max_positions]
        
        axes[2].scatter(time_positions, freq_positions, c=all_labels, cmap='RdYlBu', s=50, alpha=0.7)
        axes[2].set_xlabel('Time samples')
        axes[2].set_ylabel('Frequency (scales)')
        axes[2].set_title('Peak Attention Positions')
        cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
        cbar.set_label('Channeling Ratio')
        
        plt.tight_layout()
        save_path = 'data/results/gradcam_debug/pattern_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Pattern analysis saved to: {save_path}")
        
        # 打印统计信息
        print(f"\nStatistical Analysis:")
        print(f"Number of samples analyzed: {len(all_cams)}")
        print(f"Average channeling ratio: {np.mean(all_labels):.3f}")
        print(f"Peak attention - Freq range: {min(freq_positions)}-{max(freq_positions)}")
        print(f"Peak attention - Time range: {min(time_positions)}-{max(time_positions)}")

def main():
    """主函数"""
    print("=== Grad-CAM Visualization Debugging ===")
    
    try:
        # 加载模型和数据
        model, processed_data, labels, high_indices, feature_extractor, device = load_model_and_data()
        
        # 测试Grad-CAM在几个样本上
        test_gradcam_on_samples(model, processed_data, labels, high_indices, feature_extractor, device)
        
        # 分析Grad-CAM模式
        analyze_gradcam_patterns(model, processed_data, labels, high_indices, feature_extractor, device)
        
        print("\n=== Debugging completed successfully! ===")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 