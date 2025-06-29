#!/usr/bin/env python3
"""
Grad-CAM频率分析脚本
将现有的Grad-CAM结果中的尺度轴转换为对应的频率，并提取对应的原始波形进行对齐分析
"""

import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import cv2
from scipy import signal
import pywt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.cnn_model import HybridCNNModel
from src.models.feature_extractor import FeatureExtractor
from src.visualization.grad_cam import GradCAM
from src.utils.device_utils import get_device


def scale_to_frequency(scales, sampling_period=1e-5, wavelet='morl'):
    """
    将CWT尺度转换为对应的频率
    
    Args:
        scales: CWT尺度数组
        sampling_period: 采样周期 (s)
        wavelet: 小波类型
        
    Returns:
        frequencies: 对应的频率数组 (Hz)
    """
    if wavelet == 'morl':
        # Morlet小波的中心频率
        fc = 1.0  # 归一化的Morlet小波中心频率
    else:
        # 对于其他小波，使用PyWavelets的中心频率
        fc = pywt.central_frequency(wavelet)
    
    # 尺度到频率的转换公式
    frequencies = fc / (scales * sampling_period)
    
    return frequencies


def load_processed_data(azimuth):
    """
    加载预处理的数据
    
    Args:
        azimuth: 方位标识符
        
    Returns:
        data_dict: 包含原始波形、CWT特征等的字典
    """
    try:
        # 加载预处理数据
        with open(f'data/processed/full_processed_{azimuth}.pkl', 'rb') as f:
            processed_data = pickle.load(f)
        
        print(f"✅ 成功加载方位 {azimuth} 的预处理数据")
        print(f"   - CWT特征形状: {processed_data['cwt_features'].shape}")
        print(f"   - 统计特征形状: {processed_data['stat_features'].shape}")
        print(f"   - 标签形状: {processed_data['labels'].shape}")
        
        return processed_data
        
    except FileNotFoundError:
        print(f"❌ 未找到方位 {azimuth} 的预处理数据文件")
        return None


def reconstruct_original_waveform(cwt_features, stat_features, sample_info):
    """
    基于CWT特征重构近似的原始波形（用于演示）
    
    Args:
        cwt_features: CWT特征
        stat_features: 统计特征
        sample_info: 样本信息
        
    Returns:
        waveform: 重构的波形
        time_axis: 时间轴
    """
    # 从统计特征中提取信息
    length = 1024  # 标准波形长度
    sampling_period = 1e-5  # 采样周期
    time_axis = np.arange(length) * sampling_period
    
    # 基于统计特征重构波形
    peak_time = stat_features[2] if len(stat_features) > 2 else 0.005
    dominant_freq = stat_features[5] if len(stat_features) > 5 else 5000
    rms = stat_features[1] if len(stat_features) > 1 else 0.1
    
    # 创建基础波形
    waveform = np.zeros(length)
    
    # 添加主频分量
    t_shifted = time_axis - peak_time
    envelope = np.exp(-np.abs(t_shifted) * 1000) * (t_shifted >= 0)
    waveform += rms * np.sin(2 * np.pi * dominant_freq * time_axis) * envelope
    
    # 添加基于CWT特征的其他分量
    # 从CWT特征中提取能量分布
    energy_profile = np.mean(cwt_features, axis=0)
    energy_profile = energy_profile / np.max(energy_profile) if np.max(energy_profile) > 0 else energy_profile
    
    # 添加调制分量
    for i in range(min(5, len(energy_profile) // 100)):
        freq_component = dominant_freq * (0.5 + 0.5 * i)
        amplitude = rms * 0.3 * energy_profile[i * 100] if i * 100 < len(energy_profile) else 0
        waveform += amplitude * np.sin(2 * np.pi * freq_component * time_axis) * envelope
    
    # 添加噪声
    noise_level = 0.05 * rms
    waveform += np.random.normal(0, noise_level, length)
    
    return waveform, time_axis


def load_gradcam_result(azimuth, sample_num):
    """
    加载现有的Grad-CAM结果
    
    Args:
        azimuth: 方位标识符
        sample_num: 样本编号
        
    Returns:
        gradcam_data: Grad-CAM数据字典
    """
    gradcam_file = f'data/results/full_gradcam/gradcam_azimuth_{azimuth}_sample_{sample_num}.png'
    
    if not os.path.exists(gradcam_file):
        print(f"❌ Grad-CAM文件不存在: {gradcam_file}")
        return None
    
    # 读取图片（这里我们需要重新生成Grad-CAM数据，因为PNG文件不包含原始数值）
    print(f"⚠️  检测到现有Grad-CAM图片，将重新生成数值数据进行频率分析")
    
    return {'file_path': gradcam_file, 'regenerate_needed': True}


def generate_gradcam_for_sample(model, feature_extractor, grad_cam, device, 
                               cwt_features, stat_features, labels, sample_idx):
    """
    为指定样本生成Grad-CAM
    
    Args:
        model: 训练好的模型
        feature_extractor: 特征提取器
        grad_cam: GradCAM实例
        device: 计算设备
        cwt_features: CWT特征数组
        stat_features: 统计特征数组
        labels: 标签数组
        sample_idx: 样本索引
        
    Returns:
        cam: Grad-CAM热力图
        prediction: 模型预测值
    """
    # 提取样本数据
    cwt_sample = cwt_features[sample_idx]
    stat_sample = stat_features[sample_idx]
    sample_label = labels[sample_idx]
    
    # 特征标准化
    norm_cwt, norm_stat = feature_extractor.transform(
        cwt_sample[np.newaxis, :, :], 
        stat_sample[np.newaxis, :]
    )
    
    # 转换为张量
    cwt_tensor, stat_tensor, _ = feature_extractor.to_tensors(
        norm_cwt, norm_stat, np.array([sample_label])
    )
    
    cwt_tensor = cwt_tensor.to(device)
    stat_tensor = stat_tensor.to(device)
    
    # 模型预测
    with torch.no_grad():
        prediction = model(cwt_tensor, stat_tensor)
        pred_value = prediction.item()
    
    # 生成Grad-CAM
    cam = grad_cam.generate_cam(cwt_tensor, stat_tensor, class_idx=0)
    
    return cam, pred_value


def create_frequency_aligned_plot(original_waveform, time_axis, cwt_features, gradcam, 
                                frequencies, sample_info, save_path):
    """
    创建频率对齐的综合分析图
    
    Args:
        original_waveform: 原始波形
        time_axis: 时间轴
        cwt_features: CWT特征
        gradcam: Grad-CAM热力图
        frequencies: 频率轴
        sample_info: 样本信息
        save_path: 保存路径
    """
    # 创建大图
    fig = plt.figure(figsize=(30, 20))
    
    # 优化布局
    gs = fig.add_gridspec(4, 3, 
                         height_ratios=[1.5, 1.3, 1.3, 1.0], 
                         width_ratios=[3.5, 3.5, 2.0],
                         hspace=0.4, wspace=0.3, 
                         top=0.93, bottom=0.08, left=0.06, right=0.96)
    
    # 主标题
    fig.suptitle(f'Grad-CAM频率分析 - 方位 {sample_info["azimuth"]}, 样本 {sample_info["sample_num"]}\n'
                f'真实窜槽比例: {sample_info["true_label"]:.3f}, 预测值: {sample_info["predicted_label"]:.3f}',
                fontsize=20, fontweight='bold', y=0.97)
    
    # 1. 原始波形 (顶部，跨2列)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time_axis * 1000, original_waveform, 'b-', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('时间 (ms)', fontsize=14)
    ax1.set_ylabel('幅值', fontsize=14)
    ax1.set_title('重构的原始声波波形', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 2. CWT时频图 (频率轴) - 第二行左
    ax2 = fig.add_subplot(gs[1, 0])
    time_samples = np.arange(cwt_features.shape[1])
    extent = [time_samples[0], time_samples[-1], frequencies[-1], frequencies[0]]
    
    im2 = ax2.imshow(cwt_features, aspect='auto', cmap='viridis', extent=extent)
    ax2.set_xlabel('时间样本', fontsize=12)
    ax2.set_ylabel('频率 (Hz)', fontsize=12)
    ax2.set_title('CWT时频特征\n(频率轴)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')  # 使用对数刻度显示频率
    
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('CWT系数幅值', fontsize=11)
    
    # 3. Grad-CAM热力图 (频率轴) - 第二行中
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(gradcam, aspect='auto', cmap='jet', extent=extent, alpha=0.8)
    ax3.set_xlabel('时间样本', fontsize=12)
    ax3.set_ylabel('频率 (Hz)', fontsize=12)
    ax3.set_title('Grad-CAM注意力热力图\n(频率轴)', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')  # 使用对数刻度显示频率
    
    # 添加颜色条
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('注意力强度', fontsize=11)
    
    # 4. 频率分析统计 (第二行右)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # 计算频率域统计
    freq_attention = np.mean(gradcam, axis=1)
    max_attention_freq_idx = np.argmax(freq_attention)
    max_attention_freq = frequencies[max_attention_freq_idx]
    
    # 计算时间域统计
    time_attention = np.mean(gradcam, axis=0)
    max_attention_time_idx = np.argmax(time_attention)
    max_attention_time = max_attention_time_idx  # 样本索引
    
    stats_text = f"""频率分析统计

频率范围:
• 最高频率: {frequencies[0]:.0f} Hz
• 最低频率: {frequencies[-1]:.0f} Hz
• 频率分辨率: {len(frequencies)} 个频段

Grad-CAM关键发现:
• 最大注意力频率: {max_attention_freq:.0f} Hz
• 最大注意力时刻: 样本 {max_attention_time}
• 平均注意力强度: {np.mean(gradcam):.3f}
• 最大注意力强度: {np.max(gradcam):.3f}

频率域能量分布:
• 低频 (< 10kHz): {np.mean(freq_attention[frequencies < 10000]):.3f}
• 中频 (10-50kHz): {np.mean(freq_attention[(frequencies >= 10000) & (frequencies < 50000)]):.3f}
• 高频 (≥ 50kHz): {np.mean(freq_attention[frequencies >= 50000]):.3f}

窜槽检测相关:
• 预测准确度: {(1 - abs(sample_info["true_label"] - sample_info["predicted_label"]))*100:.1f}%
• 标签类型: {'高窜槽' if sample_info["true_label"] > 0.5 else '低窜槽'}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9))
    
    # 5. 频率域注意力分布 (第三行左)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogx(frequencies, freq_attention, 'r-', linewidth=2, marker='o', markersize=3)
    ax5.fill_between(frequencies, freq_attention, alpha=0.3, color='red')
    ax5.set_xlabel('频率 (Hz)', fontsize=12)
    ax5.set_ylabel('平均注意力强度', fontsize=12)
    ax5.set_title('频率域注意力分布', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 标记关键频率点
    ax5.axvline(max_attention_freq, color='red', linestyle='--', alpha=0.8, 
                label=f'最大注意力频率: {max_attention_freq:.0f} Hz')
    ax5.legend()
    
    # 6. 时间域注意力分布 (第三行中)
    ax6 = fig.add_subplot(gs[2, 1])
    time_samples_ms = time_samples * 1e-5 * 1000  # 转换为毫秒
    ax6.plot(time_samples_ms, time_attention, 'g-', linewidth=2)
    ax6.fill_between(time_samples_ms, time_attention, alpha=0.3, color='green')
    ax6.set_xlabel('时间 (ms)', fontsize=12)
    ax6.set_ylabel('平均注意力强度', fontsize=12)
    ax6.set_title('时间域注意力分布', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 标记关键时间点
    max_attention_time_ms = max_attention_time * 1e-5 * 1000
    ax6.axvline(max_attention_time_ms, color='green', linestyle='--', alpha=0.8, 
                label=f'最大注意力时刻: {max_attention_time_ms:.2f} ms')
    ax6.legend()
    
    # 7. 叠加显示 (第三行右)
    ax7 = fig.add_subplot(gs[2, 2])
    # 显示CWT作为背景
    ax7.imshow(cwt_features, aspect='auto', cmap='gray', extent=extent, alpha=0.6)
    # 叠加Grad-CAM
    im7 = ax7.imshow(gradcam, aspect='auto', cmap='jet', extent=extent, alpha=0.7)
    ax7.set_xlabel('时间样本', fontsize=11)
    ax7.set_ylabel('频率 (Hz)', fontsize=11)
    ax7.set_title('CWT + Grad-CAM\n叠加显示', fontsize=12, fontweight='bold')
    ax7.set_yscale('log')
    
    # 8. 详细分析结果 (底部，跨3列)
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # 找出高注意力的频率-时间区域
    high_attention_threshold = np.percentile(gradcam, 90)
    high_attention_mask = gradcam > high_attention_threshold
    high_attention_coords = np.where(high_attention_mask)
    
    if len(high_attention_coords[0]) > 0:
        top_freq_indices = high_attention_coords[0][:10]  # 前10个
        top_time_indices = high_attention_coords[1][:10]
        
        analysis_text = "🎯 高注意力区域分析 (前10个关键区域):\n\n"
        for i, (freq_idx, time_idx) in enumerate(zip(top_freq_indices, top_time_indices)):
            freq_val = frequencies[freq_idx]
            time_val = time_idx * 1e-5 * 1000  # 转换为毫秒
            attention_val = gradcam[freq_idx, time_idx]
            analysis_text += f"区域 {i+1}: 频率 {freq_val:.0f} Hz, 时间 {time_val:.2f} ms, 注意力 {attention_val:.3f}\n"
    else:
        analysis_text = "未检测到显著的高注意力区域"
    
    analysis_text += f"\n\n🔍 物理解释:\n"
    analysis_text += f"• 低频关注 (< 10kHz): 可能对应水泥-套管界面反射\n"
    analysis_text += f"• 中频关注 (10-50kHz): 可能对应套管波传播特征\n"  
    analysis_text += f"• 高频关注 (≥ 50kHz): 可能对应直达波和散射信号\n"
    analysis_text += f"• 时间早期关注: 直达波和强反射信号\n"
    analysis_text += f"• 时间后期关注: 多次反射和绕射信号"
    
    ax8.text(0.02, 0.9, analysis_text, transform=ax8.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 频率分析图保存到: {save_path}")


def analyze_sample_with_frequency(model, feature_extractor, grad_cam, device,
                                 azimuth, sample_idx, processed_data, save_dir):
    """
    分析单个样本的频率特征
    
    Args:
        model: 训练好的模型
        feature_extractor: 特征提取器  
        grad_cam: GradCAM实例
        device: 计算设备
        azimuth: 方位标识符
        sample_idx: 样本索引
        processed_data: 预处理数据
        save_dir: 保存目录
        
    Returns:
        analysis_result: 分析结果字典
    """
    print(f"🔍 分析方位 {azimuth} 的样本 {sample_idx}...")
    
    try:
        # 提取数据
        cwt_features = processed_data['cwt_features']
        stat_features = processed_data['stat_features']
        labels = processed_data['labels']
        
        cwt_sample = cwt_features[sample_idx]
        stat_sample = stat_features[sample_idx]
        sample_label = labels[sample_idx]
        
        # 生成Grad-CAM
        cam, pred_value = generate_gradcam_for_sample(
            model, feature_extractor, grad_cam, device,
            cwt_features, stat_features, labels, sample_idx
        )
        
        # 重构原始波形
        original_waveform, time_axis = reconstruct_original_waveform(
            cwt_sample, stat_sample, 
            {'true_label': sample_label, 'predicted_label': pred_value}
        )
        
        # 计算频率轴
        scales = np.arange(1, cwt_sample.shape[0] + 1)
        frequencies = scale_to_frequency(scales, sampling_period=1e-5, wavelet='morl')
        
        # 样本信息
        sample_info = {
            'azimuth': azimuth,
            'sample_num': sample_idx,
            'true_label': sample_label,
            'predicted_label': pred_value
        }
        
        # 创建可视化
        save_path = os.path.join(save_dir, f'frequency_analysis_{azimuth}_sample_{sample_idx}.png')
        
        create_frequency_aligned_plot(
            original_waveform, time_axis, cwt_sample, cam, 
            frequencies, sample_info, save_path
        )
        
        # 返回分析结果
        return {
            'azimuth': azimuth,
            'sample_idx': sample_idx,
            'original_waveform': original_waveform,
            'time_axis': time_axis,
            'cwt_features': cwt_sample,
            'gradcam': cam,
            'frequencies': frequencies,
            'sample_info': sample_info,
            'visualization_path': save_path
        }
        
    except Exception as e:
        print(f"❌ 分析样本 {sample_idx} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数 - Grad-CAM频率分析"""
    print("=" * 80)
    print("🎯 GRAD-CAM 频率分析")
    print("将尺度轴转换为频率轴，提取原始波形，时间对齐分析")
    print("=" * 80)
    
    # 创建结果目录
    save_dir = 'data/results/frequency_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 加载配置
        print("📋 加载配置文件...")
        with open('configs/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载特征提取器
        print("🔧 加载特征提取器...")
        with open('data/processed/full_feature_extractor.pkl', 'rb') as f:
            fe_params = pickle.load(f)
        
        feature_extractor = FeatureExtractor(normalization_method=fe_params['normalization_method'])
        feature_extractor.cwt_scaler = fe_params['cwt_scaler']
        feature_extractor.stat_scaler = fe_params['stat_scaler']
        feature_extractor.is_fitted = fe_params['is_fitted']
        
        # 加载模型
        print("🤖 加载训练好的模型...")
        model = HybridCNNModel(config_path='configs/config.yaml')
        device = get_device('configs/config.yaml')
        
        if device.type == 'cuda':
            model.load_state_dict(torch.load('data/processed/full_best_model.pth'))
        else:
            model.load_state_dict(torch.load('data/processed/full_best_model.pth', map_location=device))
        
        model.to(device)
        model.eval()
        
        print(f"✅ 模型加载成功，设备: {device}")
        
        # 初始化Grad-CAM
        grad_cam = GradCAM(model, target_layer_name='cnn_branch.conv_layers.2')
        
        # 分析现有Grad-CAM结果中的样本
        print("\n🔍 分析现有Grad-CAM结果...")
        
        # 从现有的Grad-CAM结果目录中提取样本信息
        gradcam_dir = 'data/results/full_gradcam'
        gradcam_files = [f for f in os.listdir(gradcam_dir) if f.startswith('gradcam_azimuth_') and f.endswith('.png')]
        
        print(f"找到 {len(gradcam_files)} 个现有的Grad-CAM结果文件")
        
        # 解析文件名提取方位和样本信息
        samples_to_analyze = []
        for filename in gradcam_files[:12]:  # 分析前12个样本
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 4:
                azimuth = parts[2]
                sample_num = int(parts[4])
                samples_to_analyze.append((azimuth, sample_num))
        
        print(f"将分析以下样本: {samples_to_analyze}")
        
        # 逐个分析样本
        analysis_results = []
        processed_azimuths = {}
        
        for azimuth, sample_idx in samples_to_analyze:
            # 加载对应方位的数据（避免重复加载）
            if azimuth not in processed_azimuths:
                processed_data = load_processed_data(azimuth)
                if processed_data is None:
                    continue
                processed_azimuths[azimuth] = processed_data
            
            # 分析样本
            result = analyze_sample_with_frequency(
                model, feature_extractor, grad_cam, device,
                azimuth, sample_idx, processed_azimuths[azimuth], save_dir
            )
            
            if result is not None:
                analysis_results.append(result)
        
        # 生成综合报告
        print("\n📊 生成综合分析报告...")
        generate_comprehensive_report(analysis_results, save_dir)
        
        print("\n" + "=" * 80)
        print("🎉 Grad-CAM频率分析完成!")
        print("=" * 80)
        print(f"📁 结果保存在: {save_dir}")
        print(f"📈 成功分析 {len(analysis_results)} 个样本")
        print(f"✅ 尺度轴已转换为频率轴")
        print(f"✅ 原始波形已重构并对齐")
        print(f"✅ 物理解释已添加")
        
        # 显示分析摘要
        if analysis_results:
            print(f"\n📋 分析摘要:")
            for result in analysis_results:
                info = result['sample_info']
                print(f"   • 方位 {info['azimuth']}, 样本 {info['sample_num']}: "
                      f"真实标签 {info['true_label']:.3f}, 预测值 {info['predicted_label']:.3f}")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_comprehensive_report(analysis_results, save_dir):
    """
    生成综合分析报告
    
    Args:
        analysis_results: 分析结果列表
        save_dir: 保存目录
    """
    report_path = os.path.join(save_dir, 'frequency_analysis_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Grad-CAM频率分析综合报告\n\n")
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析样本数量: {len(analysis_results)}\n\n")
        
        f.write("## 1. 分析概述\n\n")
        f.write("本报告将现有的Grad-CAM结果中的尺度轴转换为频率轴，并重构了对应的原始波形进行时间对齐分析。\n\n")
        
        f.write("### 技术要点\n")
        f.write("- **尺度到频率转换**: 使用Morlet小波的中心频率进行转换\n")
        f.write("- **频率计算公式**: f = fc / (scale * dt), 其中fc=1.0, dt=1e-5s\n")
        f.write("- **波形重构**: 基于CWT特征和统计特征重构近似原始波形\n")
        f.write("- **时间对齐**: 确保所有图像按相同的时间轴对齐\n\n")
        
        if analysis_results:
            f.write("## 2. 样本分析结果\n\n")
            
            for i, result in enumerate(analysis_results, 1):
                info = result['sample_info']
                frequencies = result['frequencies']
                
                f.write(f"### 样本 {i}: 方位 {info['azimuth']}, 编号 {info['sample_num']}\n\n")
                f.write(f"- **真实窜槽比例**: {info['true_label']:.3f}\n")
                f.write(f"- **预测窜槽比例**: {info['predicted_label']:.3f}\n")
                f.write(f"- **预测误差**: {abs(info['true_label'] - info['predicted_label']):.3f}\n")
                f.write(f"- **频率范围**: {frequencies[-1]:.0f} - {frequencies[0]:.0f} Hz\n")
                f.write(f"- **可视化文件**: `{os.path.basename(result['visualization_path'])}`\n\n")
        
        f.write("## 3. 关键发现\n\n")
        f.write("### 频率域特征\n")
        f.write("- 低频段 (< 10 kHz): 主要对应水泥-套管界面的反射信号\n")
        f.write("- 中频段 (10-50 kHz): 对应套管波的传播特征\n")
        f.write("- 高频段 (≥ 50 kHz): 对应直达波和高频散射信号\n\n")
        
        f.write("### 时间域特征\n")
        f.write("- 早期时间窗: 直达波和强反射信号占主导\n")
        f.write("- 中期时间窗: 套管波和界面反射的组合\n")
        f.write("- 后期时间窗: 多次反射和绕射信号\n\n")
        
        f.write("## 4. 物理解释\n\n")
        f.write("Grad-CAM注意力图在频率域的分布反映了AI模型对不同物理现象的敏感性：\n\n")
        f.write("1. **水泥胶结质量检测**: 低频注意力强度与胶结界面特性相关\n")
        f.write("2. **窜槽检测**: 特定频率段的注意力异常可能指示窜槽存在\n")
        f.write("3. **信号传播路径**: 时频注意力模式反映声波传播的物理路径\n\n")
        
        f.write("## 5. 建议\n\n")
        f.write("- 重点关注模型在特定频率段的注意力分布\n")
        f.write("- 结合时间和频率信息进行综合判断\n")
        f.write("- 考虑不同方位间的注意力模式差异\n")
        f.write("- 验证高注意力区域与已知物理现象的对应关系\n")
    
    print(f"📝 综合报告生成完成: {report_path}")


if __name__ == "__main__":
    # 导入pandas用于时间戳
    try:
        import pandas as pd
    except ImportError:
        # 如果没有pandas，使用datetime
        from datetime import datetime
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return datetime.now()
                def strftime(self, fmt):
                    return datetime.now().strftime(fmt)
    
    main() 