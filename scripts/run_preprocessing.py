#!/usr/bin/env python3
"""
数据预处理脚本
测试数据加载、信号处理和标签生成的完整流程
"""

import sys
import os
import numpy as np
import pickle
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import DataLoader
from src.data_processing.signal_processor import SignalProcessor
from src.data_processing.label_generator import LabelGenerator


def test_small_sample_preprocessing():
    """测试小样本数据预处理"""
    print("=" * 70)
    print("开始小样本数据预处理测试")
    print("=" * 70)
    
    # 1. 初始化各个模块
    data_loader = DataLoader()
    signal_processor = SignalProcessor()
    label_generator = LabelGenerator()
    
    # 2. 加载数据
    print("\n1. 加载数据...")
    data = data_loader.load_all_data()
    
    # 3. 提取关键数据
    cast_data = data['cast']
    xsilmr_data = data['xsilmr03']
    
    cast_depth = cast_data['depth']
    cast_zc = cast_data['zc']
    
    wave_data = xsilmr_data['wave_data']
    actual_depth_r3 = xsilmr_data['actual_depth_r3']
    target_indices = xsilmr_data['target_indices']
    
    # 4. 生成标签
    print("\n2. 生成标签...")
    all_labels = label_generator.generate_all_labels(
        actual_depth_r3, target_indices, cast_depth, cast_zc
    )
    
    # 5. 小样本处理测试
    print("\n3. 小样本信号处理测试...")
    
    # 为每个方位选择少量样本进行处理
    sample_size = 50  # 每个方位50个样本
    processed_data = {}
    
    for side in ['A', 'B']:  # 先测试两个方位
        print(f"\n处理方位 {side}...")
        
        # 获取该方位的波形数据和标签
        side_wave_data = wave_data[side]  # (1024, n_depth)
        side_labels = all_labels[side]
        
        # 选择目标深度范围内的数据
        target_wave_data = side_wave_data[:, target_indices]  # (1024, n_target)
        
        # 随机选择样本
        n_samples = min(sample_size, target_wave_data.shape[1])
        sample_indices = np.random.choice(target_wave_data.shape[1], n_samples, replace=False)
        
        sample_waveforms = target_wave_data[:, sample_indices].T  # (n_samples, 1024)
        sample_labels = side_labels[sample_indices]
        
        print(f"  选择样本数: {n_samples}")
        print(f"  波形形状: {sample_waveforms.shape}")
        print(f"  标签形状: {sample_labels.shape}")
        print(f"  标签范围: {sample_labels.min():.3f} - {sample_labels.max():.3f}")
        
        # 信号处理
        print(f"  进行信号处理...")
        filtered_waveforms, cwt_features, stat_features = signal_processor.process_batch_waveforms(
            sample_waveforms
        )
        
        print(f"  滤波波形形状: {filtered_waveforms.shape}")
        print(f"  CWT特征形状: {cwt_features.shape}")
        print(f"  统计特征形状: {stat_features.shape}")
        
        # 保存处理后的数据
        processed_data[side] = {
            'waveforms': sample_waveforms,
            'filtered_waveforms': filtered_waveforms,
            'cwt_features': cwt_features,
            'stat_features': stat_features,
            'labels': sample_labels,
            'sample_indices': sample_indices
        }
    
    # 6. 分析高窜槽样本
    print("\n4. 分析高窜槽样本...")
    high_channeling_indices = label_generator.get_high_channeling_samples(all_labels, threshold=0.5)
    
    # 7. 保存处理结果
    print("\n5. 保存处理结果...")
    
    # 创建保存目录
    os.makedirs('data/processed', exist_ok=True)
    
    # 保存小样本处理结果
    with open('data/processed/small_sample_processed.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    # 保存所有标签
    with open('data/processed/all_labels.pkl', 'wb') as f:
        pickle.dump(all_labels, f)
    
    # 保存高窜槽样本索引
    with open('data/processed/high_channeling_indices.pkl', 'wb') as f:
        pickle.dump(high_channeling_indices, f)
    
    print("小样本预处理测试完成！")
    print(f"结果保存在 data/processed/ 目录下")
    
    return processed_data, all_labels, high_channeling_indices


if __name__ == "__main__":
    try:
        # 运行小样本预处理测试
        processed_data, all_labels, high_channeling_indices = test_small_sample_preprocessing()
        
        print("\n" + "=" * 70)
        print("数据预处理测试成功完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 