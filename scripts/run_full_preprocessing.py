#!/usr/bin/env python3
"""
完整数据集预处理脚本
处理所有方位的完整声波测井数据
"""

import sys
import os
import numpy as np
import pickle
from tqdm import tqdm
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import DataLoader
from src.data_processing.signal_processor import SignalProcessor
from src.data_processing.label_generator import LabelGenerator


def full_dataset_preprocessing():
    """完整数据集预处理"""
    print("=" * 70)
    print("开始完整数据集预处理")
    print("=" * 70)
    
    # 加载配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
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
    
    print(f"目标深度点数: {len(target_indices)}")
    
    # 4. 生成标签
    print("\n2. 生成标签...")
    all_labels = label_generator.generate_all_labels(
        actual_depth_r3, target_indices, cast_depth, cast_zc
    )
    
    # 5. 处理所有方位的完整数据
    print("\n3. 处理所有方位的完整数据...")
    
    processed_data = {}
    
    # 所有8个方位
    azimuths = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for azimuth in azimuths:
        print(f"\n处理方位 {azimuth}...")
        
        # 获取该方位的波形数据和标签
        side_wave_data = wave_data[azimuth]  # (1024, n_depth)
        side_labels = all_labels[azimuth]
        
        # 选择目标深度范围内的数据
        target_wave_data = side_wave_data[:, target_indices]  # (1024, n_target)
        
        # 转置以获得 (n_samples, 1024) 格式
        all_waveforms = target_wave_data.T  # (n_samples, 1024)
        
        print(f"  数据形状: {all_waveforms.shape}")
        print(f"  标签形状: {side_labels.shape}")
        print(f"  标签范围: {side_labels.min():.3f} - {side_labels.max():.3f}")
        print(f"  高窜槽样本数 (>0.5): {np.sum(side_labels > 0.5)}")
        
        # 分批处理以节省内存
        batch_size = 100  # 每批处理100个样本
        n_samples = all_waveforms.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        all_filtered_waveforms = []
        all_cwt_features = []
        all_stat_features = []
        
        print(f"  分 {n_batches} 批处理，每批 {batch_size} 个样本")
        
        for batch_idx in tqdm(range(n_batches), desc=f"处理方位{azimuth}"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            batch_waveforms = all_waveforms[start_idx:end_idx]
            
            # 信号处理
            filtered_waveforms, cwt_features, stat_features = signal_processor.process_batch_waveforms(
                batch_waveforms
            )
            
            all_filtered_waveforms.append(filtered_waveforms)
            all_cwt_features.append(cwt_features)
            all_stat_features.append(stat_features)
        
        # 合并所有批次的结果
        all_filtered_waveforms = np.concatenate(all_filtered_waveforms, axis=0)
        all_cwt_features = np.concatenate(all_cwt_features, axis=0)
        all_stat_features = np.concatenate(all_stat_features, axis=0)
        
        print(f"  完成处理:")
        print(f"    滤波波形形状: {all_filtered_waveforms.shape}")
        print(f"    CWT特征形状: {all_cwt_features.shape}")
        print(f"    统计特征形状: {all_stat_features.shape}")
        
        # 保存处理后的数据
        processed_data[azimuth] = {
            'waveforms': all_waveforms,
            'filtered_waveforms': all_filtered_waveforms,
            'cwt_features': all_cwt_features,
            'stat_features': all_stat_features,
            'labels': side_labels
        }
    
    # 6. 分析高窜槽样本
    print("\n4. 分析高窜槽样本...")
    high_channeling_indices = label_generator.get_high_channeling_samples(all_labels, threshold=0.5)
    
    for azimuth in azimuths:
        high_count = len(high_channeling_indices[azimuth])
        total_count = len(all_labels[azimuth])
        percentage = high_count / total_count * 100
        print(f"方位 {azimuth}: {high_count}/{total_count} 个高窜槽样本 ({percentage:.1f}%)")
    
    # 7. 保存处理结果
    print("\n5. 保存处理结果...")
    
    # 创建保存目录
    os.makedirs('data/processed', exist_ok=True)
    
    # 分别保存每个方位的数据以节省内存
    for azimuth in azimuths:
        azimuth_file = f'data/processed/full_processed_{azimuth}.pkl'
        with open(azimuth_file, 'wb') as f:
            pickle.dump(processed_data[azimuth], f)
        print(f"  保存方位 {azimuth} 数据到: {azimuth_file}")
    
    # 保存所有标签
    with open('data/processed/all_labels_full.pkl', 'wb') as f:
        pickle.dump(all_labels, f)
    print("  保存所有标签到: data/processed/all_labels_full.pkl")
    
    # 保存高窜槽样本索引
    with open('data/processed/high_channeling_indices_full.pkl', 'wb') as f:
        pickle.dump(high_channeling_indices, f)
    print("  保存高窜槽索引到: data/processed/high_channeling_indices_full.pkl")
    
    # 保存完整数据集的元数据
    metadata = {
        'n_samples_per_azimuth': {azimuth: processed_data[azimuth]['labels'].shape[0] for azimuth in azimuths},
        'total_samples': sum(processed_data[azimuth]['labels'].shape[0] for azimuth in azimuths),
        'cwt_feature_shape': processed_data['A']['cwt_features'].shape[1:],  # (scales, time_points)
        'stat_feature_dim': processed_data['A']['stat_features'].shape[1],
        'high_channeling_counts': {azimuth: len(high_channeling_indices[azimuth]) for azimuth in azimuths}
    }
    
    with open('data/processed/full_dataset_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("  保存元数据到: data/processed/full_dataset_metadata.pkl")
    
    print("\n完整数据集预处理完成！")
    print("结果保存在 data/processed/ 目录下")
    
    # 打印总结
    total_samples = metadata['total_samples']
    total_high_channeling = sum(metadata['high_channeling_counts'].values())
    
    print(f"\n数据集总结:")
    print(f"  总样本数: {total_samples:,}")
    print(f"  高窜槽样本数: {total_high_channeling:,} ({total_high_channeling/total_samples*100:.1f}%)")
    print(f"  CWT特征形状: {metadata['cwt_feature_shape']}")
    print(f"  统计特征维度: {metadata['stat_feature_dim']}")
    
    return metadata


if __name__ == "__main__":
    try:
        # 运行完整数据集预处理
        metadata = full_dataset_preprocessing()
        
        print("\n" + "=" * 70)
        print("完整数据集预处理成功完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 