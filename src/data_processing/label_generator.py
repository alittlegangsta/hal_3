"""
标签生成器模块
负责根据CAST超声数据为第3号接收器的每个波形生成精确的窜槽比例标签
"""

import numpy as np
from typing import Tuple, Dict, List
import yaml


class LabelGenerator:
    """标签生成器类"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化标签生成器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.label_config = self.config['processing']['label_generation']
        
        # 方位角扇区映射
        self.azimuth_sectors = self.label_config['azimuth_sectors']
        
    def get_azimuth_indices(self, azimuth_sector: Tuple[float, float]) -> np.ndarray:
        """
        获取方位角扇区对应的索引
        
        Args:
            azimuth_sector: 方位角扇区范围 (start_angle, end_angle)
            
        Returns:
            azimuth_indices: 对应的方位角索引数组
        """
        start_angle, end_angle = azimuth_sector
        
        # CAST数据中的角度: 0-359度，每2度一个点，共180个点
        angles = np.arange(0, 360, 2)
        
        # 处理跨越0度的情况
        if start_angle < 0:
            # 例如 [-22.5, 22.5] -> [337.5, 360) 和 [0, 22.5]
            mask1 = angles >= (360 + start_angle)
            mask2 = angles <= end_angle
            mask = mask1 | mask2
        elif end_angle > 360:
            # 例如 [337.5, 382.5] -> [337.5, 360) 和 [0, 22.5]
            mask1 = angles >= start_angle
            mask2 = angles <= (end_angle - 360)
            mask = mask1 | mask2
        else:
            # 正常情况
            mask = (angles >= start_angle) & (angles <= end_angle)
        
        azimuth_indices = np.where(mask)[0]
        return azimuth_indices
    
    def generate_label_for_waveform(self, 
                                   depth: float, 
                                   side: str, 
                                   cast_depth: np.ndarray, 
                                   cast_zc: np.ndarray) -> float:
        """
        为单个波形生成标签
        
        Args:
            depth: 第3号接收器的实际深度 (ft)
            side: 方位标识 ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
            cast_depth: CAST深度数组 (N_depth,)
            cast_zc: CAST Zc值数组 (180, N_depth)
            
        Returns:
            channeling_ratio: 窜槽比例标签 [0, 1]
        """
        # 获取方位角扇区
        azimuth_sector = self.azimuth_sectors[side]
        azimuth_indices = self.get_azimuth_indices(azimuth_sector)
        
        # 定义深度窗口
        depth_window_half = self.label_config['depth_window_size'] / 2
        depth_min = depth - depth_window_half
        depth_max = depth + depth_window_half
        
        # 找到深度窗口内的点
        depth_mask = (cast_depth >= depth_min) & (cast_depth <= depth_max)
        depth_indices = np.where(depth_mask)[0]
        
        if len(depth_indices) == 0 or len(azimuth_indices) == 0:
            # 如果没有找到对应的数据点，返回0（无窜槽）
            return 0.0
        
        # 提取目标区域的Zc值
        target_zc = cast_zc[np.ix_(azimuth_indices, depth_indices)]
        
        # 计算窜槽比例
        channeling_mask = target_zc < self.label_config['channeling_threshold']
        channeling_ratio = np.mean(channeling_mask)
        
        return channeling_ratio
    
    def generate_labels_for_side(self, 
                                side: str,
                                actual_depths: np.ndarray,
                                target_indices: np.ndarray,
                                cast_depth: np.ndarray,
                                cast_zc: np.ndarray) -> np.ndarray:
        """
        为某个方位的所有波形生成标签
        
        Args:
            side: 方位标识
            actual_depths: 第3号接收器的实际深度数组
            target_indices: 目标深度范围内的索引
            cast_depth: CAST深度数组
            cast_zc: CAST Zc值数组
            
        Returns:
            labels: 标签数组 (n_samples,)
        """
        n_samples = len(target_indices)
        labels = np.zeros(n_samples)
        
        for i, idx in enumerate(target_indices):
            depth = actual_depths[idx]
            label = self.generate_label_for_waveform(depth, side, cast_depth, cast_zc)
            labels[i] = label
        
        return labels
    
    def generate_all_labels(self, 
                           actual_depths: np.ndarray,
                           target_indices: np.ndarray,
                           cast_depth: np.ndarray,
                           cast_zc: np.ndarray) -> Dict[str, np.ndarray]:
        """
        为所有方位的波形生成标签
        
        Args:
            actual_depths: 第3号接收器的实际深度数组
            target_indices: 目标深度范围内的索引
            cast_depth: CAST深度数组
            cast_zc: CAST Zc值数组
            
        Returns:
            all_labels: 各方位的标签字典 {side: labels}
        """
        all_labels = {}
        
        print("开始生成标签...")
        
        for side in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            print(f"  生成方位 {side} 的标签...")
            labels = self.generate_labels_for_side(
                side, actual_depths, target_indices, cast_depth, cast_zc
            )
            all_labels[side] = labels
            
            # 统计信息
            mean_ratio = np.mean(labels)
            max_ratio = np.max(labels)
            high_channeling_count = np.sum(labels > 0.5)
            
            print(f"    样本数: {len(labels)}")
            print(f"    平均窜槽比例: {mean_ratio:.3f}")
            print(f"    最大窜槽比例: {max_ratio:.3f}")
            print(f"    高窜槽样本数 (>0.5): {high_channeling_count}")
        
        print("标签生成完成！")
        return all_labels
    
    def get_high_channeling_samples(self, 
                                   labels: Dict[str, np.ndarray], 
                                   threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        获取高窜槽比例的样本索引
        
        Args:
            labels: 各方位的标签字典
            threshold: 高窜槽比例阈值
            
        Returns:
            high_channeling_indices: 各方位的高窜槽样本索引
        """
        high_channeling_indices = {}
        
        for side, side_labels in labels.items():
            high_indices = np.where(side_labels >= threshold)[0]
            high_channeling_indices[side] = high_indices
            
            print(f"方位 {side}: {len(high_indices)} 个高窜槽样本 (>={threshold})")
        
        return high_channeling_indices 