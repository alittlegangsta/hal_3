"""
数据加载器模块
负责加载CAST超声数据和XSILMR03声波数据，并进行基础预处理
"""

import os
import numpy as np
import scipy.io
from typing import Tuple, Dict, Any
import yaml


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化数据加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.processing_config = self.config['processing']
        
    def load_cast_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载CAST超声数据
        
        Returns:
            depth_cast: 深度数组 (N_depth,)
            zc: Zc值数组 (180, N_depth)
            target_indices: 目标深度范围内的索引
        """
        file_path = os.path.join(
            self.data_config['raw_data_path'], 
            self.data_config['cast_file']
        )
        
        print(f"加载CAST数据: {file_path}")
        cast_data = scipy.io.loadmat(file_path)
        cast_struct = cast_data['CAST'][0, 0]
        
        # 提取深度和Zc数据
        depth_cast = cast_struct['Depth'].flatten()
        zc = cast_struct['Zc']
        
        # 筛选目标深度范围
        target_depth_mask = (
            (depth_cast >= self.processing_config['target_depth_min']) & 
            (depth_cast <= self.processing_config['target_depth_max'])
        )
        target_indices = np.where(target_depth_mask)[0]
        
        print(f"CAST数据加载完成:")
        print(f"  深度范围: {depth_cast.min():.2f} - {depth_cast.max():.2f} ft")
        print(f"  目标深度点数: {len(target_indices)}")
        print(f"  Zc值范围: {zc.min():.2f} - {zc.max():.2f}")
        
        return depth_cast, zc, target_indices
    
    def load_xsilmr03_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        加载XSILMR03声波数据
        
        Returns:
            wave_data: 各方位的波形数据字典 {side: (1024, N_depth)}
            depth_base: 基准深度数组 (N_depth,)
            actual_depth_r3: 第3号接收器实际深度数组 (N_depth,)
            target_indices: 目标深度范围内的索引
        """
        file_path = os.path.join(
            self.data_config['raw_data_path'], 
            self.data_config['xsilmr03_file']
        )
        
        print(f"加载XSILMR03数据: {file_path}")
        xsilmr_data = scipy.io.loadmat(file_path)
        xsilmr_struct = xsilmr_data['XSILMR03'][0, 0]
        
        # 提取基准深度数据
        depth_base = xsilmr_struct['Depth'].flatten()
        
        # 计算第3号接收器的实际深度
        actual_depth_r3 = depth_base + self.processing_config['receiver_3_offset']
        
        # 筛选目标深度范围
        target_depth_mask = (
            (actual_depth_r3 >= self.processing_config['target_depth_min']) & 
            (actual_depth_r3 <= self.processing_config['target_depth_max'])
        )
        target_indices = np.where(target_depth_mask)[0]
        
        # 提取各方位的波形数据
        wave_data = {}
        sides = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for side in sides:
            key = f'WaveRng03Side{side}'
            if key in xsilmr_struct.dtype.names:
                wave_data[side] = xsilmr_struct[key]
        
        print(f"XSILMR03数据加载完成:")
        print(f"  基准深度范围: {depth_base.min():.2f} - {depth_base.max():.2f} ft")
        print(f"  第3号接收器实际深度范围: {actual_depth_r3.min():.2f} - {actual_depth_r3.max():.2f} ft")
        print(f"  目标深度点数: {len(target_indices)}")
        print(f"  方位数量: {len(wave_data)}")
        
        return wave_data, depth_base, actual_depth_r3, target_indices
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        加载所有数据
        
        Returns:
            data_dict: 包含所有数据的字典
        """
        print("=" * 50)
        print("开始加载数据")
        print("=" * 50)
        
        # 加载CAST数据
        depth_cast, zc, cast_target_indices = self.load_cast_data()
        
        # 加载XSILMR03数据
        wave_data, depth_base, actual_depth_r3, xsilmr_target_indices = self.load_xsilmr03_data()
        
        data_dict = {
            'cast': {
                'depth': depth_cast,
                'zc': zc,
                'target_indices': cast_target_indices
            },
            'xsilmr03': {
                'wave_data': wave_data,
                'depth_base': depth_base,
                'actual_depth_r3': actual_depth_r3,
                'target_indices': xsilmr_target_indices
            }
        }
        
        print("=" * 50)
        print("数据加载完成")
        print("=" * 50)
        
        return data_dict


def test_data_loader():
    """测试数据加载器"""
    loader = DataLoader()
    data = loader.load_all_data()
    
    # 检查数据完整性
    assert 'cast' in data
    assert 'xsilmr03' in data
    assert len(data['xsilmr03']['wave_data']) == 8
    
    print("数据加载器测试通过！")


if __name__ == "__main__":
    test_data_loader() 