"""
CNN模型模块
实现混合输入CNN模型，包含CWT图像分支和统计特征数值分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import yaml


class CNNBranch(nn.Module):
    """CNN图像分支，处理CWT特征"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化CNN分支
        
        Args:
            config: CNN分支配置
        """
        super(CNNBranch, self).__init__()
        
        self.config = config
        conv_layers = config['conv_layers']
        pool_size = config['pool_size']
        dropout_rate = config['dropout_rate']
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # 输入通道数
        
        for i, layer_config in enumerate(conv_layers):
            out_channels = layer_config['out_channels']
            kernel_size = tuple(layer_config['kernel_size'])
            stride = layer_config.get('stride', 1)
            padding = layer_config.get('padding', 1)
            
            # 卷积层
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            
            self.conv_layers.append(conv)
            in_channels = out_channels
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=tuple(pool_size))
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # 自适应平均池化，将特征图降到固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 输出 4x4 特征图
        
        # 计算展平后的特征维度
        self.feature_dim = conv_layers[-1]['out_channels'] * 4 * 4
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, scales, time_points)
            
        Returns:
            features: 输出特征 (batch_size, feature_dim)
        """
        # 通过卷积层
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = self.pool(x)
            x = self.dropout(x)
        
        # 自适应平均池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        return x


class NumericalBranch(nn.Module):
    """数值分支，处理统计特征"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数值分支
        
        Args:
            config: 数值分支配置
        """
        super(NumericalBranch, self).__init__()
        
        input_dim = config['input_dim']
        hidden_layers = config['hidden_layers']
        dropout_rate = config['dropout_rate']
        
        # 构建全连接层
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.output_dim = current_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            features: 输出特征 (batch_size, output_dim)
        """
        return self.layers(x)


class HybridCNNModel(nn.Module):
    """混合输入CNN模型"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化混合CNN模型
        
        Args:
            config_path: 配置文件路径
        """
        super(HybridCNNModel, self).__init__()
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        
        # CNN分支
        self.cnn_branch = CNNBranch(model_config['cnn_branch'])
        
        # 数值分支
        self.numerical_branch = NumericalBranch(model_config['numerical_branch'])
        
        # 融合层
        fusion_config = model_config['fusion']
        fusion_input_dim = self.cnn_branch.feature_dim + self.numerical_branch.output_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(fusion_config['dropout_rate']),
            nn.Linear(fusion_config['hidden_dim'], fusion_config['output_dim'])
        )
        
        # 保存配置用于Grad-CAM
        self.model_config = model_config
        
    def forward(self, cwt_features: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            cwt_features: CWT特征 (batch_size, 1, scales, time_points)
            stat_features: 统计特征 (batch_size, n_features)
            
        Returns:
            output: 预测输出 (batch_size, 1)
        """
        # CNN分支
        cnn_features = self.cnn_branch(cwt_features)
        
        # 数值分支
        numerical_features = self.numerical_branch(stat_features)
        
        # 特征融合
        combined_features = torch.cat([cnn_features, numerical_features], dim=1)
        
        # 最终预测
        output = self.fusion_layer(combined_features)
        
        return output
    
    def get_cnn_features(self, cwt_features: torch.Tensor) -> torch.Tensor:
        """
        获取CNN分支的特征（用于Grad-CAM）
        
        Args:
            cwt_features: CWT特征 (batch_size, 1, scales, time_points)
            
        Returns:
            features: CNN特征 (batch_size, feature_dim)
        """
        return self.cnn_branch(cwt_features)
    
    def get_conv_features(self, cwt_features: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        获取指定卷积层的特征图（用于Grad-CAM）
        
        Args:
            cwt_features: CWT特征 (batch_size, 1, scales, time_points)
            layer_idx: 卷积层索引，-1表示最后一层
            
        Returns:
            features: 卷积特征图
        """
        x = cwt_features
        
        # 通过指定数量的卷积层
        conv_layers = self.cnn_branch.conv_layers
        target_layer = len(conv_layers) + layer_idx if layer_idx < 0 else layer_idx
        
        for i, conv in enumerate(conv_layers):
            x = F.relu(conv(x))
            x = self.cnn_branch.pool(x)
            x = self.cnn_branch.dropout(x)
            
            if i == target_layer:
                return x
        
        return x


def create_model(config_path: str = "configs/config.yaml") -> HybridCNNModel:
    """
    创建模型实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        model: 模型实例
    """
    model = HybridCNNModel(config_path)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型创建完成:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数数量: {trainable_params:,}")
    
    return model


def test_model():
    """测试模型"""
    print("测试CNN模型...")
    
    # 创建模型
    model = create_model()
    
    # 生成测试数据
    batch_size = 8
    cwt_features = torch.randn(batch_size, 1, 64, 1024)  # (batch, channel, scales, time)
    stat_features = torch.randn(batch_size, 8)  # (batch, features)
    
    print(f"输入形状:")
    print(f"  CWT特征: {cwt_features.shape}")
    print(f"  统计特征: {stat_features.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(cwt_features, stat_features)
        cnn_features = model.get_cnn_features(cwt_features)
        conv_features = model.get_conv_features(cwt_features, -1)
    
    print(f"输出形状:")
    print(f"  最终输出: {output.shape}")
    print(f"  CNN特征: {cnn_features.shape}")
    print(f"  卷积特征: {conv_features.shape}")
    print(f"  预测值范围: {output.min().item():.3f} - {output.max().item():.3f}")
    
    print("CNN模型测试通过！")


if __name__ == "__main__":
    test_model() 