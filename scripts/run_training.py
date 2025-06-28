#!/usr/bin/env python3
"""
训练脚本
实现小样本CNN模型训练和验证
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_model import create_model
from src.models.feature_extractor import FeatureExtractor, prepare_dataset_from_processed_data
from src.utils.device_utils import get_device


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        
        # 使用统一的设备选择函数
        self.device = get_device(config_path)
        
    def prepare_data(self, processed_data_path: str):
        """
        准备训练数据
        
        Args:
            processed_data_path: 预处理数据路径
        """
        print("准备训练数据...")
        
        # 加载预处理数据
        dataset = prepare_dataset_from_processed_data(processed_data_path)
        
        # 特征提取和归一化
        feature_extractor = FeatureExtractor('standard')
        norm_cwt, norm_stat = feature_extractor.fit_transform(
            dataset['cwt_features'], 
            dataset['stat_features']
        )
        
        # 转换为张量
        cwt_tensor, stat_tensor, label_tensor = feature_extractor.to_tensors(
            norm_cwt, norm_stat, dataset['labels']
        )
        
        # 数据分割
        indices = np.arange(len(dataset['labels']))
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=self.training_config['validation']['split_ratio'],
            random_state=self.training_config['validation']['random_seed']
        )
        
        # 训练集
        train_cwt = cwt_tensor[train_indices]
        train_stat = stat_tensor[train_indices]
        train_labels = label_tensor[train_indices]
        
        # 验证集
        val_cwt = cwt_tensor[val_indices]
        val_stat = stat_tensor[val_indices]
        val_labels = label_tensor[val_indices]
        
        print(f"数据分割完成:")
        print(f"  训练集: {len(train_indices)} 样本")
        print(f"  验证集: {len(val_indices)} 样本")
        print(f"  训练标签范围: {train_labels.min():.3f} - {train_labels.max():.3f}")
        print(f"  验证标签范围: {val_labels.min():.3f} - {val_labels.max():.3f}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_cwt, train_stat, train_labels)
        val_dataset = TensorDataset(val_cwt, val_stat, val_labels)
        
        batch_size = self.training_config['small_sample']['batch_size'] if self.training_config['small_sample']['enabled'] else self.training_config['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 保存特征提取器
        os.makedirs('data/processed', exist_ok=True)
        feature_extractor.save('data/processed/feature_extractor.pkl')
        
        return train_loader, val_loader, feature_extractor
    
    def train_model(self, train_loader, val_loader):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        print("开始训练模型...")
        
        # 创建模型
        model = create_model()
        model.to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['optimizer']['weight_decay']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.training_config['scheduler']['step_size'],
            gamma=self.training_config['scheduler']['gamma']
        )
        
        # 训练参数
        num_epochs = self.training_config['small_sample']['num_epochs'] if self.training_config['small_sample']['enabled'] else self.training_config['num_epochs']
        
        # 训练历史
        train_losses = []
        val_losses = []
        val_r2_scores = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练 {num_epochs} 个epoch...")
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for cwt_batch, stat_batch, label_batch in train_pbar:
                cwt_batch = cwt_batch.to(self.device)
                stat_batch = stat_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(cwt_batch, stat_batch)
                loss = criterion(outputs.squeeze(), label_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for cwt_batch, stat_batch, label_batch in val_pbar:
                    cwt_batch = cwt_batch.to(self.device)
                    stat_batch = stat_batch.to(self.device)
                    label_batch = label_batch.to(self.device)
                    
                    outputs = model(cwt_batch, stat_batch)
                    loss = criterion(outputs.squeeze(), label_batch)
                    
                    val_loss += loss.item()
                    val_predictions.extend(outputs.squeeze().cpu().numpy())
                    val_targets.extend(label_batch.cpu().numpy())
                    
                    val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            val_loss /= len(val_loader)
            
            # 计算R2分数
            val_r2 = r2_score(val_targets, val_predictions)
            val_mae = mean_absolute_error(val_targets, val_predictions)
            
            # 记录历史
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_r2_scores.append(val_r2)
            
            # 学习率调度
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证R2: {val_r2:.4f}")
            print(f"  验证MAE: {val_mae:.4f}")
            print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'data/processed/best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.training_config['early_stopping']['patience']:
                print(f"早停：验证损失在 {self.training_config['early_stopping']['patience']} 个epoch内没有改善")
                break
        
        # 保存训练历史
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'best_val_loss': best_val_loss
        }
        
        with open('data/processed/training_history.pkl', 'wb') as f:
            pickle.dump(training_history, f)
        
        print("训练完成!")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        
        return model, training_history


def main():
    """主函数"""
    try:
        # 初始化训练器
        trainer = ModelTrainer()
        
        # 准备数据
        train_loader, val_loader, feature_extractor = trainer.prepare_data('data/processed/small_sample_processed.pkl')
        
        # 训练模型
        model, training_history = trainer.train_model(train_loader, val_loader)
        
        print("\n" + "=" * 70)
        print("小样本训练测试成功完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 