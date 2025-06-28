#!/usr/bin/env python3
"""
完整数据集训练脚本
实现完整数据集的CNN模型训练和验证
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
from src.models.feature_extractor import FeatureExtractor
from src.utils.device_utils import get_device


class FullDatasetTrainer:
    """完整数据集训练器"""
    
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
        
    def prepare_full_dataset(self):
        """
        准备完整数据集
        """
        print("准备完整数据集...")
        
        # 加载元数据
        with open('data/processed/full_dataset_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"数据集信息:")
        print(f"  总样本数: {metadata['total_samples']:,}")
        print(f"  高窜槽样本数: {sum(metadata['high_channeling_counts'].values()):,}")
        print(f"  CWT特征形状: {metadata['cwt_feature_shape']}")
        print(f"  统计特征维度: {metadata['stat_feature_dim']}")
        
        # 加载所有方位的数据
        all_cwt_features = []
        all_stat_features = []
        all_labels = []
        
        azimuths = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        print("加载各方位数据...")
        for azimuth in tqdm(azimuths, desc="加载数据"):
            with open(f'data/processed/full_processed_{azimuth}.pkl', 'rb') as f:
                azimuth_data = pickle.load(f)
            
            all_cwt_features.append(azimuth_data['cwt_features'])
            all_stat_features.append(azimuth_data['stat_features'])
            all_labels.append(azimuth_data['labels'])
        
        # 合并所有数据
        print("合并数据...")
        cwt_features = np.concatenate(all_cwt_features, axis=0)
        stat_features = np.concatenate(all_stat_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"合并后的数据形状:")
        print(f"  CWT特征: {cwt_features.shape}")
        print(f"  统计特征: {stat_features.shape}")
        print(f"  标签: {labels.shape}")
        print(f"  标签范围: {labels.min():.3f} - {labels.max():.3f}")
        
        # 特征提取和归一化
        print("进行特征标准化...")
        feature_extractor = FeatureExtractor('standard')
        norm_cwt, norm_stat = feature_extractor.fit_transform(
            cwt_features, stat_features
        )
        
        # 转换为张量
        print("转换为张量...")
        cwt_tensor, stat_tensor, label_tensor = feature_extractor.to_tensors(
            norm_cwt, norm_stat, labels
        )
        
        # 数据分割
        print("进行数据分割...")
        indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=self.training_config['validation']['split_ratio'],
            random_state=self.training_config['validation']['random_seed'],
            stratify=(labels > 0.5).astype(int)  # 按高窜槽/低窜槽分层抽样
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
        print(f"  训练集: {len(train_indices):,} 样本")
        print(f"  验证集: {len(val_indices):,} 样本")
        print(f"  训练标签范围: {train_labels.min():.3f} - {train_labels.max():.3f}")
        print(f"  验证标签范围: {val_labels.min():.3f} - {val_labels.max():.3f}")
        
        # 统计高窜槽样本比例
        train_high_ratio = (train_labels > 0.5).sum().item() / len(train_labels)
        val_high_ratio = (val_labels > 0.5).sum().item() / len(val_labels)
        print(f"  训练集高窜槽比例: {train_high_ratio*100:.1f}%")
        print(f"  验证集高窜槽比例: {val_high_ratio*100:.1f}%")
        
        # 创建数据加载器
        batch_size = self.training_config['batch_size']
        
        train_dataset = TensorDataset(train_cwt, train_stat, train_labels)
        val_dataset = TensorDataset(val_cwt, val_stat, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,  # 并行加载
            pin_memory=True  # 优化GPU传输
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 保存特征提取器
        os.makedirs('data/processed', exist_ok=True)
        feature_extractor.save('data/processed/full_feature_extractor.pkl')
        
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
        num_epochs = self.training_config['num_epochs']
        
        # 训练历史
        train_losses = []
        val_losses = []
        val_r2_scores = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练 {num_epochs} 个epoch...")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
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
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
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
            
            # 计算评估指标
            val_r2 = r2_score(val_targets, val_predictions)
            val_mae = mean_absolute_error(val_targets, val_predictions)
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            
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
            print(f"  验证RMSE: {val_rmse:.4f}")
            print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'data/processed/full_best_model.pth')
                print(f"  ✅ 保存最佳模型 (验证损失: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= self.training_config['early_stopping']['patience']:
                print(f"早停：验证损失在 {self.training_config['early_stopping']['patience']} 个epoch内没有改善")
                break
        
        print(f"训练完成!")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        
        # 保存训练历史
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'best_val_loss': best_val_loss,
            'total_epochs': len(train_losses)
        }
        
        with open('data/processed/full_training_history.pkl', 'wb') as f:
            pickle.dump(training_history, f)
        
        # 绘制训练曲线
        self.plot_training_curves(training_history)
        
        return model, training_history
    
    def plot_training_curves(self, history):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # 损失曲线
        axes[0].plot(epochs, history['train_losses'], 'b-', label='训练损失')
        axes[0].plot(epochs, history['val_losses'], 'r-', label='验证损失')
        axes[0].set_title('训练和验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].legend()
        axes[0].grid(True)
        
        # R2分数
        axes[1].plot(epochs, history['val_r2_scores'], 'g-', label='验证R2')
        axes[1].set_title('验证R2分数')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R2')
        axes[1].legend()
        axes[1].grid(True)
        
        # 对数损失
        axes[2].semilogy(epochs, history['train_losses'], 'b-', label='训练损失')
        axes[2].semilogy(epochs, history['val_losses'], 'r-', label='验证损失')
        axes[2].set_title('训练和验证损失 (对数尺度)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('损失 (对数尺度)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('data/results/full_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("训练曲线保存到: data/results/full_training_curves.png")


def main():
    """主函数"""
    print("=" * 70)
    print("开始完整数据集训练")
    print("=" * 70)
    
    # 创建结果目录
    os.makedirs('data/results', exist_ok=True)
    
    try:
        # 初始化训练器
        trainer = FullDatasetTrainer()
        
        # 准备数据
        train_loader, val_loader, feature_extractor = trainer.prepare_full_dataset()
        
        # 训练模型
        model, history = trainer.train_model(train_loader, val_loader)
        
        print("\n" + "=" * 70)
        print("完整数据集训练成功完成！")
        print("=" * 70)
        print(f"最佳验证损失: {history['best_val_loss']:.4f}")
        print(f"训练epochs: {history['total_epochs']}")
        print("模型和结果保存在 data/processed/ 和 data/results/ 目录下")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 