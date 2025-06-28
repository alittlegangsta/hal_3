# Grad-CAM可视化使用说明

## 功能概述

Grad-CAM（Gradient-weighted Class Activation Mapping）可视化工具用于分析CNN模型在处理声波测井数据时关注的时频特征，帮助理解模型如何识别水泥胶结窜槽现象。

## 快速开始

### 基本使用

```bash
# 自动选择方位A的高窜槽样本进行可视化
python scripts/visualize_gradcam.py

# 指定方位
python scripts/visualize_gradcam.py --azimuth B

# 指定特定样本
python scripts/visualize_gradcam.py --azimuth A --sample 15

# 指定输出目录
python scripts/visualize_gradcam.py --output custom_output_dir
```

### 参数说明

- `--azimuth, -a`: 方位方向 (A-H)，默认为A
- `--sample, -s`: 样本索引，默认自动选择高窜槽样本
- `--output, -o`: 输出目录，默认为`data/results`

## 输出说明

### 可视化图片包含四个子图：

1. **Original CWT Features (左上)**: 原始连续小波变换特征
   - X轴：时间采样点 (0-1023)
   - Y轴：频率尺度 (0-63)
   - 颜色：特征强度

2. **Grad-CAM Attention Heatmap (右上)**: 注意力热力图
   - 热色区域表示模型高度关注的时频区域
   - 红色：高关注度，蓝色：低关注度

3. **CWT + Grad-CAM Overlay (左下)**: 叠加图
   - 原始特征与注意力热力图的叠加
   - 便于直观理解模型关注的具体特征

4. **Statistical Features (右下)**: 统计特征
   - 8个统计特征的条形图
   - Mean, Std, Max, Min, Energy, RMS, Skewness, Kurtosis

### 关键信息输出

```
Analyzing sample 6 from azimuth A
Channeling ratio: 0.848              # 真实窜槽比例标签
Model prediction: 1820.036           # 模型预测值
Peak attention at: Frequency scale 2, Time sample 14  # 峰值关注位置
Attention intensity range: 0.134 - 1.000  # 关注强度范围
```

## 分析技巧

### 1. 高窜槽样本分析
- 自动选择窜槽比例 > 0.5 的样本
- 关注高频区域的激活模式
- 对比不同方位的注意力分布

### 2. 时频特征解读
- **低频区域关注**: 可能对应声波的主要传播模式
- **高频区域关注**: 可能对应散射和反射特征
- **时间集中关注**: 可能对应特定的声波到达时间

### 3. 对比分析
```bash
# 对比不同方位的同一样本
python scripts/visualize_gradcam.py --azimuth A --sample 10
python scripts/visualize_gradcam.py --azimuth B --sample 10

# 对比同一方位的不同样本
python scripts/visualize_gradcam.py --azimuth A --sample 5
python scripts/visualize_gradcam.py --azimuth A --sample 15
```

## 调试模式

如果需要更详细的调试信息和批量分析，可以运行：

```bash
python scripts/run_gradcam.py
```

这将生成：
- 多个样本的详细可视化
- 统计分析报告
- 关注模式分析

## 输出文件

生成的可视化文件命名格式：
```
gradcam_azimuth_{方位}_sample_{样本索引}.png
```

例如：
- `gradcam_azimuth_A_sample_6.png`
- `gradcam_azimuth_B_sample_10.png`

## 技术说明

- **目标层**: `cnn_branch.conv_layers.2` (第3个卷积层)
- **设备**: 自动使用CPU避免CUDA兼容性问题
- **输入格式**: (batch_size=1, channels=1, height=64, width=1024)
- **输出分辨率**: 150 DPI

## 常见问题

### Q: 为什么有些样本没有明显的关注区域？
A: 这可能表示：
1. 样本的窜槽特征不明显
2. 模型对该样本的预测不确定
3. 特征在多个区域分散分布

### Q: 如何选择分析哪些样本？
A: 建议分析：
1. 高窜槽比例样本 (>0.5)
2. 模型预测错误的样本
3. 不同方位的代表性样本

### Q: Grad-CAM的分辨率为什么比原始特征低？
A: Grad-CAM基于卷积层的特征图，经过池化操作后分辨率降低是正常的。这有助于突出主要的关注区域。

---

*注：本工具基于训练好的混合CNN模型，确保在使用前已完成模型训练。* 