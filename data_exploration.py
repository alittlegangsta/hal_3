import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def explore_cast_data():
    """探索CAST超声数据"""
    print("=" * 50)
    print("探索CAST数据")
    print("=" * 50)
    
    # 加载CAST数据
    cast_data = scipy.io.loadmat('data/raw/CAST.mat')
    
    print("CAST.mat文件结构:")
    for key in cast_data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(cast_data[key])}, shape: {cast_data[key].shape}, dtype: {cast_data[key].dtype}")
    
    # 正确读取结构体数据
    cast_struct = cast_data['CAST'][0, 0]  # 获取结构体
    
    print("\nCAST结构体字段:")
    for field_name in cast_struct.dtype.names:
        field_data = cast_struct[field_name]
        print(f"  {field_name}: {type(field_data)}, shape: {field_data.shape}, dtype: {field_data.dtype}")
    
    # 获取深度和Zc数据
    depth_cast = cast_struct['Depth'].flatten()
    zc = cast_struct['Zc']
    
    print(f"\nCAST深度信息:")
    print(f"  深度范围: {depth_cast.min():.2f} - {depth_cast.max():.2f} ft")
    print(f"  深度点数: {len(depth_cast)}")
    print(f"  深度间隔: 约{np.mean(np.diff(depth_cast)):.3f} ft")
    
    print(f"\nZc数据信息:")
    print(f"  Zc shape: {zc.shape}")
    print(f"  Zc值范围: {zc.min():.2f} - {zc.max():.2f}")
    print(f"  方位角数: {zc.shape[0]} (应该是180个)")
    
    # 统计窜槽情况（Zc < 2.5）
    channeling_mask = zc < 2.5
    channeling_ratio = np.mean(channeling_mask)
    print(f"  窜槽比例 (Zc < 2.5): {channeling_ratio:.3f}")
    
    # 目标深度范围
    target_depth_mask = (depth_cast >= 2732) & (depth_cast <= 4132)
    target_indices = np.where(target_depth_mask)[0]
    print(f"\n目标深度范围 (2732-4132 ft):")
    print(f"  目标深度点数: {np.sum(target_depth_mask)}")
    if len(target_indices) > 0:
        print(f"  目标深度索引范围: {target_indices[0]} - {target_indices[-1]}")
    else:
        print("  警告：没有找到目标深度范围内的数据")
    
    return cast_struct, target_indices

def explore_xsilmr03_data():
    """探索XSILMR03声波数据"""
    print("\n" + "=" * 50)
    print("探索XSILMR03数据")
    print("=" * 50)
    
    # 加载XSILMR03数据
    xsilmr03_data = scipy.io.loadmat('data/raw/XSILMR03.mat')
    
    print("XSILMR03.mat文件结构:")
    for key in xsilmr03_data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(xsilmr03_data[key])}, shape: {xsilmr03_data[key].shape}, dtype: {xsilmr03_data[key].dtype}")
    
    # 检查是否是结构体格式
    if 'XSILMR03' in xsilmr03_data:
        # 结构体格式
        xsilmr_struct = xsilmr03_data['XSILMR03'][0, 0]
        print("\nXSILMR03结构体字段:")
        for field_name in xsilmr_struct.dtype.names:
            field_data = xsilmr_struct[field_name]
            print(f"  {field_name}: {type(field_data)}, shape: {field_data.shape}, dtype: {field_data.dtype}")
        
        depth_xsilmr = xsilmr_struct['Depth'].flatten()
    else:
        # 直接格式
        depth_xsilmr = xsilmr03_data['Depth'].flatten()
        xsilmr_struct = xsilmr03_data
    
    print(f"\nXSILMR03深度信息:")
    print(f"  深度范围: {depth_xsilmr.min():.2f} - {depth_xsilmr.max():.2f} ft")
    print(f"  深度点数: {len(depth_xsilmr)}")
    print(f"  深度间隔: 约{np.mean(np.diff(depth_xsilmr)):.3f} ft")
    
    # 计算第3号接收器的实际深度
    # 由于第7号接收器是基准，第3号接收器深度 = 基准深度 + (7-3)*0.5 = 基准深度 + 2.0
    actual_depth_r3 = depth_xsilmr + 2.0
    
    print(f"\n第3号接收器实际深度:")
    print(f"  实际深度范围: {actual_depth_r3.min():.2f} - {actual_depth_r3.max():.2f} ft")
    
    # 目标深度范围
    target_depth_mask = (actual_depth_r3 >= 2732) & (actual_depth_r3 <= 4132)
    target_indices = np.where(target_depth_mask)[0]
    print(f"\n目标深度范围 (2732-4132 ft):")
    print(f"  目标深度点数: {np.sum(target_depth_mask)}")
    if len(target_indices) > 0:
        print(f"  目标深度索引范围: {target_indices[0]} - {target_indices[-1]}")
    else:
        print("  警告：没有找到目标深度范围内的数据")
    
    # 检查各个方位的波形数据
    sides = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    print(f"\n各方位波形数据:")
    wave_keys = []
    for side in sides:
        possible_keys = [f'WaveRng03Side{side}', f'WaveRng04Side{side}']  # 可能的键名
        for key in possible_keys:
            if key in xsilmr_struct.dtype.names if hasattr(xsilmr_struct, 'dtype') else key in xsilmr_struct:
                wave_data = xsilmr_struct[key] if hasattr(xsilmr_struct, 'dtype') else xsilmr_struct[key]
                print(f"  {key}: shape {wave_data.shape}, dtype {wave_data.dtype}")
                print(f"    时间点数: {wave_data.shape[0]}, 深度点数: {wave_data.shape[1]}")
                print(f"    幅值范围: {wave_data.min():.3f} - {wave_data.max():.3f}")
                wave_keys.append(key)
                break
    
    return xsilmr_struct, target_indices, actual_depth_r3, wave_keys

def plot_sample_data(cast_struct, xsilmr_struct, target_indices_cast, target_indices_xsilmr, actual_depth_r3, wave_keys):
    """绘制样本数据"""
    print("\n" + "=" * 50)
    print("绘制样本数据")
    print("=" * 50)
    
    if len(target_indices_cast) == 0 or len(target_indices_xsilmr) == 0:
        print("警告：目标深度范围内没有足够的数据用于绘图")
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. CAST深度剖面
    depth_cast = cast_struct['Depth'].flatten()
    zc = cast_struct['Zc']
    
    # 选择一个方位角（0度）的数据
    ax1 = axes[0, 0]
    ax1.plot(depth_cast[target_indices_cast], zc[0, target_indices_cast])
    ax1.axhline(y=2.5, color='r', linestyle='--', label='窜槽阈值 (2.5)')
    ax1.set_xlabel('深度 (ft)')
    ax1.set_ylabel('Zc值')
    ax1.set_title('CAST超声数据 (0度方位)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. CAST 2D热力图（部分数据）
    ax2 = axes[0, 1]
    subset_indices = target_indices_cast[::max(1, len(target_indices_cast)//100)]  # 采样显示
    angles = np.arange(0, 360, 2)  # 每2度一个角度
    im = ax2.imshow(zc[:, subset_indices], aspect='auto', cmap='viridis', 
                    extent=[depth_cast[subset_indices[0]], depth_cast[subset_indices[-1]], 0, 360])
    ax2.set_xlabel('深度 (ft)')
    ax2.set_ylabel('方位角 (度)')
    ax2.set_title('CAST超声数据热力图')
    plt.colorbar(im, ax=ax2, label='Zc值')
    
    # 3. 声波数据样本波形
    ax3 = axes[1, 0]
    if wave_keys:
        wave_data = xsilmr_struct[wave_keys[0]] if hasattr(xsilmr_struct, 'dtype') else xsilmr_struct[wave_keys[0]]
        # 选择目标深度范围内的一个样本
        sample_idx = target_indices_xsilmr[len(target_indices_xsilmr)//2]  # 中间的一个样本
        time_axis = np.arange(wave_data.shape[0]) * 10e-6  # 10微秒间隔
        ax3.plot(time_axis * 1000, wave_data[:, sample_idx])  # 转换为毫秒
        ax3.set_xlabel('时间 (ms)')
        ax3.set_ylabel('幅值')
        ax3.set_title(f'声波波形样本 (深度: {actual_depth_r3[sample_idx]:.1f} ft)')
        ax3.grid(True)
        
        # 4. 多个深度的声波波形对比
        ax4 = axes[1, 1]
        sample_indices = target_indices_xsilmr[::max(1, len(target_indices_xsilmr)//5)][:5]  # 选择5个样本
        for i, idx in enumerate(sample_indices):
            ax4.plot(time_axis * 1000, wave_data[:, idx] + i*0.1, 
                    label=f'深度: {actual_depth_r3[idx]:.1f} ft')
        ax4.set_xlabel('时间 (ms)')
        ax4.set_ylabel('幅值 (偏移显示)')
        ax4.set_title('不同深度的声波波形对比')
        ax4.legend()
        ax4.grid(True)
    else:
        ax3.text(0.5, 0.5, '没有找到波形数据', ha='center', va='center', transform=ax3.transAxes)
        ax4.text(0.5, 0.5, '没有找到波形数据', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
    print("数据探索图保存为 data_exploration.png")

if __name__ == "__main__":
    # 探索数据
    cast_struct, target_indices_cast = explore_cast_data()
    xsilmr_struct, target_indices_xsilmr, actual_depth_r3, wave_keys = explore_xsilmr03_data()
    
    # 绘制样本数据
    plot_sample_data(cast_struct, xsilmr_struct, target_indices_cast, target_indices_xsilmr, actual_depth_r3, wave_keys)
    
    print("\n数据探索完成！") 