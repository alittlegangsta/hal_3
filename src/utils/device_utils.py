#!/usr/bin/env python3
"""
设备工具模块
统一管理GPU/CPU设备选择逻辑
"""

import torch
import yaml


def get_device(config_path: str = "configs/config.yaml") -> torch.device:
    """
    根据配置文件和系统可用性选择设备
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        torch.device: 选择的设备
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        device_config = config.get('device', {})
        use_cuda = device_config.get('use_cuda', False)
        cuda_device = device_config.get('cuda_device', 0)
        
        if use_cuda and torch.cuda.is_available():
            device = torch.device(f"cuda:{cuda_device}")
            print(f"使用GPU设备: {device}")
            print(f"GPU名称: {torch.cuda.get_device_name(cuda_device)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(cuda_device).total_memory / 1024**3:.1f} GB")
            return device
        else:
            device = torch.device("cpu")
            if use_cuda and not torch.cuda.is_available():
                print("警告: 配置要求使用CUDA，但CUDA不可用，回退到CPU")
            print(f"使用CPU设备: {device}")
            return device
            
    except Exception as e:
        print(f"读取配置文件失败，使用默认CPU设备: {e}")
        return torch.device("cpu")


def load_model_with_device(model_path: str, model_class, config_path: str = "configs/config.yaml", **model_kwargs):
    """
    加载模型并设置到正确的设备上
    
    Args:
        model_path: 模型权重文件路径
        model_class: 模型类
        config_path: 配置文件路径
        **model_kwargs: 模型初始化参数
        
    Returns:
        tuple: (model, device)
    """
    device = get_device(config_path)
    
    # 创建模型
    model = model_class(**model_kwargs)
    
    # 加载权重
    if device.type == 'cuda':
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    print(f"模型已加载到设备: {device}")
    return model, device 