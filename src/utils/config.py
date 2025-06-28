"""
配置文件加载工具
"""

import yaml
import os
from pathlib import Path

def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    # 如果是相对路径，从项目根目录开始
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_project_root():
    """
    获取项目根目录
    
    Returns:
        Path: 项目根目录路径
    """
    return Path(__file__).parent.parent.parent

def ensure_dir(path):
    """
    确保目录存在
    
    Args:
        path (str or Path): 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True) 