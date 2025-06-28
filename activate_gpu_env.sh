#!/bin/bash
# 激活GPU训练环境脚本
# 使用方法: source activate_gpu_env.sh

echo "=== 激活测井数据分析GPU环境 ==="

# 加载conda配置
source /usr/local/anaconda3/etc/profile.d/conda.sh

# 激活环境
conda activate hal_logging

# 设置正确的PATH
export PATH="/usr/local/anaconda3/envs/hal_logging/bin:$PATH"

# 验证环境
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "当前GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo "✅ GPU环境激活成功！"
echo ""
echo "现在可以运行以下命令："
echo "  python scripts/run_training.py       # GPU训练"
echo "  python scripts/run_gradcam.py        # GPU Grad-CAM可视化"
echo "  python scripts/visualize_gradcam.py  # GPU可视化" 