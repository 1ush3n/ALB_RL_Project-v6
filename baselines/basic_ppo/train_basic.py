import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import the core training loop
from train import train
import configs

def run_basic_ppo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/100.csv", help='Path to the dataset')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    # 吸收所有参数，避免冲突
    args, unknown = parser.parse_known_args()
    
    print("=" * 60)
    print("🚀 启动对比基线: Basic PPO (纯 MLP, 无图注意力, 无指针网络) 🚀")
    print(">>> 正在剥离核心结构的图神经传导...")
    print(">>> 正在剥离核心的自回归注意力交互...")
    print("=" * 60)
    
    # 强制动态写入消融配置，退化模型
    setattr(configs, 'ablation_no_gat', True)
    setattr(configs, 'ablation_no_pointer', True)
    setattr(configs, 'ablation_no_mask', False) # 保留基础物理约束，防止训练完全停摆
    
    # 调用主训练循环，生成的张量板和模型将体现其性能退化
    train(args)

if __name__ == "__main__":
    run_basic_ppo()
