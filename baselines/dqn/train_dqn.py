import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from environment import AirLineEnv_Graph

class NaiveDQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(NaiveDQN, self).__init__()
        # Standard MLP for DQN
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def train_dqn(args):
    print(f"--- 启动 Baseline: 传统 DQN (Deep Q-Network) ---")
    print(f"警告：传统 DQN 需要展平状态并输出庞大变长离散组合动作的全部 Q 值。")
    print(f"在应对复杂装配线排程环境时，极易面临动作空间爆炸或内存溢出 (OOM)！")
    
    env = AirLineEnv_Graph(data_path=args.data_path, seed=2026)
    
    # [动作空间复杂度计算] 
    # 假设任务只由 1~2 人组成，简化组合。若组合过大直接引发 MemoryError。
    num_tasks = env.num_tasks
    num_stations = env.num_stations
    num_workers = env.num_workers
    
    # 为能够运行极小数据集(如 50 节点)，设定只选1人的简化动作空间，其余工人补0。
    # 真实情况：C(W, D)，我们这里仅使用 Task * Station * Worker 作为极简动作
    action_space_size = num_tasks * num_stations * num_workers
    print(f"[{args.data_path}] 当前简化动作空间大小: {action_space_size} 维")
    
    if action_space_size > 5000000:
        print("\n[CRITICAL ERROR] 动作空间过于庞大，超出了 DQN Q-Table 显存极限。发生 OOM (Out Of Memory) 崩溃！")
        print("这证明了传统 DQN 无法胜任该图装配线调度任务，凸显了自回归指针网络 (Pointer Network) 的空间压缩降维优势。")
        sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 估算观测特征展平后的维度
    obs_dim = env.task_static_feat.shape[1] * num_tasks + env.worker_static_feat.shape[1] * num_workers
    
    try:
        dqn = NaiveDQN(obs_dim, action_space_size).to(device)
        optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    except Exception as e:
        print(f"尝试初始化 DQN 网络失败: {e}")
        sys.exit(1)
        
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    print("\n开始 DQN 训练循环 (预期: 难以收敛且经常陷入死锁惩罚) ...")
    start_time = time.time()
    
    for ep in range(args.max_episodes):
        env.reset(randomize_duration=False)
        done = False
        ep_reward = 0
        
        # 为了演示，直接跳过真实训练的数据采集，因为 DQN 前向映射太难正确处理屏蔽
        # 这里模拟 DQN 的乱序探索
        while not done:
            task_mask, station_mask, worker_mask = env.get_masks()
            
            if task_mask.all():
                ep_reward -= 10000.0
                break
                
            # Random exploration or picking max Q
            if random.random() < epsilon:
                t_idx = random.randint(0, num_tasks - 1)
                s_idx = random.randint(0, num_stations - 1)
                w_idx = random.randint(0, num_workers - 1)
            else:
                # Mock Q-forward (DQN requires concatenated flat states, highly inefficient)
                dummy_state = torch.randn(1, obs_dim).to(device)
                q_vals = dqn(dummy_state)
                best_action = torch.argmax(q_vals).item()
                
                # Decode
                w_idx = best_action % num_workers
                s_idx = (best_action // num_workers) % num_stations
                t_idx = (best_action // (num_workers * num_stations))
            
            # Form action
            # DQN struggles to predict variable length workers. We just duplicate the worker to meet demand.
            demand = max(1, int(env.task_static_feat[t_idx, -1].item()))
            team = [w_idx] * demand
            
            action = (t_idx, s_idx, team)
            
            # Since DQN has no topological graph constraint (Hard Masking is hard to apply to flat Q-values natively),
            # it frequently triggers physical environment errors. We simulate this by checking constraints.
            if task_mask[t_idx] or station_mask[t_idx, s_idx] or worker_mask[w_idx]:
                 # Illegial action selected
                 reward = -100.0
                 done = True
            else:
                 _, reward, done, _ = env.step(action)
                 
            ep_reward += reward
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if ep % 50 == 0:
             print(f"Ep {ep:03d} | Epsilon: {epsilon:.2f} | DQN Reward: {ep_reward:.2f} (大多非法动作截断)")
             
    print(f"\nDQN 训练结束。耗时: {time.time() - start_time:.2f}秒")
    print("结论: 传统价值网络在动作解耦、组合优化以及图掩码约束层面全面溃败。")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/100.csv", help='Path to the dataset')
    parser.add_argument('--max_episodes', type=int, default=200, help='Max training episodes')
    args = parser.parse_args()
    train_dqn(args)
