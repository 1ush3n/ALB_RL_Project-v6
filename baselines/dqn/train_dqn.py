import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque

# 添加根路径以便导入外部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from args_parser import get_dqn_parser
from env_wrapper import init_env, standardize_env_reset, standardize_env_step, extract_flat_state_for_baselines
from utils.logger import init_logger, record_experiment_time
from utils.device_utils import get_available_device, clear_torch_cache

# DQN网络（适配环境MultiDiscrete动作空间）
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim_list, hidden_dim=128):
        super(DQN, self).__init__()
        # 共享特征层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 动作分支（适配MultiDiscrete: [task, station, worker_leader]）
        self.task_head = nn.Linear(hidden_dim, action_dim_list[0])
        self.station_head = nn.Linear(hidden_dim, action_dim_list[1])
        self.worker_head = nn.Linear(hidden_dim, action_dim_list[2])
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        task_logits = self.task_head(x)
        station_logits = self.station_head(x)
        worker_logits = self.worker_head(x)
        return task_logits, station_logits, worker_logits

class DQNAgent:
    def __init__(self, state_dim, action_dim_list, args, device):
        self.device = device
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.action_dim_list = action_dim_list
        
        # 初始化网络
        self.model = DQN(state_dim, action_dim_list).to(device)
        self.target_model = DQN(state_dim, action_dim_list).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        # 经验回放池（限制大小，避免内存溢出）
        self.memory = deque(maxlen=getattr(args, 'memory_size', 10000))
        
    def select_action(self, state, env_for_demand=None):
        """
        带探索的动作选择，适配MultiDiscrete动作空间
        """
        if np.random.rand() <= self.epsilon:
            # 随机动作（探索）
            task = np.random.randint(0, self.action_dim_list[0])
            station = np.random.randint(0, self.action_dim_list[1])
            worker = np.random.randint(0, self.action_dim_list[2])
        else:
            # 贪心动作（利用）
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                task_logits, station_logits, worker_logits = self.model(state_tensor)
                task = torch.argmax(task_logits).item()
                station = torch.argmax(station_logits).item()
                worker = torch.argmax(worker_logits).item()
                
        # To avoid ValueError unpacking in env.step due to list size
        if env_for_demand is not None:
             demand = int(env_for_demand.task_static_feat[task, 2].item())
             demand = max(1, demand)
             team = [abs(worker + i) % self.action_dim_list[2] for i in range(demand)]
        else:
             team = [worker]
        return (task, station, team)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        """
        经验回放，分批训练避免内存溢出
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        losses = []
        
        # 为了高效，可以将这里向量化。现在先用最稳妥的逐条计算
        for idx in batch_indices:
            state, action, reward, next_state, done = self.memory[idx]
            
            # 转换为tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
            
            # 当前Q值
            task_logits, station_logits, worker_logits = self.model(state_tensor)
            task_a, station_a, team_a = action
            leader_a = team_a[0]
            q_current = (task_logits[task_a] + station_logits[station_a] + worker_logits[leader_a]) / 3.0
            
            # 目标Q值
            if done:
                q_target = reward_tensor
            else:
                with torch.no_grad():
                    next_task_logits, next_station_logits, next_worker_logits = self.target_model(next_state_tensor)
                    next_q = (torch.max(next_task_logits) + torch.max(next_station_logits) + torch.max(next_worker_logits)) / 3.0
                q_target = reward_tensor + self.gamma * next_q
            
            # 计算损失
            loss = self.loss_fn(q_current, q_target)
            losses.append(loss.item())
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return np.mean(losses) if losses else 0.0

def train_dqn(args):
    # 初始化日志
    logger, exp_dir = init_logger(args, "dqn_baseline")
    start_time = time.time()
    
    try:
        # 设备初始化
        device = get_available_device()
        # 环境初始化（统一接口）
        env = init_env(args, seed=args.seed)
        
        # 状态维度适配（使用降维展平方法）
        standardize_env_reset(env)
        flat_state = extract_flat_state_for_baselines(env)
        state_dim = flat_state.shape[0]
        
        action_dim_list = [env.num_tasks, env.num_stations, env.num_workers]
        
        # 初始化Agent
        agent = DQNAgent(state_dim, action_dim_list, args, device)
        batch_size = getattr(args, 'batch_size', 32)
        
        # 训练指标
        episode_rewards = []
        episode_losses = []
        episode_makespans = []
        
        # 训练循环
        logger.info(f"开始 DQN 训练，状态维度: {state_dim}，动作维度: {action_dim_list}，最大轮次: {args.max_episodes}")
        for ep in range(args.max_episodes):
            standardize_env_reset(env)
            state = extract_flat_state_for_baselines(env)
            done = False
            ep_reward = 0
            ep_loss = 0
            step_count = 0
            max_steps = env.num_tasks * 2  # 防止无限循环
            
            while not done and step_count < max_steps:
                step_count += 1
                # 选择动作
                action = agent.select_action(state, env_for_demand=env)
                # 执行动作
                _, reward, done, info = standardize_env_step(env, action)
                next_state = extract_flat_state_for_baselines(env)
                
                # 存储经验
                agent.remember(state, action, reward, next_state, done)
                # 累加奖励
                ep_reward += reward
                # 经验回放
                loss = agent.replay(batch_size)
                ep_loss += loss
                # 更新状态
                state = next_state
            
            # 记录指标
            episode_rewards.append(ep_reward)
            episode_losses.append(ep_loss / step_count if step_count > 0 else 0)
            
            task_mask, _, _ = env.get_masks()
            makespan = np.max(env.station_wall_clock) if not task_mask.all() else 99999.0
            episode_makespans.append(makespan)
            
            # 每10轮打印日志
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(episode_losses[-10:])
                avg_makespan = np.mean(episode_makespans[-10:])
                logger.info(f"Episode {ep+1}/{args.max_episodes} | 平均奖励: {avg_reward:.2f} | 损失: {avg_loss:.4f} | Makespan: {avg_makespan:.2f} | Epsilon: {agent.epsilon:.4f}")
            
            # 每50轮更新目标网络
            if (ep + 1) % 50 == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                # 清理缓存
                clear_torch_cache()
        
        # 保存模型
        model_path = os.path.join(exp_dir, "dqn_model.pth")
        torch.save(agent.model.state_dict(), model_path)
        logger.info(f"模型保存至: {model_path}")
        
        # 结果归档
        results = pd.DataFrame({
            'episode': range(1, args.max_episodes+1),
            'reward': episode_rewards,
            'loss': episode_losses,
            'makespan': episode_makespans
        })
        results['avg_reward_10'] = results['reward'].rolling(window=10).mean()
        results['avg_makespan_10'] = results['makespan'].rolling(window=10).mean()
        results.to_csv(os.path.join(exp_dir, "dqn_results.csv"), index=False)
        
    except Exception as e:
        logger.error(f"DQN训练失败: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理资源
        record_experiment_time(logger, start_time)
        clear_torch_cache()

if __name__ == "__main__":
    parser = get_dqn_parser()
    args = parser.parse_args()
    train_dqn(args)
