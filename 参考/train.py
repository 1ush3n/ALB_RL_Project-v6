import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import traceback
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ALB_RL_Project.environment import AirLineEnv_Graph
from ALB_RL_Project.models.hb_gat_pn import HBGATPN
from ALB_RL_Project.ppo_agent import PPOAgent
from ALB_RL_Project.configs import configs

# ---------------------------------------------------------------------------
# 经验回放缓冲区 (Memory Buffer)
# ---------------------------------------------------------------------------
class Memory:
    """
    存储 PPO 训练所需的轨迹数据。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = [] # (task_mask, station_mask, worker_mask)
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]

# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------
def evaluate_model(env, agent):
    """
    使用确定性策略评估当前模型性能。
    
    Returns:
        makespan (float): 最大完工时间 (各站位负载的最大值)
        balance_std (float): 站位负载的标准差 (越小越平衡)
        total_reward (float): 有效奖励总和
    """
    state = env.reset()
    done = False
    total_reward = 0
    device = agent.device
    
    while not done:
        task_mask, station_mask, worker_mask = env.get_masks()
        
        # 确定性选择动作 (ArgMax)
        action, _, _, _ = agent.select_action(
            state.to(device), 
            mask_task=task_mask.to(device), 
            mask_station_matrix=station_mask.to(device),
            mask_worker=worker_mask.to(device),
            deterministic=True
        )
        
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
    makespan = np.max(env.station_loads)
    balance_std = np.std(env.station_loads)
    
    return makespan, balance_std, total_reward

# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train(args):
    try:
        print("--- 开始训练 (Starting Training) ---")
        
        # 1. 初始化环境
        data_path = str(configs.data_file_path) if configs.data_file_path else "3000.csv"
        # 转换为绝对路径以防万一
        if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
             data_path = os.path.join(os.getcwd(), data_path)
             
        print(f"数据路径: {data_path}")
        # 固定种子以保证训练环境的一致性 (Determinism)
        env = AirLineEnv_Graph(data_path=data_path, seed=42)
        print("环境初始化完成.")
        
        # 2. 初始化设备与模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model = HBGATPN(configs).to(device)
        print("模型已加载至设备.")
        
        # Init Agent
        # Calculate Total Timesteps for Scheduler
        total_updates = int(configs.max_episodes / configs.update_every_episodes * configs.k_epochs)

        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=configs.batch_size,
            # [Scheduler Params]
            lr_warmup_steps=configs.lr_warmup_steps,
            min_lr=configs.min_lr,
            total_timesteps=total_updates
        )

        

        print(f"Agent Initialized. Total Scheduled Updates: {total_updates}")
        
        # 3. 断点续训 (Resume Training)
        start_episode = 1
        model_dir = "ALB_RL_Project/models"
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pth")
        
        if args.resume and os.path.exists(checkpoint_path):
            print(f"正在从 {checkpoint_path} 恢复训练...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            print(f"恢复成功. 起始 Episode: {start_episode}")
        
        # 最佳模型记录
        best_makespan = float('inf')
        best_model_path = os.path.join(model_dir, "best_model.pth")
        
        # 4. TensorBoard 设置
        run_name = f"ALB_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(configs.log_dir, run_name)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard 日志目录: {log_dir}")
        
        memory = Memory()
        
        # 5. 训练循环参数
        max_episodes = configs.max_episodes 
        update_every_episodes = configs.update_every_episodes
        eval_freq = configs.eval_freq
        
        print(f"开始 Episode 循环 (Max: {max_episodes})...")
        
        for i_episode in range(start_episode, max_episodes+1):
            state = env.reset()
            ep_reward = 0
            
            # 动态设置最大步数 (防止无限循环，通常设为任务数的2倍)
            max_steps = env.num_tasks * 2 
            
            for t in range(max_steps):
                # [内存管理] 定期清理 CUDA 缓存
                if t % 500 == 0:
                    torch.cuda.empty_cache()
                
                # 获取 Mask
                task_mask, station_mask, worker_mask = env.get_masks()
                t_mask = task_mask.to(device)
                s_mask = station_mask.to(device)
                w_mask = worker_mask.to(device)
                
                # 死锁检测 (Deadlock Check)
                if t_mask.all():
                     print(f"DEADLOCK (Step {t}): 无可行任务。")
                     reward = -1000.0 
                     done = True
                     # 记录这一步以供学习
                     memory.states.append(state.to('cpu'))
                     memory.actions.append((0,0,[])) # Dummy Action
                     memory.logprobs.append(torch.tensor(0.0))
                     memory.rewards.append(reward)
                     memory.is_terminals.append(done)
                     memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
                     ep_reward += reward
                     break
                
                if w_mask.all():
                     # 所有工人都在忙，理论上 _advance_time 会跳过这段时间，
                     # 但如果出现这种情况，说明时间推进逻辑可能需要检查。
                     # 这里仅作警告。
                     pass
                
                # 选择动作 (Stochastic)
                action, logprob, val, _ = agent.select_action(
                    state.to(device), 
                    mask_task=t_mask, 
                    mask_station_matrix=s_mask,
                    mask_worker=w_mask,
                    deterministic=False
                )
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储轨迹
                memory.states.append(state.to('cpu')) 
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
                
                state = next_state
                ep_reward += reward
                
                if done:
                    break
            
            # 记录日志
            writer.add_scalar('Reward/Episode', ep_reward, i_episode)
            print(f"Episode {i_episode} | Reward: {ep_reward:.2f} | Steps: {t+1}")
            
            # PPO 更新
            if i_episode % update_every_episodes == 0:
                metrics = agent.update(memory)
                memory.clear()
                
                for k, v in metrics.items():
                    writer.add_scalar(k, v, i_episode)
                
            # 定期评估与保存
            if i_episode % eval_freq == 0:
                makespan, balance, eval_reward = evaluate_model(env, agent)
                
                print(f"[Eval] Ep {i_episode} | Makespan: {makespan:.1f} | Balance: {balance:.2f} | AvgReward: {eval_reward:.2f}")
                
                writer.add_scalar('Eval/Makespan', makespan, i_episode)
                writer.add_scalar('Eval/Balance_Std', balance, i_episode)
                
                # Save Latest
                torch.save({
                    'episode': i_episode,
                    'model_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, checkpoint_path)
                
                # Save Best
                if makespan < best_makespan:
                    best_makespan = makespan
                    torch.save(agent.policy.state_dict(), best_model_path)
                    print(f"New Best Model Saved! Makespan: {best_makespan}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()
    
    train(args)
