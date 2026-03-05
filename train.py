import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import traceback
import argparse

# 添加项目根目录到路径
# (Removed hardcoded sys.path.append to comply with standard project struct)

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs
import pandas as pd
from baseline_ga import GeneticAlgorithmScheduler
from utils.visualization import plot_gantt

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
        self.values = [] # (state_value)
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]
        del self.values[:]

# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------
def evaluate_model(env, agent, num_runs=3, temperature=None):
    """
    使用包含温度平滑的定制定向策略评估当前模型性能，并执行多次以取均值。
    
    Returns:
        makespan (float): 多次运行均值最大完工时间 
        balance_std (float): 多次运行均值站位负载的标准差
        total_reward (float): 多次运行均值有效奖励总和
    """
    if temperature is None:
        temperature = getattr(configs, 'eval_temperature', 0.0)
        
    makespans = []
    balances = []
    rewards = []
    schedules = []
    
    for _ in range(num_runs):
        # 验证场景绝对不可以使用任何数据扰乱！保证评估基线的绝对公平。
        state = env.reset(randomize_duration=False)
        done = False
        total_reward = 0
        device = agent.device
        
        while not done:
            task_mask, station_mask, worker_mask = env.get_masks()
            
            if task_mask.all():
                # [Fix]: Prevent infinite lengths caused by picking blank tasks during a deadlock. 
                print(f"[Eval] DEADLOCK detected! Returning max penalty.")
                break
                
            # 引入验证温度的动作选择
            action, _, _, _ = agent.select_action(
                state.to(device), 
                mask_task=task_mask.to(device), 
                mask_station_matrix=station_mask.to(device),
                mask_worker=worker_mask.to(device),
                deterministic=(temperature == 0.0),
                temperature=temperature
            )
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
        if task_mask.all():
            makespans.append(99999.0) # Matches GA's massive penalty
            balances.append(9999.0)
            rewards.append(total_reward - 10000.0)
            schedules.append([])
        else:
            makespans.append(np.max(env.station_wall_clock)) # [CRITICAL] True physical completion time!
            balances.append(np.std(env.station_loads))     # [Maintain] Use workloads for labor distribution stats
            rewards.append(total_reward)
            schedules.append(env.assigned_tasks)
        
    best_idx = np.argmin(makespans)
    return np.mean(makespans), np.mean(balances), np.mean(rewards), schedules[best_idx]

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
        total_updates = int(configs.max_episodes / configs.update_every_episodes)

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
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pth")
        
        if args.resume and os.path.exists(checkpoint_path):
            print(f"正在从 {checkpoint_path} 恢复训练...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                agent.policy.load_state_dict(checkpoint['model_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint['episode'] + 1
                print(f"恢复成功. 起始 Episode: {start_episode}")
            except RuntimeError as e:
                print(f"⚠️ 恢复失败: 模型结构不匹配 (可能是 configs 修改了层数/维度). 跳过恢复。\n报错信息截取: {str(e)[:100]}...")
        
        # 最佳模型记录
        best_makespan = float('inf')
        best_model_dir = os.path.join(model_dir, "bestmodel")
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_path = os.path.join(best_model_dir, "best_model.pth")
        
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
        
        for ep in range(start_episode, configs.max_episodes + 1):
            
            # [Temperature Annealing] 计算当前环境的探索“温度”
            decay_ratio = min(1.0, (ep - 1) / max(1, getattr(configs, 'temp_decay_episodes', 2000)))
            current_temp = getattr(configs, 'temp_start', 2.0) - decay_ratio * (getattr(configs, 'temp_start', 2.0) - getattr(configs, 'temp_end', 0.1))
            
            # [Domain Randomization] 如果配置开启泛化性抗干扰，则给环境施加动态工时噪音
            apply_noise = getattr(configs, 'randomize_durations', False)
            state = env.reset(randomize_duration=apply_noise, randomize_workers=apply_noise)
            
            done = False
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
                     # [Fix]: massive penalty (-10000.0) so it never prefers "suicide over working"
                     reward = -10000.0 
                     done = True
                     # 记录这一步以供学习 (改为轻量级 Snapshot)
                     memory.states.append(env.get_state_snapshot())
                     memory.actions.append((0,0,[])) # Dummy Action
                     memory.logprobs.append(torch.tensor(0.0))
                     memory.rewards.append(reward)
                     memory.is_terminals.append(done)
                     memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
                     memory.values.append(0.0) # Deadlock value
                     ep_reward += reward
                     break
                
                if w_mask.all():
                     # 所有工人都在忙，理论上 _advance_time 会跳过这段时间，
                     # 但如果出现这种情况，说明时间推进逻辑可能需要检查。
                     # 这里仅作警告。
                     pass
                
                # 选择动作 (Stochastic with Annealed Temperature)
                action, logprob, val, _ = agent.select_action(
                    state.to(device), 
                    mask_task=t_mask, 
                    mask_station_matrix=s_mask,
                    mask_worker=w_mask,
                    deterministic=False,
                    temperature=current_temp
                )
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储轨迹 (取代原有几兆的完整异构图对象，只存极小的快照)
                memory.states.append(env.get_state_snapshot()) 
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
                memory.values.append(val)
                
                state = next_state
                ep_reward += reward
                
                if done:
                    break
            
            # 提取每次训练结束时的实时完工耗时
            ep_makespan = np.max(env.station_wall_clock) if len(env.assigned_tasks) > 0 else 0.0
            is_deadlock = len(env.assigned_tasks) < env.num_tasks
            status_str = "[DEADLOCK]" if is_deadlock else "[COMPLETED]"
            
            # 记录日志
            writer.add_scalar('Reward/Episode', ep_reward, ep)
            writer.add_scalar('Train/WallClock_Makespan', ep_makespan, ep)
            
            print(f"Episode {ep} {status_str} | Reward: {ep_reward:.2f} | Steps: {t+1} | Makespan: {ep_makespan:.1f}")
            
            # PPO 更新
            if ep % update_every_episodes == 0:
                metrics = agent.update(memory, env)
                memory.clear()
                
                for k, v in metrics.items():
                    writer.add_scalar(k, v, ep)
                
            # 定期评估与保存
            if ep % eval_freq == 0:
                # 在训练过程中，使用较少的多轮评估 (如 3 轮)，带有极小温度 (如 0.0) 以检测绝对贪婪上线
                makespan, balance, eval_reward, best_sch = evaluate_model(env, agent, num_runs=3, temperature=configs.eval_temperature)
                
                print(f"[Eval] Ep {ep} | Avg Real-Makespan: {makespan:.1f} | Avg Workload-Balance_Std: {balance:.2f} | AvgReward: {eval_reward:.2f}")
                
                writer.add_scalar('Eval/WallClock_Makespan', makespan, ep)
                writer.add_scalar('Eval/Workload_Balance_Std', balance, ep)
                
                # Save Latest
                torch.save({
                    'episode': ep,
                    'model_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, checkpoint_path)
                
                # Save Best
                if makespan < best_makespan:
                    best_makespan = makespan
                    torch.save(agent.policy.state_dict(), best_model_path)
                    print(f"New Best Model Saved! Makespan: {best_makespan}")
                    
                    # [Real-time Tracer] 实时快照抓拍最好成绩的排单策略
                    trace_dir = "results/eval_traces"
                    os.makedirs(trace_dir, exist_ok=True)
                    if best_sch:
                        tasks_data = []
                        for (tid, sid, team, start, end) in best_sch:
                             tasks_data.append({
                                 'TaskID': tid,
                                 'StationID': sid + 1,
                                 'Team': str(team),
                                 'Start': start,
                                 'End': end,
                                 'Duration': end - start
                             })
                        df = pd.DataFrame(tasks_data)
                        df.to_csv(os.path.join(trace_dir, f"Ep_{ep}_Best_Schedule.csv"), index=False)
                        plot_gantt(best_sch, os.path.join(trace_dir, f"Ep_{ep}_Gantt.png"))
                        print(f"📸 Real-time Schedule Trace exported to {trace_dir}/Ep_{ep}_Gantt.png")
                        
                    
        # =======================================================================
        # 6. 训练结束 - 终局性能测评与基线对比 (End of Training Evaluation)
        # =======================================================================
        print("\n" + "="*50)
        print("🎉 强化学习训练循环已结束！开始获取最强方案对比基线。")
        print("="*50)
        
        # 加载最好验证参数
        if os.path.exists(best_model_path):
             print(f"加载训练历史上最好的验证模型用于最终推演: {best_model_path}")
             try:
                 model.load_state_dict(torch.load(best_model_path, map_location=device))
             except RuntimeError as e:
                 print(f"⚠️ 警告: 历史最佳模型 ({best_model_path}) 的结构与当前配置不匹配，无法加载。将继续使用当前最新的训练结果进行推演！")
             
        # 配置 PPO 最终推演
        print("\n>>> [1/2] 开始执行 PPO Agent 的终局推演...")
        # 重新实例环境，避免脏数据
        eval_env = AirLineEnv_Graph(data_path=data_path, seed=2026)
        ppo_makespan, ppo_balance, _, ppo_assigned = evaluate_model(eval_env, agent, num_runs=5, temperature=configs.eval_temperature)
        
        # 配置 GA 基准对抗
        print("\n>>> [2/2] 开始执行 Genetic Algorithm (GA) 基线推演...")
        ga_env = AirLineEnv_Graph(data_path=data_path, seed=2026)
        ga_scheduler = GeneticAlgorithmScheduler(ga_env, pop_size=30, max_gen=20)
        ga_makespan, ga_balance, ga_assigned = ga_scheduler.run()
        
        # --- 报表总结生成 ---
        print("\n" + "#"*50)
        print("🚀 终局对比结果报告 (PPO vs GA) 🚀")
        print(f"指标说明：Makespan (越小越好), Balance (越小越好)")
        print("-"*50)
        print(f"| 模型算法类型          | Makespan (h) | Balance Std |")
        print(f"|-----------------------|--------------|-------------|")
        print(f"| 经典运筹学: (GA 基线) | {ga_makespan:12.2f} | {ga_balance:11.2f} |")
        print(f"| 强化学习: (HB-GAT-PN) | {ppo_makespan:12.2f} | {ppo_balance:11.2f} |")
        print("#"*50 + "\n")
        
        # 导出最佳 PPO 细节到 CSV 及画图
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        def save_schedule(tasks, prefix_name):
            if not tasks: return
            tasks_data = []
            for (tid, sid, team, start, end) in tasks:
                 tasks_data.append({
                     'TaskID': tid,
                     'StationID': sid + 1,
                     'Team': str(team),
                     'Start': start,
                     'End': end,
                     'Duration': end - start
                 })
            df = pd.DataFrame(tasks_data)
            df.to_csv(os.path.join(output_dir, f"{prefix_name}_schedule.csv"), index=False)
            plot_gantt(tasks, os.path.join(output_dir, f"{prefix_name}_gantt.png"))
            
        print(f"正在向目录 ./{output_dir} 保存排程细节与甘特图...")
        save_schedule(ppo_assigned, "PPO_Final")
        save_schedule(ga_assigned, "GA_Baseline")
        print("所有流程圆满结束！")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()
    
    train(args)
