import torch
import numpy as np
import argparse
import sys
import os
import pandas as pd

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ALB_RL_Project.environment import AirLineEnv_Graph
from ALB_RL_Project.models.hb_gat_pn import HBGATPN
from ALB_RL_Project.ppo_agent import PPOAgent
from ALB_RL_Project.configs import configs
from ALB_RL_Project.utils.visualization import plot_gantt

def evaluate(args):
    """
    模型评估脚本。
    
    功能:
    1. 加载训练好的模型 (.pth)。
    2. 在环境中运行一轮确定性推理 (Deterministic Inference)。
    3. 输出评估指标 (Makespan, Balance)。
    4. 生成排程结果 CSV 和甘特图 PNG。
    """
    print("--- 开始评估 (Starting Evaluation) ---")
    
    # 1. 加载数据与环境
    data_path = args.data_path if args.data_path else configs.data_file_path
    if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
            data_path = os.path.join(os.getcwd(), data_path)
    
    print(f"数据路径: {data_path}")
    env = AirLineEnv_Graph(data_path=data_path, seed=42)
    print("环境初始化完成.")
    
    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = HBGATPN(configs).to(device)
    
    # 3. 加载 Checkpoint
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        return

    print(f"加载模型: {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 支持加载完整 checkpoint 或仅 state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载 Checkpoint (Episode {checkpoint.get('episode', 'Unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("已加载 State Dict.")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # Agent 包装 (主要为了使用 select_action 方法)
    agent = PPOAgent(model, lr=0, gamma=0, k_epochs=0, eps_clip=0, device=device)
    
    # 4. 执行推理
    state = env.reset()
    done = False
    total_reward = 0
    
    print("正在执行推理...")
    while not done:
        task_mask, station_mask, worker_mask = env.get_masks()
        
        # 确定性选择
        action, _, _, _ = agent.select_action(
            state.to(device),
            mask_task=task_mask.to(device),
            mask_station_matrix=station_mask.to(device),
            mask_worker=worker_mask.to(device),
            deterministic=True
        )
        
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
    # 5. 计算指标
    makespan = np.max(env.station_loads)
    balance_std = np.std(env.station_loads)
    print("--- 评估结果 ---")
    print(f"最大完工时间 (Makespan): {makespan:.2f} h")
    print(f"站位负载标准差 (Balance Std): {balance_std:.2f}")
    print(f"总奖励 (Total Reward): {total_reward:.4f}")
    
    # 6. 导出排程结果
    # env.assigned_tasks: [(task_id, station_id, team, start, end), ...]
    tasks_data = []
    for (tid, sid, team, start, end) in env.assigned_tasks:
        tasks_data.append({
            'TaskID': tid,
            'StationID': sid + 1, # 转换为 1-based 用于展示
            'Team': str(team),
            'Start': start,
            'End': end,
            'Duration': end - start
        })
    
    df_res = pd.DataFrame(tasks_data)
    csv_path = "schedule_result.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"详细排程表已保存至: {csv_path}")
    
    # 7. 生成甘特图
    png_path = "schedule_gantt.png"
    print("正在生成甘特图...")
    plot_gantt(env.assigned_tasks, png_path)
    print(f"甘特图已保存至: {png_path}")
    
    print("评估流程结束.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径 (.pth)')
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径 (.csv)')
    
    args = parser.parse_args()
    evaluate(args)
