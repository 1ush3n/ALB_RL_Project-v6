import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
import torch

# 添加根路径以便导入外部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from environment import AirLineEnv_Graph
from utils.visualization import plot_gantt

def run_spt(args):
    print(f"--- 启动 Heuristic SPT (Shortest Processing Time) 基准测试 ---")
    print(f"数据集: {args.data_path}")
    
    env = AirLineEnv_Graph(data_path=args.data_path, seed=2026)
    env.reset(randomize_duration=False)
    done = False
    
    # 提取所有任务的时长用于排序
    durations = env.task_static_feat[:, 0].numpy()
    
    start_time = time.time()
    
    while not done:
        task_mask, station_mask, worker_mask = env.get_masks()
        
        if task_mask.all():
            print("[SPT] 检测到死锁！任务掩码全满。")
            break
            
        # 1. 寻找合法任务中处理时间最短的
        valid_tasks = torch.where(~task_mask)[0].numpy()
        valid_tasks_sorted = sorted(valid_tasks, key=lambda t: durations[t])
        t_idx = valid_tasks_sorted[0]
        
        # 2. 选择合法的也是最早可用的工位
        valid_stations = torch.where(~station_mask[t_idx])[0].numpy()
        s_idx = valid_stations[0]
        
        # 3. 选择工人
        # 需要获取技能和锁定掩码
        worker_feats = env.obs_data['worker'].x
        worker_skills = worker_feats[:, 1:11]
        task_type_idx = torch.argmax(env.obs_data['task'].x[t_idx, 5:15]).item()
        demand = int(env.obs_data['task'].x[t_idx, -1].item())
        demand = max(1, demand)
        
        has_skill = worker_skills[:, task_type_idx] > 0.5
        worker_locks = torch.argmax(worker_feats[:, 12:20], dim=1)
        
        valid_workers = []
        for w in range(env.num_workers):
            if worker_mask[w].item(): continue
            if not has_skill[w].item(): continue
            lock = worker_locks[w].item()
            if lock != 0 and lock != (s_idx + 1): continue
            valid_workers.append(w)
            
        if len(valid_workers) >= demand:
             team = valid_workers[:demand]
        else:
             # SPT 不带后退机制，选不够说明启发式规则走入了死胡同 (类似随机死锁)
             team = valid_workers + [0] * (demand - len(valid_workers))
             
        action = (t_idx, s_idx, team)
        state, reward, done, _ = env.step(action)
        
    end_time = time.time()
    duration = end_time - start_time
    
    if task_mask.all():
        makespan = 99999.0
        balance = 9999.0
        schedules = []
    else:
        makespan = np.max(env.station_wall_clock)
        balance = np.std(env.station_loads)
        schedules = env.assigned_tasks
        
    print("\n" + "#"*50)
    print("🚀 Heuristic (SPT) 运行结果 🚀")
    print("-" * 50)
    print(f"Makespan: {makespan:.2f} 小时")
    print(f"Balance Std: {balance:.2f} 小时")
    print(f"Inference Time: {duration:.4f} 秒")
    print("#"*50 + "\n")
    
    # 保存结果
    output_dir = os.path.join(parent_dir, "results", "Baselines", "SPT")
    os.makedirs(output_dir, exist_ok=True)
    
    if schedules:
        plot_gantt(schedules, os.path.join(output_dir, "SPT_gantt.png"))
        tasks_data = []
        for (tid, sid, team, start, end) in schedules:
              tasks_data.append({
                  'TaskID': tid,
                  'StationID': sid + 1,
                  'Team': str(team),
                  'Start': start,
                  'End': end,
                  'Duration': end - start
              })
        df = pd.DataFrame(tasks_data)
        df.to_csv(os.path.join(output_dir, "SPT_schedule.csv"), index=False)
        print(f"已导出甘特图与明细至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/100.csv", help='Path to the dataset')
    args = parser.parse_args()
    run_spt(args)
