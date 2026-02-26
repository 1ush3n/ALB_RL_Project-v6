
import os
import sys
import torch
import pandas as pd
import numpy as np
import glob

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ALB_RL_Project.environment import AirLineEnv_Graph
from ALB_RL_Project.models.hb_gat_pn import HBGATPN
from ALB_RL_Project.ppo_agent import PPOAgent
from ALB_RL_Project.configs import configs

def find_latest_checkpoint(model_dir):
    list_of_files = glob.glob(os.path.join(model_dir, '*.pth')) 
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def generate_schedule(model_path=None):
    print("--- Generating Schedule (Deterministic) ---")
    
    # Init Env
    env = AirLineEnv_Graph(data_path=configs.DATASET_PATH) # Use config default
    print(f"Dataset: {configs.DATASET_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Model
    model = HBGATPN(configs).to(device)
    
    if model_path is None:
        model_dir = "ALB_RL_Project/models"
        model_path = find_latest_checkpoint(model_dir)
        
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No checkpoint found or specified. Using random weights.")
        
    agent = PPOAgent(model, configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip, device)
    
    # Rollout
    state = env.reset()
    done = False
    total_reward = 0
    
    # Track Schedule Details
    # env.assigned_tasks already has (task_id, station_id, worker_id, start, end)
    # But internal_id. Need mapping back to original?
    # env.raw_data['id_map'] is {orig: internal}.
    # We need internal -> orig.
    
    id_map = env.raw_data['id_map']
    internal_to_orig = {v: k for k, v in id_map.items()}
    
    while not done:
        # Mask
        task_mask, station_mask = env.get_masks()
        t_mask = task_mask.to(device)
        s_mask = station_mask.to(device)
        
        # Action
        action, _, _, _ = agent.select_action(
            state.to(device),
            mask_task=t_mask,
            mask_station_matrix=s_mask,
            deterministic=True
        )
        
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
    print("--- Rollout Complete ---")
    
    # Metrics
    station_loads = env.station_loads
    makespan = np.max(station_loads)
    balance_std = np.std(station_loads)
    
    print(f"Makespan: {makespan:.2f}")
    print(f"Station Balance (Std): {balance_std:.2f}")
    
    # Export CSV
    # Required: Task_ID, Station_ID, Worker_ID, Task_Start_Time, Task_End_Time, Worker_Start/End, Work_Station
    
    records = []
    
    # env.assigned_tasks: list of (task_id, station_id, worker_id, start_time, finish_time)
    # Be careful: assigned_tasks are appended in order of assignment decision, not necessarily execution start time.
    # But we want row per task.
    
    for (t_int, s_id, w_id, start, end) in env.assigned_tasks:
        orig_id = internal_to_orig.get(t_int, t_int)
        
        records.append({
            "Task_ID": orig_id,
            "Station_ID": s_id,
            "Worker_ID": w_id,
            "Task_Start_Time": start,
            "Task_End_Time": end,
            "Worker_Start_Work_Time": start,
            "Worker_End_Work_Time": end,
            "Work_Station": s_id # Worker is at this station for this task
        })
        
    df = pd.DataFrame(records)
    df = df.sort_values("Task_Start_Time")
    
    out_file = "final_schedule.csv"
    df.to_csv(out_file, index=False)
    print(f"Schedule saved to: {out_file}")
    
    return df

if __name__ == "__main__":
    generate_schedule()
