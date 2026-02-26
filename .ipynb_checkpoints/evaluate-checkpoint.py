import torch
import numpy as np
import argparse
import sys
import os
import pandas as pd

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs
from utils.visualization import plot_gantt

def evaluate(args):
    print("--- Starting Evaluation ---")
    
    # 1. Init Env
    data_path = args.data_path if args.data_path else configs.data_file_path
    # Ensure path
    if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
            data_path = os.path.join(os.getcwd(), data_path)
    
    print(f"Loading data from: {data_path}")
    print(f"Loading data from: {data_path}")
    env = AirLineEnv_Graph(data_path=data_path, seed=42)
    print("Environment Initialized.")
    
    # Update config dims if needed (though model loading handles weights, robust to update)
    # configs.n_j = env.num_tasks
    
    # 2. Init Model & Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = HBGATPN(configs).to(device)
    
    # Load Checkpoint
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model file not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    # Handle both full checkpoint dict and direct state_dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from Episode {checkpoint['episode']}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded pure state_dict.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Agent wrapper (for select_action)
    agent = PPOAgent(model, lr=0, gamma=0, k_epochs=0, eps_clip=0, device=device)
    
    # 3. Run Inference (Deterministic)
    state = env.reset()
    done = False
    total_reward = 0
    
    print("Running determinisitic inference...")
    while not done:
        task_mask, station_mask, worker_mask = env.get_masks()
        
        action, _, _, _ = agent.select_action(
            state.to(device),
            mask_task=task_mask.to(device),
            mask_station_matrix=station_mask.to(device),
            mask_worker=worker_mask.to(device),
            deterministic=True
        )
        
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
    # 4. Results
    makespan = np.max(env.station_loads)
    balance_std = np.std(env.station_loads)
    print("--- Evaluation Results ---")
    print(f"Makespan: {makespan:.2f} h")
    print(f"Load Balance (Std): {balance_std:.2f}")
    print(f"Total Reward: {total_reward:.4f}")
    
    # 5. Export Schedule
    # assigned_tasks: [(task_id, station_id, team, start, end), ...]
    # Convert to DataFrame
    tasks_data = []
    for (tid, sid, team, start, end) in env.assigned_tasks:
        tasks_data.append({
            'TaskID': tid,
            'StationID': sid + 1, # 1-based
            'Team': str(team),
            'Start': start,
            'End': end,
            'Duration': end - start
        })
    
    df_res = pd.DataFrame(tasks_data)
    csv_path = "schedule_result.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"Detailed schedule saved to {csv_path}")
    
    # 6. Visualization
    png_path = "schedule_gantt.png"
    print("Generating Gantt Chart...")
    plot_gantt(env.assigned_tasks, png_path)
    
    print("Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--data_path', type=str, default=None, help='Path to .csv data file')
    
    args = parser.parse_args()
    evaluate(args)
