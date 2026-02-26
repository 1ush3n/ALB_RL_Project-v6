import torch
import numpy as np
import os
import sys

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs

def debug_eval():
    data_path = "100.csv"
    if not os.path.exists(data_path):
         data_path = os.path.join(os.getcwd(), data_path)
    
    print(f"Loading environment with {data_path}")
    env = AirLineEnv_Graph(data_path=data_path, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HBGATPN(configs).to(device)
    
    best_model_path = "models/best_model.pth"
    if os.path.exists(best_model_path):
        print(f"Loading {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("No trained model found, using random weights.")
        
    # fix: set lr > 0 to avoid Division by Zero in Step Scheduler 
    agent = PPOAgent(model, lr=1e-5, gamma=0.99, k_epochs=4, eps_clip=0.2, device=device)
    
    state = env.reset()
    done = False
    
    total_reward = 0
    total_duration_penalty = 0
    total_blocking_penalty = 0
    total_makespan_penalty = 0
    
    step = 0
    while not done:
        task_mask, station_mask, worker_mask = env.get_masks()
        
        action, _, _, _ = agent.select_action(
            state.to(device), 
            mask_task=task_mask.to(device), 
            mask_station_matrix=station_mask.to(device),
            mask_worker=worker_mask.to(device),
            deterministic=False,
            temperature=1.0
        )
        
        prev_makespan = np.max(env.station_loads)
        state, reward, done, _ = env.step(action)
        
        # Calculate components manually to verify
        task_id, station_id, team = action
        duration = env.calculate_duration(task_id, team)
        
        dur_pen = -0.1 * duration
        
        new_makespan = np.max(env.station_loads)
        make_pen = -0.5 * (new_makespan - prev_makespan)
        
        # In environment, reward = 1.0 + dur_pen + block_pen + make_pen (+ maybe st_std)
        block_pen = reward - (1.0 + dur_pen + make_pen)
        if done:
            st_std = np.std(env.station_loads)
            block_pen += (0.5 * st_std)
            
        total_duration_penalty += dur_pen
        total_blocking_penalty += block_pen
        total_makespan_penalty += make_pen
        
        total_reward += reward
        step += 1
        
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"  Completion Payout: {step * 1.0:.2f}")
    print(f"  Duration Penalty: {total_duration_penalty:.2f}")
    print(f"  Makespan Penalty: {total_makespan_penalty:.2f} (Final Makespan: {np.max(env.station_loads):.2f})")
    print(f"  Blocking Penalty: {total_blocking_penalty:.2f}")
    if done:
        print(f"  Balance Penalty: {-0.5 * np.std(env.station_loads):.2f} (Std: {np.std(env.station_loads):.2f})")

if __name__ == "__main__":
    debug_eval()
