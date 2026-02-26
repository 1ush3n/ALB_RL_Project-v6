import sys
import os
import torch
import networkx as nx
import pandas as pd
import numpy as np

# Add project root to path
# Assuming we are running inside ALB_RL_Project/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from ALB_RL_Project.data_loader import load_data
except ImportError:
    # Try adding subdirectory
    sys.path.append(os.path.join(project_root, 'ALB_RL_Project'))
    from data_loader import load_data

def verify_graph(file_path):
    print(f"\n[Verification] Loading data from {file_path}...")
    try:
        data = load_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df = data['task_df']
    edges_tensor = data['precedence_edges'] # [2, E]
    edges = edges_tensor.t().numpy() # [E, 2]
    
    print(f"Loaded {len(df)} tasks.")
    print(f"Loaded {edges.shape[0]} edges.")
    
    # 1. Build NetworkX Graph
    G = nx.DiGraph()
    # Add nodes explicitly to ensure all are present
    G.add_nodes_from(range(len(df)))
    G.add_edges_from(edges)
    
    # 2. Check for Cycles (DAG)
    print("Checking for cycles...")
    try:
         cycle = nx.find_cycle(G, orientation='original')
         print(f"[FAIL] CYCLES DETECTED! Example: {cycle}")
         # Map internal IDs back to task IDs for readability
         cycle_names = [df.iloc[u[0]]['task_id'] for u in cycle]
         print(f"   Cycle Tasks: {cycle_names}")
    except nx.NetworkXNoCycle:
         print("[OK] Graph is a DAG (No Cycles).")

    # 3. Check Connectivity
    num_components = nx.number_weakly_connected_components(G)
    print(f"Weakly Connected Components: {num_components}")
    
    # 4. Check specific logic (Rule A, B, C)
    try:
        if 'A' in df['task_id'].values and 'A-1' in df['task_id'].values:
            root_a_idx = df[df['task_id'] == 'A'].index[0]
            sub_a1_idx = df[df['task_id'] == 'A-1'].index[0]
            if G.has_edge(root_a_idx, sub_a1_idx):
                 print(f"[OK] Rule E Success: 'A' -> 'A-1'")
            else:
                 print(f"[FAIL] Rule E Failed: 'A' -> 'A-1' Missing")
        else:
            print("[INFO] Nodes 'A' or 'A-1' not found, skipping specific rule check.")

    except IndexError as e:
        print(f"[WARN] Specific check skipped: {e}")
    except Exception as e:
        print(f"[WARN] Specific check error: {e}")

    # 5. Zero Duration Check
    zero_dur_tasks = df[df['duration'] == 0]
    print(f"Zero Duration Tasks Count: {len(zero_dur_tasks)}")

if __name__ == "__main__":
    # Check current directory for 3000.csv
    # We are in ALB_RL_Project/
    files = ["../3000.csv", "3000.csv"]
    target = None
    for f in files:
        if os.path.exists(f): 
            target = f
            print(f"Found target file: {f}")
            break
            
    if target:
        verify_graph(target)
    else:
        print("Could not find 3000.csv in likely locations.")
