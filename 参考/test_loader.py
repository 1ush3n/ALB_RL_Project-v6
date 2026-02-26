
import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ALB_RL_Project.data_loader import load_data
from ALB_RL_Project.configs import configs

def test():
    file_path = "3000.xlsx" # Hardcode for check
    print(f"Testing load_data with: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
        
    try:
        data = load_data(file_path)
        print("Success!")
        print(f"Num Tasks: {data['num_tasks']}")
        print(f"Num Edges: {data['precedence_edges'].shape[1]}")
        print(f"Columns: {data['task_df'].columns.tolist()}")
        print(f"First 5 IDs: {data['task_df']['task_id'].head().tolist()}")
        print(f"Internal IDs sorted check: {data['task_df']['internal_id'].head().tolist()}")
        
        # Check if internal_id is sorted
        internals = data['task_df']['internal_id'].values
        if (internals == sorted(internals)).all():
            print("Verified: DataFrame is sorted by internal_id.")
        else:
            print("ERROR: DataFrame is NOT sorted by internal_id.")
            
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
