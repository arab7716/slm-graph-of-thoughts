# Evaluation script to calculate error rates for thought nodes in Generate() and Aggregate() steps 
# Code syntax generation partially aided by generative AI. 
# All code thoroughly human reviewed.
import os
import json
import argparse
from collections import defaultdict


TARGET_FOLDERS =[
   # insert folder paths
]

def analyze_baseline_bottlenecks():
    if not TARGET_FOLDERS:
        print("Please add folder paths to TARGET_FOLDERS.")
        return

    # stats[task][operation] = {"total": 0, "failed": 0}
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "failed": 0}))

    for folder_path in TARGET_FOLDERS:
        if not os.path.exists(folder_path): continue

        task_type = "UNKNOWN"
        if "sorting" in folder_path.lower(): task_type = "SORTING"
        elif "keyword" in folder_path.lower(): task_type = "KEYWORDS"
        elif "set_intersection" in folder_path.lower(): task_type = "SETS"

        methods_found = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        # look just at the unmitigated original baselines
        target_methods = [m for m in methods_found if "original" in m.lower()]

        for method in target_methods:
            method_path = os.path.join(folder_path, method)
            files = [f for f in os.listdir(method_path) if f.endswith(".json")]
            
            for f_name in files:
                try:
                    with open(os.path.join(method_path, f_name), 'r') as f: 
                        data = json.load(f)
                    
                    last_op_type = None
                    
                    for item in data:
                        if not isinstance(item, dict): continue
                        op = item.get("operation", "").lower()
                        
                        # Track if the last operation was Generate or Aggregate
                        if op in ["generate", "aggregate"]:
                            last_op_type = op
                            
                        # score-- evaluate
                        elif op == "score" and last_op_type in ["generate", "aggregate"]:
                            scores = item.get("scores", [])
                            if not scores: continue
                            
                            stats[task_type][last_op_type]["total"] += len(scores)
                            
                            # 0 = correct, >0 is error
                            failures = sum(1 for s in scores if isinstance(s, (int, float)) and s > 0)
                            stats[task_type][last_op_type]["failed"] += failures
                            
                            # reset
                            last_op_type = None
                                
                except Exception as e:
                    continue

    #print
    print("\n" + "="*80)
    print(f" BASELINE BOTTLENECKS (got_original)")
    print("="*80)
    print(f"{'Task':<15} | {'Operation':<12} | {'Total Executions':<18} | {'Failed (Score > 0)':<18} | {'Failure Rate %':<20}")
    print("-" * 80)

    for task, ops in stats.items():
        for op, counts in ops.items():
            total = counts["total"]
            failed = counts["failed"]
            if total == 0: continue
            
            rate = (failed / total) * 100
            print(f"{task:<15} | {op.upper():<12} | {total:<18} | {failed:<18} | {rate:>5.1f}% ")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="Specific folder to analyze")
    args = parser.parse_args()
    if args.folder: TARGET_FOLDERS = [args.folder]
    analyze_baseline_bottlenecks()