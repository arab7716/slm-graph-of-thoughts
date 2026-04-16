# Evaluation script to calculate diversity of thoughts in GoT baseline 
# Code syntax generation partially aided by generative AI. 
# All code thoroughly human reviewed.

import os
import json
import difflib
import argparse
from collections import defaultdict

TARGET_FOLDERS =[
    # insert folder paths here
]

def calc_similarity(strings):
    if len(strings) < 2: return 1.0
    total, pairs = 0.0, 0
    for i in range(len(strings)):
        for j in range(i+1, len(strings)):
            total += difflib.SequenceMatcher(None, strings[i], strings[j]).ratio()
            pairs += 1
    return total / pairs

def analyze_diversity():
    # stats[task][temp] = {"total_ops": 0, "sum_k": 0, "sum_unique": 0, "sum_sim": 0.0}
    stats = defaultdict(lambda: defaultdict(lambda: {"total_ops": 0, "sum_k": 0, "sum_unique": 0, "sum_sim": 0.0}))

    for folder_path in TARGET_FOLDERS:
        if not os.path.exists(folder_path): continue

        task_type = "UNKNOWN"
        if "sorting" in folder_path.lower(): task_type = "SORTING"
        elif "keyword" in folder_path.lower(): task_type = "KEYWORDS"
        elif "set_intersection" in folder_path.lower(): task_type = "SETS"

        temp = "T=0.1" if "T0p1" in folder_path else "T=0.6" if "T0p6" in folder_path else "UNKNOWN"

        methods_found = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        # look only at got original
        target_methods = [m for m in methods_found if "original" in m.lower()]

        for method in target_methods:
            method_path = os.path.join(folder_path, method)
            files = [f for f in os.listdir(method_path) if f.endswith(".json")]
            
            for f_name in files:
                try:
                    with open(os.path.join(method_path, f_name), 'r') as f: 
                        data = json.load(f)
                    
                    for item in data:
                        if not isinstance(item, dict): continue
                        op = item.get("operation", "").lower()
                        
                        if op in ["generate", "aggregate"]:
                            thoughts = item.get("thoughts", [])
                            # look only where multiple branches generated
                            if len(thoughts) > 1:
                                strings = [str(t.get("current", "")).strip() for t in thoughts]
                                
                                k = len(strings)
                                unique_count = len(set(strings))
                                sim = calc_similarity(strings)
                                
                                stats[task_type][temp]["total_ops"] += 1
                                stats[task_type][temp]["sum_k"] += k
                                stats[task_type][temp]["sum_unique"] += unique_count
                                stats[task_type][temp]["sum_sim"] += sim
                                
                except Exception as e:
                    continue

    # print
    print("\n" + "="*80)
    print(f"Thought Diversity Analysis")
    print("="*80)
    print(f"{'Task':<15} | {'Temp':<7} | {'Avg Branches (k)':<18} | {'Avg Unique Thoughts':<20} | {'Pairwise Similarity':<18}")
    print("-" * 80)

    for task in ["SORTING", "SETS", "KEYWORDS"]:
        if task not in stats: continue
        for temp in ["T=0.1", "T=0.6"]:
            if temp not in stats[task]: continue
            
            data = stats[task][temp]
            ops = data["total_ops"]
            if ops == 0: continue
            
            avg_k = data["sum_k"] / ops
            avg_u = data["sum_unique"] / ops
            avg_sim = (data["sum_sim"] / ops) * 100
            
            print(f"{task:<15} | {temp:<7} | {avg_k:<18.1f} | {avg_u:<20.1f} | {avg_sim:>5.1f}%")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="Specific folder to analyze")
    args = parser.parse_args()
    if args.folder: TARGET_FOLDERS = [args.folder]
    analyze_diversity()