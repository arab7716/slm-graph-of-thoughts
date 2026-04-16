# Evaluation script to calculate diversity of thoughts in GoT ablations-- python_moe, and python_no_moe 
# Code syntax generation partially aided by generative AI. 
# All code thoroughly human reviewed.

import os
import json
import difflib
import argparse
from collections import defaultdict

TARGET_FOLDERS =[
    "examples/sorting/results/qwen2.5-14b_T0p1_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-45-14",
    "examples/sorting/results/qwen2.5-14b_T0p6_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-45-14",
    "examples/keyword_counting/results/qwen2.5-14b_T0p1_io-cot-got4_original-got4_2_nodes-got4_python_moe-got4_python_no_moe-got4_llm_no_moe-got4_full_2026-04-13_22-45-14_528054",
    "examples/keyword_counting/results/qwen2.5-14b_T0p6_io-cot-got4_original-got4_2_nodes-got4_python_moe-got4_python_no_moe-got4_llm_no_moe-got4_full_2026-04-13_22-45-14_528056",
    "examples/set_intersection/results/qwen2.5-14b_T0p1_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-46-10_478224",
    "examples/set_intersection/results/qwen2.5-14b_T0p6_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-46-10_478222",
]

def calc_similarity(strings):
    if len(strings) < 2: return 1.0
    total, pairs = 0.0, 0
    for i in range(len(strings)):
        for j in range(i+1, len(strings)):
            total += difflib.SequenceMatcher(None, strings[i], strings[j]).ratio()
            pairs += 1
    return total / pairs

def analyze_diversity_comparison():
    # stats[task][temp][method] = {"total_ops": 0, "sum_sim": 0.0}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"total_ops": 0, "sum_sim": 0.0})))

    for folder_path in TARGET_FOLDERS:
        if not os.path.exists(folder_path): continue

        task_type = "UNKNOWN"
        if "sorting" in folder_path.lower(): task_type = "SORTING"
        elif "keyword" in folder_path.lower(): task_type = "KEYWORDS"
        elif "set_intersection" in folder_path.lower(): task_type = "SETS"

        temp = "T=0.1" if "T0p1" in folder_path else "T=0.6" if "T0p6" in folder_path else "UNKNOWN"

        methods_found = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        # original and python validated ablations
        target_methods = [m for m in methods_found if "original" in m.lower() or "python" in m.lower()]

        for method in target_methods:
            if "original" in method.lower(): m_key = "BASELINE"
            elif "python_moe" in method.lower() and "no_moe" not in method.lower(): m_key = "MOE"
            elif "python_no_moe" in method.lower(): m_key = "NO_MOE"
            else: continue
            
            method_path = os.path.join(folder_path, method)
            for f_name in os.listdir(method_path):
                if not f_name.endswith(".json"): continue
                try:
                    with open(os.path.join(method_path, f_name), 'r') as f: 
                        data = json.load(f)
                    
                    for item in data:
                        if not isinstance(item, dict): continue
                        op = item.get("operation", "").lower()
                        
                        if op in ["generate", "aggregate"]:
                            thoughts = item.get("thoughts", [])
                            
                            strings_to_compare = []
                            
                            # baseline -- look at all thoughts
                            if m_key == "BASELINE":
                                if len(thoughts) > 1:
                                    strings_to_compare = [str(t.get("current", "")).strip() for t in thoughts]
                            
                            # interventions-- just look at intervened branches
                            else:
                                if any(t.get("intervened") for t in thoughts):
                                    strings_to_compare = [str(t.get("current", "")).strip() for t in thoughts if t.get("intervened")]
                            
                            if len(strings_to_compare) > 1:
                                sim = calc_similarity(strings_to_compare)
                                stats[task_type][temp][m_key]["total_ops"] += 1
                                stats[task_type][temp][m_key]["sum_sim"] += sim
                                
                except Exception as e:
                    continue

    # print
    print("\n" + "="*110)
    print(f"Thought Diversity Comparison: Baseline vs. Generic Retry vs. MoE")
    print("="*110)
    print(f"{'Task':<15} | {'Temp':<7} | {'Baseline (Original)':<25} | {'Generic Retry':<25} | {'MoE Lenses':<25}")
    print("-" * 110)

    for task in ["SORTING", "SETS", "KEYWORDS"]:
        if task not in stats: continue
        for temp in ["T=0.1", "T=0.6"]:
            if temp not in stats[task]: continue
            
            base_data = stats[task][temp].get("BASELINE", {"total_ops": 0, "sum_sim": 0.0})
            no_moe_data = stats[task][temp].get("NO_MOE", {"total_ops": 0, "sum_sim": 0.0})
            moe_data = stats[task][temp].get("MOE", {"total_ops": 0, "sum_sim": 0.0})
            
            base_sim = (base_data["sum_sim"] / base_data["total_ops"] * 100) if base_data["total_ops"] > 0 else 0.0
            no_moe_sim = (no_moe_data["sum_sim"] / no_moe_data["total_ops"] * 100) if no_moe_data["total_ops"] > 0 else 0.0
            moe_sim = (moe_data["sum_sim"] / moe_data["total_ops"] * 100) if moe_data["total_ops"] > 0 else 0.0
            
            # formatting
            base_str = f"{base_sim:>5.1f}% (n={base_data['total_ops']})" if base_data['total_ops'] > 0 else "N/A"
            no_moe_str = f"{no_moe_sim:>5.1f}% (n={no_moe_data['total_ops']})" if no_moe_data['total_ops'] > 0 else "N/A"
            moe_str = f"{moe_sim:>5.1f}% (n={moe_data['total_ops']})" if moe_data['total_ops'] > 0 else "N/A"
            
            print(f"{task:<15} | {temp:<7} | {base_str:<25} | {no_moe_str:<25} | {moe_str:<25}")
        print("-" * 110)

if __name__ == "__main__":
    analyze_diversity_comparison()