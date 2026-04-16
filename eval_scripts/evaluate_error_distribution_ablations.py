import os
import json
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

def analyze_error_distribution_ablations():
    if not TARGET_FOLDERS:
        print("Please add folder paths to TARGET_FOLDERS.")
        return

    # Added [temp] to the tracking dictionary
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"total_steps": 0, "failed_steps": 0}))))

    for folder_path in TARGET_FOLDERS:
        if not os.path.exists(folder_path): continue

        task_type = "UNKNOWN"
        if "sorting" in folder_path.lower(): task_type = "SORTING"
        elif "keyword" in folder_path.lower(): task_type = "KEYWORDS"
        elif "set_intersection" in folder_path.lower(): task_type = "SETS"

        temp = "T=0.1" if "T0p1" in folder_path else "T=0.6" if "T0p6" in folder_path else "UNKNOWN"

        methods_found = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        target_methods = [m for m in methods_found if "original" in m.lower() or "python" in m.lower()]

        for method in target_methods:
            if "original" in method.lower(): m_key = "BASELINE"
            elif "python_moe" in method.lower() and "no_moe" not in method.lower(): m_key = "MOE"
            elif "python_no_moe" in method.lower(): m_key = "GENERIC_RETRY"
            else: continue

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
                        
                        if op in ["generate", "aggregate"]:
                            last_op_type = op
                            
                        elif op == "score" and last_op_type in ["generate", "aggregate"]:
                            scores = item.get("scores", [])
                            if not scores: continue
                            
                            stats[task_type][temp][last_op_type][m_key]["total_steps"] += 1
                            
                            valid_scores = [s for s in scores if isinstance(s, (int, float))]
                            if valid_scores:
                                best_score = min(valid_scores)
                                if best_score > 0:
                                    stats[task_type][temp][last_op_type][m_key]["failed_steps"] += 1
                            
                            last_op_type = None
                                
                except Exception as e:
                    continue

    # --- PRINT THE RESULTS ---
    print("\n" + "="*115)
    print(f" BRANCH-LEVEL ERROR DISTRIBUTION BY TEMP (Baseline vs. Interventions)")
    print("="*115)
    print(f"{'Task':<15} | {'Temp':<7} | {'Operation':<12} | {'Baseline Fail %':<20} | {'Generic Retry Fail %':<22} | {'MoE Fail %':<15}")
    print("-" * 115)

    for task in ["SORTING", "SETS", "KEYWORDS"]:
        if task not in stats: continue
        for temp in ["T=0.1", "T=0.6"]:
            if temp not in stats[task]: continue
            for op in ["generate", "aggregate"]:
                if op not in stats[task][temp]: continue
                
                base_data = stats[task][temp][op].get("BASELINE", {"total_steps": 0, "failed_steps": 0})
                gen_data = stats[task][temp][op].get("GENERIC_RETRY", {"total_steps": 0, "failed_steps": 0})
                moe_data = stats[task][temp][op].get("MOE", {"total_steps": 0, "failed_steps": 0})
                
                base_rate = (base_data["failed_steps"] / base_data["total_steps"] * 100) if base_data["total_steps"] > 0 else 0.0
                gen_rate = (gen_data["failed_steps"] / gen_data["total_steps"] * 100) if gen_data["total_steps"] > 0 else 0.0
                moe_rate = (moe_data["failed_steps"] / moe_data["total_steps"] * 100) if moe_data["total_steps"] > 0 else 0.0
                
                base_str = f"{base_rate:>5.1f}% (n={base_data['total_steps']})" if base_data["total_steps"] > 0 else "N/A"
                gen_str = f"{gen_rate:>5.1f}% (n={gen_data['total_steps']})" if gen_data["total_steps"] > 0 else "N/A"
                moe_str = f"{moe_rate:>5.1f}% (n={moe_data['total_steps']})" if moe_data["total_steps"] > 0 else "N/A"

                print(f"{task:<15} | {temp:<7} | {op.upper():<12} | {base_str:<20} | {gen_str:<22} | {moe_str:<15}")
            print("-" * 115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="Specific folder to analyze")
    args = parser.parse_args()
    if args.folder: TARGET_FOLDERS = [args.folder]
    analyze_error_distribution_ablations()