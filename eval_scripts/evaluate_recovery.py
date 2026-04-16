# Evaluation script to calculate recovery rates of our intervention (generic retry vs MoE lenses)
# Code syntax generation partially aided by generative AI. 
# All code thoroughly human reviewed.
import os
import json
import re
import ast
import argparse
from collections import Counter

TARGET_FOLDERS =[
    #insert folders here
   "examples/sorting/results/qwen2.5-14b_T0p1_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-45-14",
    "examples/sorting/results/qwen2.5-14b_T0p6_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-45-14",
    "examples/keyword_counting/results/qwen2.5-14b_T0p1_io-cot-got4_original-got4_2_nodes-got4_python_moe-got4_python_no_moe-got4_llm_no_moe-got4_full_2026-04-13_22-45-14_528054",
    "examples/keyword_counting/results/qwen2.5-14b_T0p6_io-cot-got4_original-got4_2_nodes-got4_python_moe-got4_python_no_moe-got4_llm_no_moe-got4_full_2026-04-13_22-45-14_528056",
    "examples/set_intersection/results/qwen2.5-14b_T0p1_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-46-10_478224",
    "examples/set_intersection/results/qwen2.5-14b_T0p6_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-46-10_478222",
]


def robust_parse_list(text):
    if not text: return None
    matches = re.findall(r'\[[\d,\s]+\]', str(text))
    for match in matches:
        try:
            c = ast.literal_eval(match)
            if isinstance(c, list): return [int(x) for x in c]
        except: continue
    return None

def is_sorting_correct(thought, is_aggregate):
    out_list = robust_parse_list(thought.get("current", "[]"))
    if out_list is None: return False
    
    if is_aggregate:
        in_list = robust_parse_list(thought.get("original", "[]"))
    else:
        in_list = robust_parse_list(thought.get("unsorted_sublist", thought.get("original", "[]")))
        
    if in_list is None: return False
    return out_list == sorted(in_list)

def is_sets_correct(thought, is_aggregate):
    out_list = robust_parse_list(thought.get("current", "[]"))
    if out_list is None: return False
    
    if is_aggregate:
        gt_list = robust_parse_list(thought.get("result", thought.get("ground_truth", "[]")))
        if gt_list is None: return False
        return set(out_list) == set(gt_list)
    else:
        set1 = robust_parse_list(thought.get("set1", "[]"))
        set2 = robust_parse_list(thought.get("subset", thought.get("set2", "[]")))
        if set1 is None or set2 is None: return False
        return set(out_list) == set(set1).intersection(set(set2))
def get_ground_truth_dict(gt_str):
    try:
        if not gt_str: return {}
        try: return dict(Counter(ast.literal_eval(gt_str)))
        except:
            content = gt_str.strip("[]")
            if not content: return {}
            return dict(Counter([c.strip() for c in content.split(",")]))
    except: return {}

def is_keywords_correct(thought, is_aggregate):
    out_str = thought.get("current", "{}")
    match = re.search(r'\{.*?\}', str(out_str), re.DOTALL)
    if not match: return False
    
    try: out_dict = json.loads(match.group())
    except: return False
        
    if not is_aggregate:
        input_text = thought.get("sub_text", thought.get("original", "")).strip()
        if not input_text: return len(out_dict) == 0
        
        if len(out_dict) == 0: return False
        for country in out_dict.keys():
            if country.lower() not in input_text.lower(): return False
        return True
    else:
        # aggregate
        try:
            dict1 = json.loads(thought.get("aggr1", "{}"))
            dict2 = json.loads(thought.get("aggr2", "{}"))
            true_combined = {**dict1}
            for k, v in dict2.items():
                true_combined[k] = true_combined.get(k, 0) + v
                
            true_norm = {k.lower().strip(): v for k, v in true_combined.items() if v > 0}
            out_norm = {k.lower().strip(): int(v) for k, v in out_dict.items() if int(v) > 0}
            return true_norm == out_norm
        except:
            return False


def evaluate_recovery_performance():
    if not TARGET_FOLDERS:
        print("Please add folder paths to TARGET_FOLDERS.")
        return

    for folder_path in TARGET_FOLDERS:
        base_folder = folder_path.replace("/got", "") if folder_path.endswith("/got") else folder_path
        
        if not os.path.exists(base_folder):
            print(f"\nSkipping (Not found): {base_folder}")
            continue

        task_type = "UNKNOWN"
        if "sorting" in base_folder.lower(): task_type = "SORTING"
        elif "keyword" in base_folder.lower(): task_type = "KEYWORDS"
        elif "set_intersection" in base_folder.lower(): task_type = "SETS"

        run_name = os.path.basename(os.path.normpath(base_folder))
        print(f"\n\n" + "="*80)
        print(f" EVALUATING RECOVERY RATE: {run_name} [{task_type}]")
        print("="*80)
        print(f"{'Method':<20} | {'Interventions':<15} | {'Recovered':<10} | {'Recovery Rate %':<15}")
        print("-" * 80)

        methods_found =[d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        #only look at methods where interventions occurred
        target_methods =[m for m in methods_found if "moe" in m.lower() or "full" in m.lower()]
        target_methods.sort()

        if not target_methods:
            print("No intervention methods found in this folder.")
            continue

        for method in target_methods:
            method_path = os.path.join(base_folder, method)
            
            total_interventions = 0
            successful_recoveries = 0
            
            files = sorted([f for f in os.listdir(method_path) if f.endswith(".json")])
            
            for f_name in files:
                try:
                    with open(os.path.join(method_path, f_name), 'r') as f: 
                        data = json.load(f)
                    
                    for item in data:
                        if isinstance(item, dict) and item.get("operation") in ["generate", "aggregate"]:
                            thoughts = item.get("thoughts",[])
                            
                            
                            intervened = any(t.get("intervened") for t in thoughts)
                            
                            if intervened:
                                total_interventions += 1
                                
                                # look after the first two thoughts (initial probe) to see if correct answer was found
                                intervention_thoughts = thoughts[2:] if len(thoughts) > 2 else[]
                                
                                recovered = False
                                for t in intervention_thoughts:
                                    is_aggregate = item.get("operation") == "aggregate"
                                    
                                    if task_type == "SORTING":
                                        is_correct = is_sorting_correct(t, is_aggregate)
                                    elif task_type == "SETS":
                                        is_correct = is_sets_correct(t, is_aggregate)
                                    elif task_type == "KEYWORDS":
                                        is_correct = is_keywords_correct(t, is_aggregate)
                                    else:
                                        is_correct = False
                                        
                                    if is_correct:
                                        recovered = True
                                        break 
                                        
                                if recovered:
                                    successful_recoveries += 1

                except Exception as e:
                    continue
            
            #print
            if total_interventions == 0:
                print(f"{method:<20} | {'0':<15} | {'0':<10} | {'N/A':<15}")
            else:
                recovery_rate = (successful_recoveries / total_interventions) * 100
                print(f"{method:<20} | {total_interventions:<15} | {successful_recoveries:<10} | {recovery_rate:<5.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="Specific folder to analyze")
    args = parser.parse_args()
    
    if args.folder:
        TARGET_FOLDERS = [args.folder]
        
    evaluate_recovery_performance()