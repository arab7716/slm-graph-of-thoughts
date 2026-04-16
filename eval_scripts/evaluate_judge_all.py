# Evaluation script to calculate accuracy of LLM Judge
# Code syntax generation partially aided by generative AI. 
# All code thoroughly human reviewed.
import os
import json
import re
import ast
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


#python validators
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
    in_list = robust_parse_list(thought.get("original", "[]")) if is_aggregate else robust_parse_list(thought.get("unsorted_sublist", thought.get("original", "[]")))
    if in_list is None: return False
    return out_list == sorted(in_list)

def is_sets_correct(thought, is_aggregate):
    out_list = robust_parse_list(thought.get("current", "[]"))
    if out_list is None: return False
    if is_aggregate:
        gt_list = robust_parse_list(thought.get("result", "[]"))
        if gt_list is not None: return set(out_list) == set(gt_list)
        return True 
    else:
        set1 = robust_parse_list(thought.get("set1", "[]"))
        set2 = robust_parse_list(thought.get("subset", "[]"))
        if set1 is None or set2 is None: return False
        return set(out_list) == set(set1).intersection(set(set2))

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
        try:
            dict1 = json.loads(thought.get("aggr1", "{}"))
            dict2 = json.loads(thought.get("aggr2", "{}"))
            true_combined = {**dict1}
            for k, v in dict2.items():
                true_combined[k] = true_combined.get(k, 0) + v
            true_norm = {k.lower().strip(): v for k, v in true_combined.items() if v > 0}
            out_norm = {k.lower().strip(): int(v) for k, v in out_dict.items() if int(v) > 0}
            return true_norm == out_norm
        except: return False


def evaluate_judge_performance():
    for folder_path in TARGET_FOLDERS:
        if not os.path.exists(folder_path): continue

        task_type = "UNKNOWN"
        if "sorting" in folder_path.lower(): task_type = "SORTING"
        elif "keyword" in folder_path.lower(): task_type = "KEYWORDS"
        elif "set_intersection" in folder_path.lower(): task_type = "SETS"

        run_name = os.path.basename(os.path.normpath(folder_path))
        print(f"\n" + "="*95)
        print(f"EVALUATING: {run_name} [{task_type}]")
        print(f"{'Ablation':<18} | {'Judges':<6} | {'Acc':<6} | {'FPR':<5} | {'TP (YES)':<8} | {'TN (NO)':<7} | {'FP (Gullible)':<13} | {'FN (Punish)':<12}")
        print("-" * 95)

        methods_found = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        # only look at runs where llm judge was enabled
        target_methods = [m for m in sorted(methods_found) if m in ["got_llm_no_moe", "got_full", "got4_llm_no_moe", "got4_full"]]

        for method in target_methods:
            method_path = os.path.join(folder_path, method)
            tp, tn, fp, fn, total = 0, 0, 0, 0, 0
            
            for f_name in os.listdir(method_path):
                if not f_name.endswith(".json"): continue
                try:
                    with open(os.path.join(method_path, f_name), 'r') as f: data = json.load(f)
                    
                    for item in data:
                        if isinstance(item, dict) and item.get("operation") in ["generate", "aggregate"]:
                            thoughts = item.get("thoughts", [])
                            if not thoughts: continue
                            
                            is_aggregate = item.get("operation") == "aggregate"
                            
                    
                            # If exactly 2 thoughts exist, early stopping triggered --> judge verdict is yes
                            if len(thoughts) == 2:
                                judge_says_correct = True
                            # If >2 thoughts exist, and ANY have intervened: True, --> judge verdict is no
                            elif any(t.get("intervened") for t in thoughts):
                                judge_says_correct = False
                            # If >2 thoughts and NO intervention, Similarity was < 0.90 --> no judge invoked
                            else:
                                continue 
                                
                            total += 1
                            probe_thought = thoughts[0] 
                            
                            
                            is_math_correct = False
                            if task_type == "SORTING": is_math_correct = is_sorting_correct(probe_thought, is_aggregate)
                            elif task_type == "SETS": is_math_correct = is_sets_correct(probe_thought, is_aggregate)
                            elif task_type == "KEYWORDS": is_math_correct = is_keywords_correct(probe_thought, is_aggregate)
                                
                            if is_math_correct and judge_says_correct: tp += 1
                            elif not is_math_correct and not judge_says_correct: tn += 1
                            elif not is_math_correct and judge_says_correct: fp += 1
                            elif is_math_correct and not judge_says_correct: fn += 1
                                
                except: continue
            
            if total == 0: continue
            acc = (tp + tn) / total * 100
            fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
            print(f"{method:<18} | {total:<6} | {acc:<5.1f}% | {fpr:<4.1f}% | {tp:<8} | {tn:<7} | {fp:<13} | {fn:<12}")

if __name__ == "__main__":
    evaluate_judge_performance()