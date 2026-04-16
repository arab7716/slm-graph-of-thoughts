# Evaluation script to analyze all ablation runs for accuracy, execution time, nodes, bad nodes, etc.
# Code syntax generation partially aided by generative AI. 
# All code thoroughly human reviewed.

import os
import json
import csv
import re
import ast
import argparse
from collections import Counter

ANALYSIS_ROOT = "analysis_output_master"

# saved here are the results from the runs outlined in the paper. Replace with new file paths for new runs as necessary!
TARGET_FOLDERS =[
 "examples/sorting/results/qwen2.5-14b_T0p1_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-45-14",
 "examples/sorting/results/qwen2.5-14b_T0p6_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-45-14",
 "examples/keyword_counting/results/qwen2.5-14b_T0p1_io-cot-got4_original-got4_2_nodes-got4_python_moe-got4_python_no_moe-got4_llm_no_moe-got4_full_2026-04-13_22-45-14_528054",
 "examples/keyword_counting/results/qwen2.5-14b_T0p6_io-cot-got4_original-got4_2_nodes-got4_python_moe-got4_python_no_moe-got4_llm_no_moe-got4_full_2026-04-13_22-45-14_528056",
 "examples/set_intersection/results/qwen2.5-14b_T0p1_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-46-10_478224",
 "examples/set_intersection/results/qwen2.5-14b_T0p6_io-cot-got_original-got_2_nodes-got_python_moe-got_python_no_moe-got_llm_no_moe-got_full_2026-04-13_22-46-10_478222",
]


def detect_task_size(text_input):
    try:
        if not text_input or text_input == "N/A": return 32
        lst = ast.literal_eval(text_input)
        return len(lst)
    except: return 32

def robust_parse_list(text, expected_length=None):
    matches = re.findall(r'\[[\d,\s]+\]', text)
    for match in matches:
        try:
            candidate = ast.literal_eval(match)
            if isinstance(candidate, list):
                if expected_length is None or len(candidate) == expected_length:
                    return [int(x) for x in candidate]
        except: continue
    return None

def calculate_set_error(gt_list, output_list):
    if gt_list is None or output_list is None: return 999, False
    gt_set, out_set = set(gt_list), set(output_list)
    total_error = len(gt_set - out_set) + len(out_set - gt_set)
    return total_error, (total_error == 0)

def get_ground_truth_dict(gt_str):
    try:
        if not gt_str: return {}
        try:
            return dict(Counter(ast.literal_eval(gt_str)))
        except:
            content = gt_str.strip("[]")
            if not content: return {}
            return dict(Counter([c.strip() for c in content.split(",")]))
    except: return {}

def robust_extract_dict(text):
    matches = re.findall(r'\{.*?\}', text, re.DOTALL)
    for match in matches:
        try: return json.loads(match)
        except:
            try: return ast.literal_eval(match)
            except: continue
    return None

def compare_dicts(dict1, dict2):
    if dict1 is None or dict2 is None: return False
    d1 = {k.strip().lower(): v for k, v in dict1.items() if v > 0}
    d2 = {k.strip().lower(): v for k, v in dict2.items() if v > 0}
    return d1 == d2

def calculate_keyword_error(gt_dict, out_dict):
    if gt_dict is None: gt_dict = {}
    if out_dict is None: out_dict = {}
    gt_norm = {k.lower().strip(): v for k, v in gt_dict.items()}
    out_norm = {k.lower().strip(): v for k, v in out_dict.items()}
    all_keys = set(gt_norm.keys()) | set(out_norm.keys())
    return sum(abs(gt_norm.get(k, 0) - out_norm.get(k, 0)) for k in all_keys)

def analyze_folder(folder_path, output_root):
    if not os.path.exists(folder_path):
        print(f"Skipping (Not found): {folder_path}")
        return

    # detect task type
    task_type = "UNKNOWN"
    if "sorting" in folder_path.lower(): task_type = "SORTING"
    elif "keyword" in folder_path.lower(): task_type = "KEYWORDS"
    elif "set_intersection" in folder_path.lower(): task_type = "SETS"

    run_name = os.path.basename(os.path.normpath(folder_path))
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)
        
    print(f"--> Analyzing: {run_name} [{task_type}]")
    
    metrics_data = []
    summary_lines =[f"Analysis Summary for Run: {run_name}[{task_type}]\n" + "="*80]

    methods_found =[d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    order = {
        "io": 0, "cot": 1,
        "got_original": 2, "got4_original": 2,
        "got_2_nodes": 3,  "got4_2_nodes": 3,
        "got_python_moe": 4, "got4_python_moe": 4,
        "got_python_no_moe": 5, "got4_python_no_moe": 5,
        "got_llm_no_moe": 6, "got4_llm_no_moe": 6,
        "got_full": 7, "got4_full": 7
    }
    methods_found.sort(key=lambda x: order.get(x, 99))

    for method in methods_found:
        method_path = os.path.join(folder_path, method)
        
        
        total_samples = 0
        cnt_strict, cnt_recovered, cnt_failed, cnt_broken = 0, 0, 0, 0
        failure_scores, execution_times, total_nodes_list, bad_nodes_list = [], [], [], []
        interventions_list =[] 

        files = sorted([f for f in os.listdir(method_path) if f.endswith(".json")])
        
        for f_name in files:
            total_samples += 1
            try:
                with open(os.path.join(method_path, f_name), 'r') as f:
                    data = json.load(f)
                if not data or len(data) < 2:
                    cnt_broken += 1; continue

                
                sample_time = 0.0
                num_nodes = 0
                num_bad_nodes = 0
                num_interventions = 0

                for item in data:
                    if isinstance(item, dict):
                        #time
                        if "execution_time_seconds" in item:
                            sample_time = float(item["execution_time_seconds"])
                        
                        # total nodes
                        if item.get("operation") in ["generate", "aggregate"]:
                            thoughts = item.get("thoughts",[])
                            num_nodes += len(item.get("thoughts",[]))
                            for t in thoughts:
                                if t.get("intervened") is True:
                                    num_interventions += 1
                            
                        # bad nodes (where score > 0)
                        if item.get("operation") == "score" and "scores" in item:
                            for s in item["scores"]:
                                if isinstance(s, (int, float)) and s > 0:
                                    num_bad_nodes += 1
                
                execution_times.append(sample_time)
                total_nodes_list.append(num_nodes)
                bad_nodes_list.append(num_bad_nodes)
                interventions_list.append(num_interventions)

                # final eval
                ground_truth_op = None
                last_thought = None
                
                for item in reversed(data):
                    if isinstance(item, dict):
                        if item.get("operation") == "ground_truth_evaluator":
                            ground_truth_op = item
                        if item.get("thoughts") and len(item["thoughts"]) > 0 and last_thought is None:
                            last_thought = item["thoughts"][0]

                if not ground_truth_op:
                    cnt_broken += 1; continue

                status = "STRICT_SUCCESS" if ground_truth_op.get("problem_solved", [False])[0] else "STRICT_FAILURE"

                # task specific data
                sample_data = {"input": "N/A", "output": "N/A", "gt": "N/A", "score": 0}
                if last_thought:
                    sample_data["output"] = last_thought.get("current", "[]")
                    sample_data["input"] = last_thought.get("original", last_thought.get("set1", "") + last_thought.get("set2", ""))
                    sample_data["gt"] = last_thought.get("ground_truth", last_thought.get("result", "[]"))
                
                if ground_truth_op.get("scores") and len(ground_truth_op["scores"]) > 0:
                    try: sample_data["score"] = float(ground_truth_op["scores"][0])
                    except: sample_data["score"] = 0

                # eval successes
                if status == "STRICT_SUCCESS":
                    cnt_strict += 1
                else:
                    is_recovered = False
                    error_val = sample_data["score"] 

                    # logic for task specific success
                    if task_type == "SORTING":
                        t_size = detect_task_size(sample_data["input"])
                        found_list = robust_parse_list(sample_data["output"], t_size)
                        try:
                            if found_list == sorted(ast.literal_eval(sample_data["input"])): is_recovered = True
                        except: pass

                    elif task_type == "SETS":
                        gt_list, out_list = robust_parse_list(sample_data["gt"]), robust_parse_list(sample_data["output"])
                        err, is_perfect = calculate_set_error(gt_list, out_list)
                        if is_perfect: is_recovered = True
                        else: error_val = err

                    elif task_type == "KEYWORDS":
                        gt_dict, out_dict = get_ground_truth_dict(sample_data["gt"]), robust_extract_dict(sample_data["output"])
                        if compare_dicts(gt_dict, out_dict): is_recovered = True
                        else: error_val = calculate_keyword_error(gt_dict, out_dict)

                    if is_recovered: cnt_recovered += 1
                    else: 
                        cnt_failed += 1
                        failure_scores.append(error_val)

            except Exception as e:
                cnt_broken += 1

        # final averages
        strict_rate = (cnt_strict / total_samples * 100) if total_samples > 0 else 0
        true_rate = ((cnt_strict + cnt_recovered) / total_samples * 100) if total_samples > 0 else 0
        avg_err = (sum(failure_scores) / len(failure_scores)) if failure_scores else 0
        
        avg_time = (sum(execution_times) / len(execution_times)) if execution_times else 0.0
        avg_nodes = (sum(total_nodes_list) / len(total_nodes_list)) if total_nodes_list else 0.0
        avg_bad_nodes = (sum(bad_nodes_list) / len(bad_nodes_list)) if bad_nodes_list else 0.0
        avg_interventions = (sum(interventions_list) / len(interventions_list)) if interventions_list else 0.0 # <--- ADD THIS

        # print
        print(f"  {method.upper():<5} | Strict: {strict_rate:5.1f}% | True: {true_rate:5.1f}% | AvgErr: {avg_err:5.2f} | Time: {avg_time:5.1f}s | Nodes/Spl: {avg_nodes:4.1f} | BadNodes: {avg_bad_nodes:4.1f} | Intv: {avg_interventions:4.1f} | Broken: {cnt_broken}")

        # update csv
        metrics_data.append([method, total_samples, f"{strict_rate:.1f}", f"{true_rate:.1f}", f"{avg_err:.2f}", f"{avg_time:.1f}", f"{avg_nodes:.1f}", f"{avg_bad_nodes:.1f}", f"{avg_interventions:.1f}"])


    # save csv
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Total", "Strict%", "True%", "AvgErr", "AvgTime(s)", "AvgNodes", "AvgBadNodes", "AvgInterventions"])
        writer.writerows(metrics_data)
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="Specific folder to analyze")
    args = parser.parse_args()

    if args.folder:
        analyze_folder(args.folder, ANALYSIS_ROOT)
    elif len(TARGET_FOLDERS) > 0:
        print(f"Processing {len(TARGET_FOLDERS)} folders from TARGET_FOLDERS list...\n")
        for folder in TARGET_FOLDERS:
            analyze_folder(folder, ANALYSIS_ROOT)
    else:
        print("Please specify TARGET_FOLDERS in the script or pass --folder in the command line.")