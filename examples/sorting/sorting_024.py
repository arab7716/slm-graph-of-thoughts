# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach
# copied from sorting_032.py and rewritten for sorting lists of length 24.
# Reconfigured for proactive failure mitigation by Artha Abeysinghe 
# Generative AI was used to assist partially with syntax. Every line of code was nonetheless thoroughly human reviewed.



import os
import logging
import datetime
import json
import csv
from typing import Dict, List, Callable, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser
import argparse

# This is a hack to also allow execution of this file from the examples directory
try:
    from . import utils
except ImportError:
    import utils

class SortingPrompter(prompter.Prompter):
    """
    prompt rewritten for 24 numbers
    SortingPrompter provides the generation of prompts specific to the sorting
    example for the language models.

    Inherits from the Prompter class and implements its abstract methods.    
    """

    sort_prompt = """<Instruction> Sort the following list of numbers in ascending order. Output only the sorted list of numbers, no additional text. </Instruction>
<Examples>
Input: [5, 1, 0, 1, 2, 0, 4, 8]
Output: [0, 0, 1, 1, 2, 4, 5, 8]
</Examples>
Input: {input}"""

    sort_prompt_cot = """<Instruction> Sort the following list of numbers in ascending order. You can generate any intermediate lists, but the final output should be the sorted list of numbers, prefixed with "Output: ". </Instruction>
<Approach>
To sort the list of numbers follow these steps:
1. Split the list of numbers into two sublists, each containing an equal number of elements from the original list (make sure they don't overlap).
2. Sort each of the sublists.
3. Merge the sorted sublists into a single sorted list using the merging algorithm from merge sort.
</Approach>
Input: {input}"""

    tot_improve_prompt = """<Instruction> The following two lists represent an unsorted list of numbers and a sorted variant of that list. The sorted variant is not correct. Fix the sorted variant so that it is correct.
Make sure that the output list is sorted in ascending order, has the same number of elements as the input list ({length}), and contains the same elements as the input list. </Instruction>
<Approach>
To fix the incorrectly sorted list follow these steps:
1. For each number from 0 to 9, compare the frequency of that number in the incorrectly sorted list to the frequency of that number in the input list.
2. Iterate through the incorrectly sorted list and add or remove numbers as needed to make the frequency of each number in the incorrectly sorted list match the frequency of that number in the input list.
</Approach>
Input: {input}
Incorrectly Sorted: {incorrectly_sorted}
"""

    #rewritten for 24 elements
    got_split_prompt = """<Instruction> Split the following list of 24 numbers into 2 lists of 12 numbers each, the first list should contain the first 12 numbers and the second list the second 12 numbers.
Only output the final 2 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, 2, 9, 0, 1, 4],
    "List 2": [2, 9, 2, 4, 7, 1, 5, 5, 8, 3, 2, 6]
}} </Instruction>
Input: {input}"""

    got_merge_prompt = """<Instruction> Merge the following 2 sorted lists of length {length1} each, into one sorted list of length {length2} using a merge sort style approach.
Only output the final merged list without any additional text or thoughts!:</Instruction>
<Approach>
To merge the two lists in a merge-sort style approach, follow these steps:
1. Compare the first element of both lists.
2. Append the smaller element to the merged list and move to the next element in the list from which the smaller element came.
3. Repeat steps 1 and 2 until one of the lists is empty.
4. Append the remaining elements of the non-empty list to the merged list.
</Approach>
Merge the following two lists into one sorted list:
1: {input1}
2: {input2}
Merged list:
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        :raise AssertionError: If not exactly two thought states are provided.
        """
        assert len(state_dicts) == 2, "Expected two states for aggregation prompt."
        len_input1 = len(utils.string_to_list(state_dicts[0]["current"]))
        len_input2 = len(utils.string_to_list(state_dicts[1]["current"]))
        
        # changed for 24 elements
        if len_input1 == len_input2:
            length = len_input1
        elif len_input1 + len_input2 - 24 <= 12:
            length = 12
        else:
            length = 24

        return self.got_merge_prompt.format(
            input1=state_dicts[0]["current"],
            input2=state_dicts[1]["current"],
            length1=length,
            length2=length * 2,
        )

    def generate_prompt(self, num_branches: int, original: str, current: str, method: str, **kwargs) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param original: Input list of numbers.
        :type original: str
        :param current: Intermediate solution.
        :type current: str
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If the requested number of branches is not one.
        """
        if current is None or current == "":
            input = original
        else:
            input = current
        if method.startswith("io"):
            return self.sort_prompt.format(input=input)
        elif method.startswith("cot"):
            return self.sort_prompt_cot.format(input=input)
        elif method.startswith("tot"):
            if current is None or current == "":
                return self.sort_prompt.format(input=input)
            return self.tot_improve_prompt.format(
                input=original,
                incorrectly_sorted=current,
                length=len(utils.string_to_list(original)),
            )
        elif method.startswith("got"):
            if current is None or current == "":
                return self.got_split_prompt.format(input=input)
            # if current is just a sublist of the original input, return the split prompt
            if kwargs["phase"] == 1:
                return self.sort_prompt.format(input=current)
            if (
                "unsorted_sublist" in kwargs
                and kwargs["unsorted_sublist"] != ""
                and len(kwargs["unsorted_sublist"]) < len(original) - 5
            ):
                original = kwargs["unsorted_sublist"]
            return self.tot_improve_prompt.format(
                input=original,
                incorrectly_sorted=current,
                length=len(utils.string_to_list(original)),
            )
    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        """
        pass
    def judge_prompt_generate(self, base_state: Dict, collapsed_output: str) -> str:
        raw_input = base_state.get("unsorted_sublist", base_state.get("original", "[]"))
        # Index mapping
        return (
            f"You are a strict data validation AI. Compare the Input against the Proposed Output.\n\n"
            f"Task: Sort the Input Data in ascending order. Do not drop or add any numbers.\n\n"
            f"Input Data:\n{raw_input}\n\n"
            f"Proposed Output:\n{collapsed_output}\n\n"
            f"To verify if the output is mathematically perfect, you MUST follow these steps:\n"
            f"Step 1: Count the elements in the Input array by numbering them (1, 2, 3...).\n"
            f"Step 2: Count the elements in the Proposed Output array by numbering them (1, 2, 3...).\n"
            f"Step 3: Compare the final counts. Do they match exactly?\n"
            f"Step 4: Check if the Proposed Output is perfectly sorted in ascending order.\n"
            f"Step 5: Check if the Proposed Output contains conversational text or extra brackets. It must be a single flat list.\n\n"
            f"Write out your counting process step-by-step. Conclude on a new line with exactly 'VERDICT: YES' if it is perfect, or 'VERDICT: NO' if it dropped numbers, is unsorted, or is formatted wrong."
        )

    def judge_prompt_aggregate(self, previous_thought_states: List[Dict], collapsed_output: str) -> str:
        list1 = previous_thought_states[0].get("current", "[]")
        list2 = previous_thought_states[1].get("current", "[]") if len(previous_thought_states) > 1 else "[]"
        return (
            f"You are a strict data validation AI. Compare the Input against the Proposed Output.\n\n"
            f"Task: Merge the two Input lists into a single sorted list containing all elements from both.\n\n"
            f"Input Data:\nList 1: {list1}\nList 2: {list2}\n\n"
            f"Proposed Output:\n{collapsed_output}\n\n"
            f"To verify if the output is mathematically perfect, you MUST follow these steps:\n"
            f"Step 1: Count the total combined elements in the Input arrays by numbering them (1, 2, 3...).\n"
            f"Step 2: Count the elements in the Proposed Output array by numbering them (1, 2, 3...).\n"
            f"Step 3: Compare the final counts. Do they match exactly?\n"
            f"Step 4: Check if the Proposed Output is perfectly sorted in ascending order.\n"
            f"Step 5: Check if the Proposed Output contains conversational text or extra brackets. It must be a single flat list.\n\n"
            f"Write out your counting process step-by-step. Conclude on a new line with exactly 'VERDICT: YES' if it is perfect, or 'VERDICT: NO' if it dropped numbers, is unsorted, or is formatted wrong."
        )
    def moe_lenses_generate(self) -> List[str]:
        return [
            "THE STEP-BY-STEP LOGICIAN: Do not jump to the final answer. Trace the movement of elements logically first.",
            "THE OMISSION CATCHER: You dropped numbers in your last attempt. Count the input elements explicitly, and ensure your output list has the exact same number of elements.",
            "THE CONSTRAINT VALIDATOR: Explicitly count the input elements. Ensure the output length matches exactly.",
        ]

    def moe_lenses_aggregate(self) -> List[str]:
        return [
            "THE LOGIC TRACER: Simulate a two-pointer merge step-by-step before outputting the final answer.",
            "THE MERGE AUDITOR: Explicitly verify that the final list length equals the sum of lengths of the input lists.",
            "THE DUPLICATE TRACKER: If a number exists in both input lists, it MUST exist twice in the final list.",
        ]
    

    def moe_critique_generate(self, collapsed_output: str, lens: str) -> str:
        return (
            f"\n\n[System Intervention]: Your previous output was evaluated as INCORRECT:\n"
            f"=== FAILED OUTPUT ===\n{collapsed_output}\n=====================\n\n"
            f"You must NOT generate this exact output again. To break your previous reasoning pattern, you MUST adopt this specific perspective:\n"
            f"-> {lens}\n\n"
            f"First, write 1-2 sentences applying this perspective to identify your previous mistake. "
            f"Then, you MUST output the corrected data strictly formatted within brackets (e.g.,[x, y, z])."
        )

    def moe_critique_aggregate(self, collapsed_output: str, lens: str) -> str:
        return (
            f"\n\n[System Intervention]: Your previous merge output was evaluated as INCORRECT:\n"
            f"=== FAILED OUTPUT ===\n{collapsed_output}\n=====================\n\n"
            f"You must NOT generate this exact output again. To fix this, adopt this perspective:\n"
            f"-> {lens}\n\n"
            f"First, write 1-2 sentences applying this perspective. "
            f"Then, you MUST output the corrected merged data strictly formatted within brackets (e.g.,[x, y, z])."
        )

class SortingParser(parser.Parser):
    """
    SortingParser provides the parsing of language model reponses specific to
    the sorting example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        :raise AssertionError: If not exactly two thought states are provided.
        """
        assert len(states) == 2, "Expected two states for aggregation answer."
        new_states = []
        for text in texts:
            answers = text.strip().split("\n")
            if any(["Output" in answer for answer in answers]):
                # cut elements until last output is found
                for answer in reversed(answers):
                    if "Output" in answer:
                        answers = answers[answers.index(answer) :]
                        break
            answers_stripped = [answer for answer in answers if "[" in answer and "]" in answer]
            if len(answers_stripped) == 0:
                for answer in answers:
                    answer = "[" + answer + "]"
                    try:
                        answer_converted = utils.string_to_list(answer)
                        if len(answer_converted) > 0: answers_stripped.append(answer)
                    except: pass
            if len(answers_stripped) == 0:
                logging.warning(f"Could not parse aggregation answer: {text}. Returning empty list.")
                answer = "[]"
            else:
                answer = [answer[answer.index("[") : answer.index("]") + 1] for answer in answers_stripped][0]
            states = sorted(states, key=lambda x: x["part"])
            try:
                merged_unsorted_sublists = (states[0]["unsorted_sublist"][:-1] + ", " + states[1]["unsorted_sublist"][1:])
            except: merged_unsorted_sublists = states[0].get("original", "[]")
            new_state = states[0].copy()
            new_state["current"] = answer
            new_state["unsorted_sublist"] = merged_unsorted_sublists
            new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            if state["method"].startswith("got") and state["current"] == "":
                # We expect a json which contains the four lists named "List 1" to "List 4"
                # cut everything until the opening bracket and everything after the closing bracket
                try:
                    # changed logic to regex search for the formatted output
                    import re
                    match = re.search(r'\{.*?"List 1".*?\}', text, re.S)
                    if match: text = match.group(0)
                    else:
                        start = text.find("{")
                        end = text.rfind("}")
                        if start != -1 and end != -1: text = text[start : end + 1]
                    json_dict = json.loads(text)
                    if len(json_dict.keys()) != 2:
                        logging.warning(f"Expected 2 lists in json, but found {len(json_dict.keys())}.")
                    for key, value in json_dict.items():
                        if "List" not in key: continue
                        if not isinstance(value, list): value = utils.string_to_list(value)
                        new_state = state.copy()
                        new_state["current"] = str(value)
                        new_state["unsorted_sublist"] = str(value)
                        new_state["phase"] = 1
                        new_state["part"] = key
                        new_states.append(new_state)
                except Exception as e:
                    logging.error(f"Could not parse step answer: {text}. Encountered exception: {e}")
            else:
                answers = text.strip().split("\n")
                answers = [answer for answer in answers if "[" in answer and "]" in answer]
                if any(["Output" in answer for answer in answers]):
                    for answer in reversed(answers):
                        if "Output" in answer:
                            answers = answers[answers.index(answer) :]
                            break
                final_answer = "[]"
                if len(answers) > 0:
                     final_answer = answers[0][answers[0].index("[") : answers[0].rindex("]") + 1]
                new_state = state.copy()
                new_state["current"] = final_answer
                new_state["phase"] = 2
                new_states.append(new_state)
        return new_states
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.
    ToT uses a wider tree, where on each level there are more branches.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 20))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(1):
        operations_graph.append_operation(operations.Generate(1, 20))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT2 method.
    ToT2 uses a tree with more levels, but with fewer branches per level.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, 10))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph

def check_sorting_validity(output_str, state_data):
    """Python validator function for list"""
    try:
        import ast, re
        match = re.search(r'\[[\d,\s]+\]', output_str)
        if not match: return False
        out_list = ast.literal_eval(match.group())
        
        # aggregate
        if isinstance(state_data, list):
            list1 = ast.literal_eval(state_data[0].get("current", "[]"))
            list2 = ast.literal_eval(state_data[1].get("current", "[]")) if len(state_data) > 1 else []
            true_sorted = sorted(list1 + list2)
            return out_list == true_sorted
            
        #generate
        base_state = state_data
        original_input = base_state.get("unsorted_sublist", base_state.get("original", "[]"))
        in_match = re.search(r'\[[\d,\s]+\]', original_input)
        if in_match:
            in_list = ast.literal_eval(in_match.group())
            return out_list == sorted(in_list)
        return True
    except:
        return False

def got_original() -> operations.GraphOfOperations:
    # Uses standard Generate and Aggregate (No intervention code at all)
    g = operations.GraphOfOperations()
    plans = operations.Generate(1, 1)
    g.append_operation(plans)
    endpoints =[]
    for i in range(1, 3):
        sub_list = operations.Selector(lambda t, lid=f"List {i}": [x for x in t if x.state["part"] == lid])
        sub_list.add_predecessor(plans)
        g.add_operation(sub_list)
        
        sort = operations.Generate(1, 5)
        sort.add_predecessor(sub_list)
        g.add_operation(sort)
        
        score = operations.Score(1, False, utils.num_errors)
        score.add_predecessor(sort)
        g.add_operation(score)
        
        keep = operations.KeepBestN(1, False)
        keep.add_predecessor(score)
        g.add_operation(keep)
        endpoints.append(keep)

    agg = operations.Aggregate(10)
    for ep in endpoints: agg.add_predecessor(ep)
    g.add_operation(agg)
    g.append_operation(operations.Score(1, False, utils.num_errors))
    g.append_operation(operations.KeepBestN(1, False))
    g.append_operation(operations.Generate(1, 10))
    g.append_operation(operations.Score(1, False, utils.num_errors))
    g.append_operation(operations.KeepBestN(1, False))
    g.append_operation(operations.GroundTruth(utils.test_sorting))
    return g

# helper function for various ablations
def create_proactive_got(intervention_enabled=True, use_llm_judge=True, use_moe=True, validator_fn=None):
    g = operations.GraphOfOperations()
    plans = operations.Generate(1, 1)
    g.append_operation(plans)
    endpoints =[]
    for i in range(1, 3):
        sub_list = operations.Selector(lambda t, lid=f"List {i}":[x for x in t if x.state["part"] == lid])
        sub_list.add_predecessor(plans)
        g.add_operation(sub_list)
        
        sort = operations.ProactiveGenerate(1, 5, 0.90, intervention_enabled, use_llm_judge, use_moe, validator_fn)
        sort.add_predecessor(sub_list)
        g.add_operation(sort)
        
        score = operations.Score(1, False, utils.num_errors)
        score.add_predecessor(sort)
        g.add_operation(score)
        
        keep = operations.KeepBestN(1, False)
        keep.add_predecessor(score)
        g.add_operation(keep)
        endpoints.append(keep)

    agg = operations.ProactiveAggregate(10, 0.90, intervention_enabled, use_llm_judge, use_moe, validator_fn)
    for ep in endpoints: agg.add_predecessor(ep)
    g.add_operation(agg)
    g.append_operation(operations.Score(1, False, utils.num_errors))
    g.append_operation(operations.KeepBestN(1, False))
    
    refine = operations.ProactiveGenerate(1, 10, 0.90, intervention_enabled, use_llm_judge, use_moe, validator_fn)
    g.append_operation(refine)
    g.append_operation(operations.Score(1, False, utils.num_errors))
    g.append_operation(operations.KeepBestN(1, False))
    g.append_operation(operations.GroundTruth(utils.test_sorting))
    return g

def got_2_nodes(): 
    return create_proactive_got(intervention_enabled=False)

def got_python_moe(): 
    return create_proactive_got(use_llm_judge=False, use_moe=True, validator_fn=check_sorting_validity)

def got_python_no_moe(): 
    return create_proactive_got(use_llm_judge=False, use_moe=False, validator_fn=check_sorting_validity)

def got_llm_no_moe(): 
    return create_proactive_got(use_llm_judge=True, use_moe=False)

def got_full(): 
    return create_proactive_got(use_llm_judge=True, use_moe=True)



def run(data_ids, methods, budget, lm_name, config_path, temperature=None, job_id_suffix=""):
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "sorting_024.csv")
    data = []
    with open(data_path, "r") as f:
    
        reader = csv.reader(f); next(reader)
        for row in reader: data.append([int(row[0]), row[1], row[2]])
    if data_ids is None: data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_to_use = temperature if temperature is not None else 0.6
    temp_str = f"_T{str(temp_to_use).replace('.', 'p')}"
    extra_info = f"{lm_name}{temp_str}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump({"methods": [m.__name__ for m in methods], "lm": lm_name}, f)
    logging.basicConfig(filename=os.path.join(results_folder, "log.log"), filemode="w", level=logging.DEBUG)
    for method in methods: os.makedirs(os.path.join(results_folder, method.__name__))
    logging.info("Loading model(once).")
    lm = language_models.Llama2HF(config_path, model_name=lm_name, temperature=temperature)
    for data in selected_data:
        logging.info(f"Running data {data[0]}")
        for method in methods:
            ops = method()
            ctrl = controller.Controller(lm, ops, SortingPrompter(), SortingParser(), {"original": data[1], "current": "", "phase": 0, "method": method.__name__})
            
            # timing
            start_time = datetime.datetime.now().timestamp()
            try: 
                ctrl.run()
            except Exception as e: 
                logging.error(f"Error: {e}")
            end_time = datetime.datetime.now().timestamp()
            execution_time = end_time - start_time
            
        
            output_path = os.path.join(results_folder, method.__name__, f"{data[0]}.json")
            ctrl.output_graph(output_path)
            
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    graph_data = json.load(f)
                
                # custom metrics added to json
                if isinstance(graph_data, dict):
                    graph_data["execution_time_seconds"] = execution_time
                elif isinstance(graph_data, list):
                    graph_data.append({"execution_time_seconds": execution_time})
                
                with open(output_path, "w") as f:
                    json.dump(graph_data, f, indent=4)
                    
    return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--model_name", type=str, default="qwen2.5-14b", help="Model name key from config.json")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature from config")
    args = parser.parse_args()

    budget = 100
    samples = list(range(100))
    
    
    approaches =[
        io,
        cot,
        got_original, 
        got_2_nodes, 
        got_python_moe, 
        got_python_no_moe,
        got_llm_no_moe, 
        got_full
    ]

    spent = run(samples, approaches, budget, args.model_name, args.config_path, args.temperature)

    logging.info(f"Spent {spent} out of {budget} budget.")