# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# The source code is adapted from the sorting source code written by
# Nils Blach.
#
# main author: Robert Gerstenberger 
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


class SetIntersectionPrompter(prompter.Prompter):
    """
    SetIntersectionPrompter provides the generation of prompts specific to the
    set intersection example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    intersection_prompt = """<Instruction> Find the intersection of two sets of numbers. Output only the set of numbers that are present in both sets, no additional text. </Instruction>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Output: [24, 10, 20, 8]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Output: [40, 15, 5, 24, 35, 59, 16, 63]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Output: [115, 35, 90, 117, 63, 40, 30, 33, 15, 89, 50, 12, 2, 84, 7, 57, 96]
</Examples>

Input Set 1: {set1}
Input Set 2: {set2}"""

    intersection_prompt_cot = """<Instruction> Find the intersection of two sets of numbers. You can generate any intermediate solutions, but the final output should be the set of numbers that are present in both sets, prefixed with "Output: ". </Instruction>

<Approach>
To find the intersection of the two sets follow these steps:
1. Split the second input set of numbers into two to four subsets, each containing an equal number of elements from the original set (make sure they don't overlap).
2. For each subset find the set of numbers that are present in the subset and the first input set.
3. Merge the resulting sets into a single output set.
</Approach>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Subsets of Input Set 2:
[25, 24, 10, 4, 27, 0, 14, 12]
[8, 2, 29, 20, 17, 19, 26, 23]
Intersected Subsets with Input Set 1:
[24, 10]
[8, 20]
Output: [24, 10, 8, 20]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Subsets of Input Set 2:
[16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53]
[58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Intersected Subsets with Input Set 1:
[16, 15, 5, 24]
[35, 63, 40, 59]
Output: [16, 15, 5, 24, 35, 63, 40, 59]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Subsets of Input Set 2:
[13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122]
[85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12]
[65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81]
[124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Intersected Subsets with Input Set 1:
[35, 96]
[15, 33, 57, 30, 89, 12]
[84, 40, 7, 50, 90]
[63, 117, 115, 2]
Output: [35, 96, 15, 33, 57, 30, 89, 12, 84, 40, 7, 50, 90, 63, 117, 115, 2]
</Examples>

Input Set 1: {set1}
Input Set 2: {set2}"""

    tot_improve_prompt = """<Instruction> The following three sets represent two sets and an intersection set of those two sets. The intersection set is not correct. Fix the intersection set so that it is correct.
Make sure that the numbers in the intersection set can be found in both input sets. </Instruction>

<Approach>
To fix the incorrectly intersection set follow these steps:
1. Check for each number in the incorrect intersection set, whether it can be found in both input sets. If not, remove that number from the intersection set.
2. Iterate through the second input set and check whether each number is already in the incorrect intersection set and if not, check whether that number can also be found in the first input set. If so, add that number to the intersection set.
</Approach>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Incorrect Intersection Set: [24, 20, 25]
Reason: The incorrect intersection set contains the number 25, which is not present in the first input set and is missing the numbers 10 and 8.
Output: [24, 10, 20, 8]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Incorrect Intersection Set: [57, 16, 15, 24, 35, 10, 40]
Reason: The incorrect intersection set contains the numbers 57, which is not present in the second input set, and 10, which is not present in the first input set, and is missing the numbers 5, 63 and 59.
Output: [16, 15, 5, 24, 35, 63, 40, 59]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Incorrect Intersection Set: [35, 96, 44, 15, 33, 57, 30, 50, 90, 119, 123, 63, 117, 115, 2]
Reason: The incorrect intersection set contains the numbers 44 and 119, which are not present in the second input set, and 123, which is not present in the first input set, and is missing the numbers 89, 12, 84, 40 and 7.
Output: [35, 96, 15, 33, 57, 30, 89, 12, 84, 40, 7, 50, 90, 63, 117, 115, 2]
</Examples>

Input Set 1: {set1}
Input Set 2: {set2}
Incorrect Intersection Set: {incorrect_intersection}
"""

    got_split_prompt = """<Instruction> Split the following list of 32 numbers into 2 lists of 16 numbers each, the first list should contain the first 16 numbers and the second list the second 16 numbers.
Only output the 2 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [13, 16, 30, 6, 21, 7, 31, ...],
    "List 2": [25, 24, 10, 4, 27, 0, 14, ...]
}} </Instruction>

<Example>
Input: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Output:
{{
    "List 1": [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22],
    "List 2": [33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
}}
</Example>

Input: {input}"""

    got_merge_prompt = """<Instruction> Merge the following 2 lists into one list by appending the second list to the first list.
Only output the final list without any additional text or thoughts! </Instruction>

List 1: {input1}
List 2: {input2}
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

        return self.got_merge_prompt.format(
            input1=state_dicts[0]["current"],
            input2=state_dicts[1]["current"],
        )

    def generate_prompt(
        self,
        num_branches: int,
        set1: str,
        set2: str,
        current: str,
        method: str,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param set1: First input set.
        :type set1: str
        :param set2: Second input set.
        :type set2: str
        :param current: Intermediate solution.
        :type current: str
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If the requested number of branches is not one.
        """

        assert num_branches == 1, "Branching should be done via multiple requests."
        if method.startswith("io"):
            return self.intersection_prompt.format(set1=set1, set2=set2)
        elif method.startswith("cot"):
            return self.intersection_prompt_cot.format(set1=set1, set2=set2)
        elif method.startswith("tot"):
            if current is None or current == "":
                return self.intersection_prompt.format(set1=set1, set2=set2)
            return self.tot_improve_prompt.format(
                set1=set1, set2=set2, incorrect_intersection=current
            )
        elif method.startswith("got"):
            if kwargs["phase"] == 0:
                return self.got_split_prompt.format(input=set2)

            input_set = set2
            if "subset" in kwargs and kwargs["subset"] != "":
                input_set = kwargs["subset"]

            return self.intersection_prompt.format(set1=set1, set2=input_set)

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
    # all prompts for moe and llm judge
    def moe_lenses_generate(self) -> List[str]:
        return [
            "THE REVERSE CHECKER: Assume your previous overlap was incomplete. Scan the lists backwards to catch missed elements.",
            "THE SUBTRACTOR: Find all numbers in Set 1. Subtract any numbers that do NOT appear in Set 2. Output what remains.",
            "THE DUPLICATE REMOVER: Combine both lists perfectly, and ensure no number appears twice in the final output."
        ]

    def moe_lenses_aggregate(self) -> List[str]:
        return[
            "THE CONCATENATOR: You are simply joining two lists of valid intersections. Do not drop any elements from either list.",
            "THE DUPLICATE REMOVER: Combine both lists perfectly, and ensure no number appears twice in the final output.",
            "THE SYNTAX ENFORCER: Output ONLY a flat Python list. No conversational text."
        ]
    def judge_prompt_generate(self, base_state: Dict, collapsed_output: str) -> str:
        set1 = base_state.get("set1", "[]")
        set2 = base_state.get("subset", base_state.get("set2", "[]"))
        return (
            f"You are a strict data validation AI.\n\n"
            f"Task: Find the exact mathematical intersection of Set 1 and Set 2.\n\n"
            f"Set 1: {set1}\n"
            f"Set 2: {set2}\n\n"
            f"Proposed Output:\n{collapsed_output}\n\n"
            f"To verify if the output is perfect, follow these steps:\n"
            f"Step 1: Iterate through every single number in Set 2. Check if it also exists in Set 1.\n"
            f"Step 2: Create a list of all numbers found in BOTH sets.\n"
            f"Step 3: Compare your list to the Proposed Output. Did the Proposed Output miss any numbers? Did it include numbers not present in BOTH sets?\n"
            f"Step 4: Check if the Proposed Output is strictly a flat Python list with no conversational text.\n\n"
            f"Write out your step-by-step verification. Conclude on a new line with exactly 'VERDICT: YES' if it is perfect, or 'VERDICT: NO' if it missed overlaps, hallucinated, or broke formatting."
        )

    def judge_prompt_aggregate(self, previous_thought_states: List[Dict], collapsed_output: str) -> str:
        list1 = previous_thought_states[0].get("current", "[]")
        list2 = previous_thought_states[1].get("current", "[]") if len(previous_thought_states) > 1 else "[]"
        return (
            f"You are a strict data validation AI.\n\n"
            f"Task: Merge List 1 and List 2 into a single flat list. Do NOT remove duplicates.\n\n"
            f"List 1: {list1}\n"
            f"List 2: {list2}\n\n"
            f"Proposed Output:\n{collapsed_output}\n\n"
            f"To verify if the output is mathematically perfect, you MUST follow these steps:\n"
            f"Step 1: Count the exact number of elements in List 1 and List 2. Sum these counts.\n"
            f"Step 2: Count the exact number of elements in the Proposed Output array.\n"
            f"Step 3: Compare the counts. If they do not match exactly, it dropped elements.\n"
            f"Step 4: Check if the Proposed Output is strictly a flat Python list with no conversational text.\n\n"
            f"Write out your step-by-step verification. Conclude on a new line with exactly 'VERDICT: YES' if it is perfect, or 'VERDICT: NO' if it dropped numbers or broke formatting."
        )

    def moe_critique_generate(self, collapsed_output: str, lens: str) -> str:
        return (
            f"\n\n[System Intervention]: Your previous output was evaluated as INCORRECT:\n"
            f"=== FAILED OUTPUT ===\n{collapsed_output}\n=====================\n\n"
            f"You must NOT generate this exact output again. To break your previous reasoning pattern, adopt this specific perspective:\n"
            f"-> {lens}\n\n"
            f"First, write 1-2 sentences applying this perspective to identify your mistake. "
            f"Then, you MUST output the corrected data strictly formatted within brackets (e.g.,[x, y, z])."
        )

    def moe_critique_aggregate(self, collapsed_output: str, lens: str) -> str:
        return self.moe_critique_generate(collapsed_output, lens)

    
class SetIntersectionParser(parser.Parser):
    """
    SetIntersectionParser provides the parsing of language model reponses
    specific to the set intersection example.

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

            answers_stripped = [
                answer for answer in answers if "[" in answer and "]" in answer
            ]
            if len(answers_stripped) == 0:
                for answer in answers:
                    answer = "[" + answer + "]"
                    try:
                        answer_converted = utils.string_to_list(answer)
                        if len(answer_converted) > 0:
                            answers_stripped.append(answer)
                    except:
                        pass
            if len(answers_stripped) == 0:
                logging.warning(
                    f"Could not parse aggregation answer: {text}. Returning empty list."
                )
                answer = "[]"
            else:
                answer = [
                    answer[answer.index("[") : answer.index("]") + 1]
                    for answer in answers_stripped
                ][0]
            states = sorted(states, key=lambda x: x["part"])
            merged_subsets = states[0]["subset"][:-1] + ", " + states[1]["subset"][1:]
            new_state = states[0].copy()
            new_state["current"] = answer
            new_state["subset"] = merged_subsets
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
            if state["method"].startswith("got") and state["phase"] == 0:
                # We expect a json which contains the two lists named "List 1" and "List 2"
                # cut everything until the opening bracket and everything after the closing bracket

                try:
                    text = text[text.index("{") : text.index("}") + 1]
                    json_dict = json.loads(text)
                    if len(json_dict.keys()) != 2:
                        logging.warning(
                            f"Expected 2 lists in json, but found {len(json_dict.keys())}."
                        )
                    for key, value in json_dict.items():
                        if "List" not in key:
                            logging.warning(
                                f"Expected key to contain 'List', but found {key}."
                            )
                            continue
                        if not isinstance(value, list):
                            value = utils.string_to_list(value)
                        new_state = state.copy()
                        new_state["current"] = ""
                        new_state["subset"] = str(value)
                        new_state["phase"] = 1
                        new_state["part"] = key
                        new_states.append(new_state)
                except Exception as e:
                    logging.error(
                        f"Could not parse step answer: {text}. Encountered exception: {e}"
                    )
            else:
                answers = text.strip().split("\n")
                answers = [
                    answer for answer in answers if "[" in answer and "]" in answer
                ]
                if any(["Output" in answer for answer in answers]):
                    # cut elements until last output is found
                    for answer in reversed(answers):
                        if "Output" in answer:
                            answers = answers[answers.index(answer) :]
                            break

                answers = [
                    answer[answer.index("[") : answer.index("]") + 1]
                    for answer in answers
                ]
                if len(answers) == 0:
                    logging.warning(
                        f"Could not parse step answer: {text}. Returning empty list."
                    )
                    answer = "[]"
                else:
                    if len(answers) > 1:
                        logging.warning(
                            f"Multiple answers found for step answer: {text}. Using the first one."
                        )
                    answer = answers[0]

                new_state = state.copy()
                new_state["current"] = answer
                new_state["phase"] = 2
                new_states.append(new_state)
        return new_states

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

def check_set_validity(output_str, previous_states_or_base_state):
    try:
        import ast, re
        match = re.search(r'\[[\d,\s]+\]', str(output_str))
        if not match: return False
        out_set = set(ast.literal_eval(match.group()))
        
        # if aggregate logic:
        if isinstance(previous_states_or_base_state, list): 
            list1 = ast.literal_eval(previous_states_or_base_state[0].get("current", "[]"))
            list2 = ast.literal_eval(previous_states_or_base_state[1].get("current", "[]"))
            true_combined = set(list1).union(set(list2))
            return out_set == true_combined
            
        # if generate logic:
        base_state = previous_states_or_base_state
        set1 = set(ast.literal_eval(re.search(r'\[[\d,\s]+\]', str(base_state.get("set1", "[]"))).group()))
        set2 = set(ast.literal_eval(re.search(r'\[[\d,\s]+\]', str(base_state.get("subset", base_state.get("set2", "[]")))).group()))
        
        return out_set == set1.intersection(set2)
    except:
        return False

def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(
        operations.GroundTruth(utils.test_set_intersection)
    )

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
    operations_graph.append_operation(
        operations.GroundTruth(utils.test_set_intersection)
    )

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
    op_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(op_1)

    for _ in range(1):
        operations_graph.append_operation(operations.Generate(1, 20))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        op_2 = operations.KeepBestN(1, False)
        op_2.add_predecessor(op_1)
        operations_graph.append_operation(op_2)
        op_1 = op_2

    operations_graph.append_operation(
        operations.GroundTruth(utils.test_set_intersection)
    )

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
    op_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(op_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, 10))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        op_2 = operations.KeepBestN(1, False)
        op_2.add_predecessor(op_1)
        operations_graph.append_operation(op_2)
        op_1 = op_2

    operations_graph.append_operation(
        operations.GroundTruth(utils.test_set_intersection)
    )

    return operations_graph


#helper fn for various ablations
def create_proactive_got(intervention_enabled=True, use_llm_judge=True, use_moe=True, validator_fn=None):
    operations_graph = operations.GraphOfOperations()

    plans = operations.Generate(1, 1)
    operations_graph.append_operation(plans)
    
    branch_endpoints =[]
    for i in range(1, 3):
        list_id = f"List {i}"
        sub_list = operations.Selector(
            lambda thoughts, lid=list_id:[t for t in thoughts if t.state["part"] == lid]
        )
        sub_list.add_predecessor(plans)
        operations_graph.add_operation(sub_list)
        
        # proactive generate
        intersected_subset = operations.ProactiveGenerate(1, 5, 0.90, intervention_enabled, use_llm_judge, use_moe, validator_fn)
        intersected_subset.add_predecessor(sub_list)
        operations_graph.add_operation(intersected_subset)
        
        score_sub_list = operations.Score(1, False, utils.num_errors)
        score_sub_list.add_predecessor(intersected_subset)
        operations_graph.add_operation(score_sub_list)
        
        keep_best_sub_list = operations.KeepBestN(1, False)
        keep_best_sub_list.add_predecessor(score_sub_list)
        operations_graph.add_operation(keep_best_sub_list)
        branch_endpoints.append(keep_best_sub_list)

    # proactive aggregate
    final_aggregate = operations.ProactiveAggregate(10, 0.90, intervention_enabled, use_llm_judge, use_moe, validator_fn)
    for endpoint in branch_endpoints: 
        final_aggregate.add_predecessor(endpoint)
    operations_graph.add_operation(final_aggregate)

    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_aggregate_final = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_aggregate_final)

    operations_graph.append_operation(operations.GroundTruth(utils.test_set_intersection))
    return operations_graph

def got_original() -> operations.GraphOfOperations:
    # original got behavior
    g = operations.GraphOfOperations()
    plans = operations.Generate(1, 1)
    g.append_operation(plans)
    endpoints =[]
    for i in range(1, 3):
        sub_list = operations.Selector(lambda t, lid=f"List {i}": [x for x in t if x.state["part"] == lid])
        sub_list.add_predecessor(plans)
        g.add_operation(sub_list)
        subset = operations.Generate(1, 5)
        subset.add_predecessor(sub_list)
        g.add_operation(subset)
        score = operations.Score(1, False, utils.num_errors)
        score.add_predecessor(subset)
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
    g.append_operation(operations.GroundTruth(utils.test_set_intersection))
    return g

def got_2_nodes(): 
    return create_proactive_got(intervention_enabled=False)

def got_python_moe(): 
    return create_proactive_got(use_llm_judge=False, use_moe=True, validator_fn=check_set_validity)

def got_python_no_moe(): 
    return create_proactive_got(use_llm_judge=False, use_moe=False, validator_fn=check_set_validity)

def got_llm_no_moe(): 
    return create_proactive_got(use_llm_judge=True, use_moe=False)

def got_full(): 
    return create_proactive_got(use_llm_judge=True, use_moe=True)


def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
    config_path: str,
    temperature: float = None,
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample.
    """
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "set_intersection_032.csv")
    data = []
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([int(row[0]), row[1], row[2], row[3]])

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    # add output dir, temp config, + time tracking for metrics
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_val = temperature if temperature is not None else 0.6 
    temp_str = f"_T{str(temp_val).replace('.', 'p')}"
    extra_info = f"{lm_name}{temp_str}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}_{os.getpid()}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder, exist_ok=True)

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__), exist_ok=True)

    # change to load model once
    logging.info("Loading language model(once)")
    lm = language_models.Llama2HF(config_path, model_name=lm_name, temperature=temperature)
    logging.info("Language model loaded.")

    for data in selected_data:
        logging.info(f"Running data {data[0]}: {data[1]} {data[2]}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            # got rid of chatgpt config
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                SetIntersectionPrompter(),
                SetIntersectionParser(),
                {
                    "set1": data[1],
                    "set2": data[2],
                    "result": data[3],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )

            #timing
            start_time = datetime.datetime.now().timestamp()
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            end_time = datetime.datetime.now().timestamp()
            execution_time = end_time - start_time
                        
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data[0]}.json",
            )
            
            #original graph json
            executor.output_graph(path)
            
            # put execution time
            if os.path.exists(path):
                with open(path, "r") as f:
                    graph_data = json.load(f)
                
                # make sure type safe (for safety)
                if isinstance(graph_data, dict):
                    graph_data["execution_time_seconds"] = execution_time
                elif isinstance(graph_data, list):
                    graph_data.append({"execution_time_seconds": execution_time})
                
                with open(path, "w") as f:
                    json.dump(graph_data, f, indent=4)
            
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    """
    Input(x)  : a list of 32 numbers between 0 and 63 (inclusive)
    Input(y)  : a list of 32 numbers between 0 and 63 (inclusive)
    Output(z) : a list of the intersection between x and y
    Correct   : z = intersection(x, y)
    Input Example:
        [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
        [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
    Output Example:
        [24, 10, 20, 8]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--model_name", type=str, default="qwen2.5-14b", help="Model name key from config.json")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature from config")
    args = parser.parse_args()

    budget = 100
    # run on 50 samples
    samples = [item for item in range(0, 100)]
    

    approaches =[io, cot, got_original, got_2_nodes, got_python_moe,got_python_no_moe, got_llm_no_moe, got_full]
    spent = run(samples, approaches, budget, args.model_name, args.config_path, args.temperature)

    logging.info(f"Spent {spent} out of {budget} budget.")
