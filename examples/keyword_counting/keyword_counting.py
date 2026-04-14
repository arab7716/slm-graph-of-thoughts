# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# The source code is adapted from the sorting source code written by
# Nils Blach.
#
# main author: Nils Blach
# contributions: Ales Kubicek
# Reconfigured for proactive failure mitigation by Artha Abeysinghe 
# Generative AI was used to assist partially with syntax. Every line of code was nonetheless thoroughly human reviewed.

import argparse
import os
import logging
import datetime
import json
import csv
from collections import Counter
from functools import partial
from typing import Dict, List, Callable, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from functools import partial 


def string_to_list(string: str) -> List[str]:
    """
    Helper function to convert a list encoded inside a string into a Python
    list object of string elements.

    :param string: Input string containing a list.
    :type string: str
    :return: List of string elements.
    :rtype: List[str]
    :raise AssertionError: If input string does not contain a list.
    """

    assert string[0] == "[" and string[-1] == "]", "String is not a list."
    return [
        item.strip().replace("'", "").replace('"', "")
        for item in string[1:-1].split(", ")
    ]


def list_to_freq_dict(lst: List[str]) -> Dict[str, int]:
    """
    Helper function that converts a list of string elements, where each element
    can occur multiple times, into a dictionary, where the elements are the keys
    and the number of their occurrences in the input list is the value.

    :param lst: List of string elements.
    :type lst: List[str]
    :return: Frequency dictionary of string elements.
    :rtype: Dict[str, int]
    """

    return dict(Counter(lst))


def valid_aggregation(state: Dict) -> bool:
    """
    Helper function to determine whether the aggregation of two intermediate
    solutions produces valid results.

    :param state: Thought state resulting from an aggregation of thoughts.
    :type state: Dict
    :return: Returns whether the aggregation produced valid results.
    :rtype: bool
    """

    aggr1 = json.loads(state["aggr1"])
    aggr2 = json.loads(state["aggr2"])
    current = json.loads(state["current"])

    if set(aggr1.keys()) | set(aggr2.keys()) != set(current.keys()):
        return False

    for country in current.keys():
        aggr1_freq = aggr1[country] if country in aggr1.keys() else 0
        aggr2_freq = aggr2[country] if country in aggr2.keys() else 0
        if aggr1_freq + aggr2_freq != current[country]:
            return False

    return True


def num_errors(all_possible_countries: List[str], state: Dict) -> float:
    """
    Function to locally count the number of errors that serves as a score.

    :param all_possible_countries: List of keywords.
    :type all_possible_countries: List[str]
    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """

    try:
        if (
            "sub_text" in state
            and (state["sub_text"] != "" or state["current"] == "{}")
            and len(state["sub_text"]) < len(state["original"]) * 0.75
        ):
            text = state["sub_text"]
            correct_freq_dict = dict()
            for country in all_possible_countries:
                # find number of times country appears in text
                num_occurrences = text.count(country)
                correct_freq_dict[country] = num_occurrences
        else:
            correct_freq_dict = list_to_freq_dict(string_to_list(state["ground_truth"]))
        current_freq_dict = json.loads(state["current"])
        countries_not_in_current = set(correct_freq_dict.keys()) - set(
            current_freq_dict.keys()
        )
        countries_not_in_correct = set(current_freq_dict.keys()) - set(
            correct_freq_dict.keys()
        )
        # count the number of errors
        num_errors = 0
        for country in countries_not_in_current:
            num_errors += abs(correct_freq_dict[country])
        for country in countries_not_in_correct:
            num_errors += abs(current_freq_dict[country])
        for country in set(correct_freq_dict.keys()) & set(current_freq_dict.keys()):
            num_errors += abs(correct_freq_dict[country] - current_freq_dict[country])
        return num_errors
    except:
        return 100


def test_keyword_counting(state: Dict) -> bool:
    """
    Function to test whether the final solution matches ground truth.

    :param state: Thought state that represents the final solution.
    :type state: Dict
    :return: Returns whether the solution matches the ground truth.
    :rtype: bool
    """

    try:
        ground_truth = state["ground_truth"]
        correct_freq_dict = list_to_freq_dict(string_to_list(ground_truth))
        current_freq_dict = json.loads(state["current"])
        # check that the keys are the same
        if set(correct_freq_dict.keys()) != set(current_freq_dict.keys()):
            return False
        # check that the values are the same
        for key in correct_freq_dict.keys():
            if correct_freq_dict[key] != current_freq_dict[key]:
                return False
        return True
    except:
        return False


class KeywordCountingPrompter(prompter.Prompter):
    """
    KeywordCountingPrompter provides the generation of prompts specific to the
    keyword counting example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    count_prompt = """<Instruction> Count the frequency of how many times each country is explicitly named in the input text. Output only the frequency of each country that appears at least once in the following json format; make sure to keep the same spelling and output no additional text:
{{
    "country1": frequency1,
    "country2": frequency2,
    ...
}}
</Instruction>

<Examples>
Input:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Output: 
{{
    "Canada": 1,
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1    
}}

Input:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Output: 
{{
    "Peru": 2,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}

Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Output:
{{
    "Italy": 2,
    "France": 1,
    "Spain": 1,
    "Greece": 1,
    "Norway": 2,
    "Sweden": 2,
    "Finland": 1,
    "Denmark": 1,
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 2,
    "Russia": 1
}}
</Examples>

Input:
{input}
Output:
"""

    count_prompt_cot = """<Instruction> Count the frequency of how many times each country is explicitly named in the input text. You can generate any intermedate lists and states, but the final output should only contain the frequency of each country that appears at least once in the following json format, prefixed with "Output: " (make sure to keep the same spelling for each country in the output as in the input text):
{{
    "country1": frequency1,
    "country2": frequency2,
    ...
}}
</Instruction>

<Approach>
To count the frequency for each country follow these steps:
1. Split the input passage into four paragraphs of similar length.
2. Count the frequency of each country in each paragraph.
3. Combine the frequencies of each country from each paragraph by adding them together.
</Approach>

<Examples>
Input:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Paragraphs:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. 

Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Sublist frequencies:
{{
    "Canada": 1
}}

{{
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1
}}
Output: 
{{
    "Canada": 1,
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1
}}

Input:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Paragraphs:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. 

A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Sublists:
{{
    "Peru": 1,
    "Chile": 1
}}

{{
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Peru": 1,
    "Brazil": 1,
    "Mexico": 1
}}
Output: 
{{
    "Peru": 2,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}

Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Paragraphs:
Journeying westward, she admired the art in Italy and sipped coffee in France. 

The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. 

She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. 

Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Sublists:
{{
    "Italy": 1,
    "France": 1
}}

{{
    "Spain": 1,
    "Greece": 1,
    "Norway": 1,
    "Sweden": 1,
    "Finland": 1,
    "Denmark": 1
}}

{{
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 1,
    "Russia": 1
}}

{{
    "Italy": 1,
    "Norway": 1,
    "Sweden": 1,
    "Germany": 1
}}
Output: 
{{
    "Italy": 2,
    "France": 1,
    "Spain": 1,
    "Greece": 1,
    "Norway": 2,
    "Sweden": 2,
    "Finland": 1,
    "Denmark": 1,
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 2,
    "Russia": 1
}}
</Examples>

Input:
{input}
"""

    count_prompt_sentence = """<Instruction> Count the frequency of how many times each country is explicitly named in the input text. Output only the frequency of each country that appears at least once in the following json format; make sure to keep the same spelling and output no additional text:
{{
    "country1": frequency1,
    "country2": frequency2,
    ...
}}
</Instruction>

<Approach>
To count the frequency for each country follow these steps:
1. Create an empty dictionary.
2. Iterate through the text word by word.
3. If the word corresponds to a country, add the country to the dictionary and set its value to 1 if it is not already in the dictionary. If the word is already in the dictionary, increment its value by 1.
</Approach>

<Examples>
Input:
Alexandra explored the rainforests of Brazil and danced the tango in Argentina.
Output: 
{{
    "Brazil": 1,
    "Argentina": 1    
}}

Input:
In Norway she found stones that were identical to those in Sweden, indicating a deep-rooted cultural connection between Sweden and Norway.
Output:
{{
    "Norway": 2,
    "Sweden": 2
}}

Input:
A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Output: 
{{
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Peru": 1,
    "Brazil": 1,
    "Mexico": 1
}}

Input:
Italy, Sweden, Sweden and Germany will always stay her favourite destinations to visit.
Output:
{{
    "Italy": 1,
    "Sweden": 2,
    "Germany": 1
}}
</Examples>

Input:
{input}
Output:
"""

    tot_improve_prompt = """<Instruction> The following two inputs represent an initial input text and a dictionary of countries and their frequencies of explicit appearance in the input text. The dictionary is incorrect and might not contain all countries, extra countries or incorrect frequencies.
Fix the dictionary such that it has the correct frequencies for each country that appears at least once in the input text. </Instruction>

<Approach>
To fix the incorrect list of countries follow these steps:
1. Iterate through the input text and find all countries that are explicitly mentioned.
2. Count the frequency of each country in the input text.
3. Compare the frequency of each country in the input text with the frequency of the country in the incorrect dictionary and update the frequency in the incorrect dictionary if they are different.

</Approach>

<Examples>
Input:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Incorrect Dictionary:
{{
    "Canada": 1,
    "Mexico": 1,
    "Argentina": 1
}}
Reason: The input text names Brasil once but the incorrect dictionary does not contain Brasil at all, the remaining countries are correct.
Output: 
{{
    "Canada": 1,
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1
}}

Input:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Incorrect Dictionary:
{{
    "Peru": 3,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1,
    "Argentina": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}
Reason: The input text names Peru twice, but the incorrect dictionary lists it with a frequency of 3 instead of 2. The incorrect dictionary also contains Argentina which does not appear in the input text.
Output: 
{{
    "Peru": 2,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}

Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Incorrect Dictionary:
{{
    "Italy": 1,
    "France": 1,
    "Spain": 1,
    "Greece": 1,
    "Norway": 1,
    "Sweden": 1,
    "Finland": 1,
    "Denmark": 1,
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 1,
    "Russia": 1
}}
Reason: The input text names Italy, Norway, Sweden and Germany twice each, but the incorrect dictionary lists them with a frequency of 1 each instead of 2.
Output: 
{{
    "Italy": 2,
    "France": 1,
    "Spain": 1,
    "Greece": 1,
    "Norway": 2,
    "Sweden": 2,
    "Finland": 1,
    "Denmark": 1,
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 2,
    "Russia": 1
}}
</Examples>

Input: 
{input}
Incorrect Dictionary: 
{incorrect_dict}
"""

    sentence_improve_prompt = """<Instruction> The following two inputs represent an initial input text (usually a sinlge sentence) and a dictionary of countries and their frequencies of explicit appearance in the input text/sentence. The dictionary is incorrect and might not contain all countries, contain extra countries or countries with incorrect frequencies.
Fix the dictionary such that it has the correct frequencies for each country and only contains countries that are explicitly named in the text/sentence. </Instruction>

<Approach>
To fix the incorrect dictionary of countries follow these steps:
1. Iterate through the input text/sentence and find all countries that are explicitly mentioned.
2. For each of these countries, count how many times they are explicitly mentioned in the input text/sentence.
3. Compare the frequency of each country in the input text with the frequency of the country in the incorrect dictionary and update the frequency in the incorrect dictionary if they are different.

</Approach>

<Examples>
Input:
Alexandra boarded the first flight of her grand journey, starting from Canada. 
Incorrect Dictionary:
{{
    "Canada": 1,
    "Mexico": 1,
    "Argentina": 1
}}
Reason: The input text only names Canada once, but the incorrect dictionary contains Mexico and Argentina which do not appear in the input text.
Output: 
{{
    "Canada": 1
}}

Input:
A quick detour to Peru and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Incorrect Dictionary:
{{
    "Peru": 3,
    "Argentina": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}
Reason: The input text names Peru twice, but the incorrect dictionary lists it with a frequency of 3 instead of 2. The incorrect dictionary also contains Argentina which does not appear in the input text and is missing Paraguay.
Output: 
{{
    "Peru": 2,
    "Paraguay": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}

Input:
She danced in Ireland and Russia, explored castles in England, and marveled at the architecture in Germany and Russia. 
Incorrect Dictionary:
{{
    "Ireland": 1,
    "England": 1,
    "Germany": 1,
    "Russia": 1
}}
Reason: The input text names Russia twice each, but the incorrect dictionary lists Russia with a frequency of 1 instead of 2. The incorrect dictionary also contains England which does not appear in the input text and is missing Scotland.
Output: 
{{
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 1,
    "Russia": 2
}}
</Examples>

Input: 
{input}
Incorrect Dictionary: 
{incorrect_dict}
"""

    got_split_prompt = """<Instruction> Split the following input text into 4 paragraphs of approximately same length.
Only output the final 4 paragraphs in the following format without any additional text or thoughts:
{{
    "Paragraph 1": "Some paragraph text ...",
    "Paragraph 2": "Some paragraph text ...",
    "Paragraph 3": "Some paragraph text ...",
    "Paragraph 4": "Some paragraph text ..."
}} </Instruction>

<Example>
Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Output: 
{{
    "Paragraph 1": "Journeying westward, she admired the art in Italy and sipped coffee in France. ",
    "Paragraph 2": "The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away.",
    "Paragraph 3": "She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.",
    "Paragraph 4": "Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit."
}}
</Example>

Input:
{input}
"""

    got_split_prompt2 = """<Instruction> Split the following input text into 8 paragraphs of approximately same length.
Only output the final 8 paragraphs in the following format without any additional text or thoughts:
{{
    "Paragraph 1": "Some paragraph text ...",
    "Paragraph 2": "Some paragraph text ...",
    "Paragraph 3": "Some paragraph text ...",
    "Paragraph 4": "Some paragraph text ...",
    "Paragraph 5": "Some paragraph text ...",
    "Paragraph 6": "Some paragraph text ...",
    "Paragraph 7": "Some paragraph text ...",
    "Paragraph 8": "Some paragraph text ..."
}} </Instruction>

<Example>
Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit. However, nothing compared to her experiences in Egypt, where she began her journey as an archaeologist. One evening in Egypt, she discovered a mysterious artifact that existed not only in Egypt but also in distant lands like Peru and Canada. The artifact was said to harness the energy of the earth, which she only started believing when experiencing it while traveling in Sweden and Notway. A similar relic was rumored to exist in the bustling streets of Thailand and the snowy landscapes of Sweden.
Output: 
{{
Output: 
    "Paragraph 1": "Journeying westward, she admired the art in Italy and sipped coffee in France. ",
    "Paragraph 2": "The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. ",
    "Paragraph 3": "She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. ",
    "Paragraph 4": "Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit. ",
    "Paragraph 5": "However, nothing compared to her experiences in Egypt, where she began her journey as an archaeologist. ",
    "Paragraph 6": "One evening in Egypt, she discovered a mysterious artifact that existed not only in Egypt but also in distant lands like Peru and Canada. ",
    "Paragraph 7": "The artifact was said to harness the energy of the earth, which she only started believing when experiencing it while traveling in Sweden and Notway. ",
    "Paragraph 8": "A similar relic was rumored to exist in the bustling streets of Thailand and the snowy landscapes of Sweden."
}}
</Example>

Input:
{input}
"""

    got_split_prompt3 = """<Instruction> Split the following input text into individual sentences.
Output each sentence in the following format without any additional text or thoughts:
{{
    "Sentence 1": "Some sentence text ...",
    "Sentence 2": "Some sentence text ...",
    "Sentence 3": "Some sentence text ...",
    ...
}} </Instruction>

<Example>
Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Output: 
{{
    "Sentence 1": "Journeying westward, she admired the art in Italy and sipped coffee in France. ",
    "Sentence 2": "The music of Spain and the history of Greece deepened her love for Europe. "
    "Sentence 3": "The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away.",
    "Sentence 4": "She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.",
    "Sentence 5": "Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit."
}}
</Example>

Input:
{input}
"""

    got_aggregate_prompt = """<Instruction> Combine the following 2 dictionaries, each containing the frequency of countries in a text, into a single dictionary.
Simply add the frequencies together for each country and if a country is not present in one of the dictionaries, add it to the final dictionary with the frequency from the other dictionary.
Only output the final merged dictionary without any additional text or thoughts! </Instruction>

<Approach>
To combine the 2 dictionaries into single one, follow these steps:
1. Create a new dictionary to store the combined frequencies.
2. Iterate through the keys of the first dictionary and add the frequency of each country to the new dictionary.
3. Iterate through the keys of the second dictionary and add the frequency of each country to the new dictionary and if it is already present, add the frequency to the existing value.
</Approach>

Combine the following 2 dictionaries into a single dictionary:
{input1}

{input2}

Combined Output:
"""

    got_improve_aggregate_prompt = """<Instruction> The following 2 dictionaries were combined into the third dictionary below. 
However, some mistakes occured and the third dictionary is incorrect. Please fix the third dictionary so that it contains the correct frequencies for each country.
The correct frequencies are the sum of the frequencies from the first 2 dictionaries. If a country is not present in one of the dictionaries, add it to the final dictionary with the frequency from the other dictionary.

<Example>
Dictionary 1:
{{
    "Peru": 2,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1
}}
Dictionary 2:
{{
    "Peru": 1,
    "Argentina": 1,
    "Canada": 1,
    "Chile": 3,
    "Germany": 2
}}
Incorrectly Combined Dictionary:
{{
    "Peru": 3,
    "Chile": 2,
    "Uruguay": 1,
    "Paraguay": 1,
    "Argentina": 1,
    "Chile": 3,
    "Germany": 2
}}
Output:
{{
    "Peru": 3,
    "Chile": 4,
    "Uruguay": 1,
    "Paraguay": 1,
    "Argentina": 1,
    "Canada": 1,
    "Germany": 2
}}
</Example>

Dictionary 1:
{input1}
Dictionary 2:
{input2}
Incorrectly Combined Dictionary:
{input3}
Output:
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        :raise AssertionError: If more than two thought states are provided.
        """
        assert len(state_dicts) <= 2, "Expected 2 states for aggregation prompt."
        if len(state_dicts) == 0:
            state_dicts = [{"current": "{}"}, {"current": "{}"}]
        elif len(state_dicts) == 1:
            state_dicts.append({"current": "{}"})
        return self.got_aggregate_prompt.format(
            input1=state_dicts[0]["current"], input2=state_dicts[1]["current"]
        )

    def generate_prompt(
        self, num_branches: int, original: str, current: str, method: str, **kwargs
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param original: Input text.
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

        assert num_branches == 1, "Branching should be done via multiple requests."
        if current is None or current == "":
            input = original
        else:
            input = current
        if method.startswith("io"):
            return self.count_prompt.format(input=input)
        elif method.startswith("cot"):
            return self.count_prompt_cot.format(input=input)
        elif method.startswith("tot"):
            if current is None or current == "":
                return self.count_prompt_cot.format(input=input)
            return self.tot_improve_prompt.format(
                input=original,
                incorrect_dict=current,
            )
        elif method.startswith("got"):
            if (current is None or current == "") and kwargs["phase"] == 0:
                if method == "got8":
                    return self.got_split_prompt2.format(input=input)
                if method == "gotx":
                    return self.got_split_prompt3.format(input=input)
                return self.got_split_prompt.format(input=input)

            if kwargs["phase"] == 1:
                if method == "gotx":
                    return self.count_prompt_sentence.format(input=kwargs["sub_text"])
                return self.count_prompt_cot.format(input=kwargs["sub_text"])

            if (
                "sub_text" in kwargs
                and kwargs["sub_text"] != ""
                and len(kwargs["sub_text"]) < len(original) * 0.75
            ):
                original = kwargs["sub_text"]
            if method == "gotx":
                return self.sentence_improve_prompt.format(
                    input=original, incorrect_dict=current
                )
            return self.tot_improve_prompt.format(
                input=original, incorrect_dict=current
            )

    def improve_prompt(self, current: str, aggr1: str, aggr2: str, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param current: Intermediate solution.
        :type current: str
        :param aggr1: Partially solution 1 before aggregation.
        :type aggr1: str
        :param aggr2: Partially solution 2 before aggregation.
        :type aggr2: str
        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        return self.got_improve_aggregate_prompt.format(
            input1=aggr1, input2=aggr2, input3=current
        )

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

    def moe_lenses_generate(self) -> List[str]:
        return [
            "THE CONTEXT READER: You missed countries mentioned in passing. Re-read the text line-by-line and extract every country.",
            "THE LINGUIST: Scan the text for capitalized words that represent geographical nations. Tally them carefully.",
            "THE JSON VALIDATOR: Your previous output broke standard JSON format. Ensure all keys are strings in double quotes."
        ]

    def moe_lenses_aggregate(self) -> List[str]:
        return [
            "THE SUMMATION EXPERT: Mathematically add the values of matching keys from the input dictionaries. Do not miss any keys.",
            "THE DICTIONARY MERGER: Ensure every single country listed in the input dictionaries appears in the final output dictionary.",
            "THE JSON VALIDATOR: Output ONLY valid, parseable JSON format. Do not use markdown blocks or conversational apologies."
        ]
    
    def judge_prompt_generate(self, base_state: Dict, collapsed_output: str) -> str:
        raw_input = base_state.get("sub_text", base_state.get("original", "")).strip()
        return (
            f"You are a strict data validation AI.\n\n"
            f"Task: Extract a JSON dictionary of countries and their frequencies from the Input Text.\n\n"
            f"Input Text:\n{raw_input}\n\n"
            f"Proposed Output:\n{collapsed_output}\n\n"
            f"To verify if the output is perfect, follow these steps:\n"
            f"Step 1: Check the Input Text. If it is completely empty or blank, the Proposed Output MUST be exactly {{}}. Is the input empty?\n"
            f"Step 2: If not empty, list every country explicitly named in the text and tally its frequency.\n"
            f"Step 3: Compare your tally to the Proposed Output. Do the keys and values match exactly?\n"
            f"Step 4: Check if the Proposed Output is strictly a JSON dictionary with no conversational text or markdown.\n\n"
            f"Write out your step-by-step verification. Conclude on a new line with exactly 'VERDICT: YES' if it is perfect, or 'VERDICT: NO' if it hallucinated, missed countries, or broke formatting."
        )

    def judge_prompt_aggregate(self, previous_thought_states: List[Dict], collapsed_output: str) -> str:
        dict1 = previous_thought_states[0].get("current", "{}")
        dict2 = previous_thought_states[1].get("current", "{}") if len(previous_thought_states) > 1 else "{}"
        return (
            f"You are a strict data validation AI.\n\n"
            f"Task: Combine two frequency dictionaries by adding the values of matching keys.\n\n"
            f"Dictionary 1: {dict1}\n"
            f"Dictionary 2: {dict2}\n\n"
            f"Proposed Output:\n{collapsed_output}\n\n"
            f"To verify if the output is perfect, follow these steps:\n"
            f"Step 1: List all unique keys from Dictionary 1 and Dictionary 2.\n"
            f"Step 2: For each key, calculate the mathematical sum of its values from both dictionaries.\n"
            f"Step 3: Compare your calculated sums to the Proposed Output. Do they match exactly?\n"
            f"Step 4: Check if the Proposed Output is strictly a JSON dictionary with no conversational text or markdown.\n\n"
            f"Write out your step-by-step verification. Conclude on a new line with exactly 'VERDICT: YES' if it is perfect, or 'VERDICT: NO' if the math is wrong or format is broken."
        )

    def moe_critique_generate(self, collapsed_output: str, lens: str) -> str:
        return (
            f"\n\n[System Intervention]: Your previous output was evaluated as INCORRECT:\n"
            f"=== FAILED OUTPUT ===\n{collapsed_output}\n=====================\n\n"
            f"You must NOT generate this exact output again. To break your previous reasoning pattern, adopt this specific perspective:\n"
            f"-> {lens}\n\n"
            f"First, write 1-2 sentences applying this perspective to identify your mistake. "
            f"Then, you MUST output the corrected data strictly formatted as a JSON dictionary."
        )

    def moe_critique_aggregate(self, collapsed_output: str, lens: str) -> str:
        return self.moe_critique_generate(collapsed_output, lens) # critique prompt can be the same


class KeywordCountingParser(parser.Parser):
    """
    KeywordCountingParser provides the parsing of language model reponses
    specific to the keyword counting example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def strip_answer_json(self, text: str) -> str:
        """
        Helper function to retrieve a text from a json string.

        :param text: Input json string.
        :type text: str
        :return: Retrieved text.
        :rtype: str
        """

        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        # find the last "{" and "}" and only keep the text in between including the brackets
        start = text.rfind("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return "{}"
        text = text[start : end + 1]
        try:
            json.loads(text)
            return text
        except:
            return "{}"

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
        :raise AssertionError: If more than two thought states are provided.
        """

        assert len(states) <= 2, "Expected 2 states for aggregation answer."
        if len(states) == 0:
            states = [
                {"current": "{}", "sub_text": ""},
                {"current": "{}", "sub_text": ""},
            ]
        elif len(states) == 1:
            states.append({"current": "{}", "sub_text": ""})
        new_states = []
        for text in texts:
            answer = self.strip_answer_json(text)
            new_state = states[0].copy()
            new_state["sub_text"] = (
                states[0]["sub_text"] if "sub_text" in states[0] else ""
            ) + (states[1]["sub_text"] if "sub_text" in states[1] else "")
            new_state["current"] = answer
            new_state["aggr1"] = states[0]["current"]
            new_state["aggr2"] = states[1]["current"]
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
        :raise AssertionError: If there is not exactly one response text.
        """

        assert len(texts) == 1, "Expected 1 text for improve answer."
        text = texts[0]
        answer = self.strip_answer_json(text)
        new_state = state.copy()
        new_state["current"] = answer
        return new_state

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
            try:
                if (
                    state["method"].startswith("got")
                    and state["current"] == ""
                    and state["phase"] == 0
                ):
                    answer = self.strip_answer_json(text)
                    json_dict = json.loads(answer)
                    if len(json_dict.keys()) != 4 or len(json_dict.keys()) != 8:
                        logging.warning(
                            f"Expected 4 or 8 paragraphs in json, but found {len(json_dict.keys())}."
                        )
                    for key, value in json_dict.items():
                        if "Paragraph" not in key and "Sentence" not in key:
                            logging.warning(
                                f"Expected key to contain 'Paragraph' or 'Sentence', but found {key}."
                            )
                            continue
                        new_state = state.copy()
                        new_state["current"] = ""
                        new_state["sub_text"] = value
                        new_state["phase"] = 1
                        new_state["part"] = key
                        new_states.append(new_state)
                else:
                    answer = self.strip_answer_json(text)
                    new_state = state.copy()
                    new_state["current"] = answer
                    new_state["phase"] = 2
                    new_states.append(new_state)
            except Exception as e:
                logging.error(f"Could not parse step answer: {text}. Error: {e}")
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

def check_keyword_validity(output_str, state_data):
    """Python validator script"""
    try:
        import json, re
        match = re.search(r'\{.*?\}', str(output_str), re.DOTALL)
        if not match: return False
        out_dict = json.loads(match.group())
        
        # aggregate
        if isinstance(state_data, list):
            dict1 = json.loads(state_data[0].get("current", "{}"))
            dict2 = json.loads(state_data[1].get("current", "{}")) if len(state_data) > 1 else {}
            
            # add dicts
            true_combined = {**dict1}
            for k, v in dict2.items():
                true_combined[k] = true_combined.get(k, 0) + v
                
            # normalize and compare
            true_norm = {k.lower().strip(): v for k, v in true_combined.items() if v > 0}
            out_norm = {k.lower().strip(): int(v) for k, v in out_dict.items() if int(v) > 0}
            return true_norm == out_norm
            
        # generate
        base_state = state_data
        input_text = base_state.get("sub_text", base_state.get("original", "")).strip()
        if not input_text: return len(out_dict) == 0
        return True
    except:
        return False

def io(all_potential_countries) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(
        operations.Score(1, False, partial(num_errors, all_potential_countries))
    )
    operations_graph.append_operation(operations.GroundTruth(test_keyword_counting))

    return operations_graph


def cot(all_potential_countries) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(
        operations.Score(1, False, partial(num_errors, all_potential_countries))
    )
    operations_graph.append_operation(operations.GroundTruth(test_keyword_counting))

    return operations_graph


def tot(all_potential_countries) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.
    ToT uses a wider tree, where on each level there are more branches.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 20))
    operations_graph.append_operation(
        operations.Score(1, False, partial(num_errors, all_potential_countries))
    )
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(3):
        operations_graph.append_operation(operations.Generate(1, 20))
        operations_graph.append_operation(
            operations.Score(1, False, partial(num_errors, all_potential_countries))
        )
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(test_keyword_counting))

    return operations_graph


def tot2(all_potential_countries) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT2 method.
    ToT2 uses a tree with more levels, but with fewer branches per level.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(
        operations.Score(1, False, partial(num_errors, all_potential_countries))
    )
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(5):
        operations_graph.append_operation(operations.Generate(1, 10))
        operations_graph.append_operation(
            operations.Score(1, False, partial(num_errors, all_potential_countries))
        )
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))

    operations_graph.append_operation(operations.GroundTruth(test_keyword_counting))

    return operations_graph



#helper function for ablations
def create_proactive_got4(all_potential_countries, intervention_enabled=True, use_llm_judge=True, use_moe=True, validator_fn=None):
    operations_graph = operations.GraphOfOperations()

    sub_texts = operations.Generate(1, 1)
    operations_graph.append_operation(sub_texts) 
    
    sub_paragraphs =[]
    for i in range(1, 5):
        paragraph_id = f"Paragraph {i}"
        sub_text = operations.Selector(
            lambda thoughts, lid=paragraph_id: [t for t in thoughts if t.state["part"] == lid]
        )
        sub_text.add_predecessor(sub_texts)
        operations_graph.add_operation(sub_text)
        
        # proactive generate 
        count_sub_text = operations.ProactiveGenerate(1, 10, 0.90, intervention_enabled, use_llm_judge, use_moe, validator_fn)
        count_sub_text.add_predecessor(sub_text)
        operations_graph.add_operation(count_sub_text)
        
        score_sub_text = operations.Score(1, False, partial(num_errors, all_potential_countries))
        score_sub_text.add_predecessor(count_sub_text)
        operations_graph.add_operation(score_sub_text)
        
        keep_best_sub_text = operations.KeepBestN(1, False)
        keep_best_sub_text.add_predecessor(score_sub_text)
        operations_graph.add_operation(keep_best_sub_text)
        sub_paragraphs.append(keep_best_sub_text)

    # pairwise merging
    while len(sub_paragraphs) > 1:
        new_sub_paragraphs =[]
        for i in range(0, len(sub_paragraphs), 2):
            # proactive aggregate
            aggregate = operations.ProactiveAggregate(3, 0.90, intervention_enabled, use_llm_judge, use_moe, validator_fn)
            aggregate.add_predecessor(sub_paragraphs[i])
            aggregate.add_predecessor(sub_paragraphs[i + 1])
            operations_graph.add_operation(aggregate)
            
            val_im_aggregate = operations.ValidateAndImprove(1, True, 3, valid_aggregation)
            val_im_aggregate.add_predecessor(aggregate)
            operations_graph.add_operation(val_im_aggregate)
            
            score_aggregate = operations.Score(1, False, partial(num_errors, all_potential_countries))
            score_aggregate.add_predecessor(val_im_aggregate)
            operations_graph.add_operation(score_aggregate)
            
            keep_best_aggregate = operations.KeepBestN(1, False)
            keep_best_aggregate.add_predecessor(score_aggregate)
            operations_graph.add_operation(keep_best_aggregate)
            new_sub_paragraphs.append(keep_best_aggregate)
        sub_paragraphs = new_sub_paragraphs

    operations_graph.append_operation(operations.GroundTruth(test_keyword_counting))
    return operations_graph

# original
def got4_original(all_potential_countries) -> operations.GraphOfOperations:
    # using standard aggregate/generate for this
    g = operations.GraphOfOperations()
    sub_texts = operations.Generate(1, 1)
    g.append_operation(sub_texts)
    sub_paragraphs =[]
    for i in range(1, 5):
        sub_text = operations.Selector(lambda t, lid=f"Paragraph {i}":[x for x in t if x.state["part"] == lid])
        sub_text.add_predecessor(sub_texts)
        g.add_operation(sub_text)
        count_sub_text = operations.Generate(1, 10)
        count_sub_text.add_predecessor(sub_text)
        g.add_operation(count_sub_text)
        score_sub_text = operations.Score(1, False, partial(num_errors, all_potential_countries))
        score_sub_text.add_predecessor(count_sub_text)
        g.add_operation(score_sub_text)
        keep_best_sub_text = operations.KeepBestN(1, False)
        keep_best_sub_text.add_predecessor(score_sub_text)
        g.add_operation(keep_best_sub_text)
        sub_paragraphs.append(keep_best_sub_text)

    while len(sub_paragraphs) > 1:
        new_sub_paragraphs =[]
        for i in range(0, len(sub_paragraphs), 2):
            aggregate = operations.Aggregate(3)
            aggregate.add_predecessor(sub_paragraphs[i])
            aggregate.add_predecessor(sub_paragraphs[i + 1])
            g.add_operation(aggregate)
            val_im_aggregate = operations.ValidateAndImprove(1, True, 3, valid_aggregation)
            val_im_aggregate.add_predecessor(aggregate)
            g.add_operation(val_im_aggregate)
            score_aggregate = operations.Score(1, False, partial(num_errors, all_potential_countries))
            score_aggregate.add_predecessor(val_im_aggregate)
            g.add_operation(score_aggregate)
            keep_best_aggregate = operations.KeepBestN(1, False)
            keep_best_aggregate.add_predecessor(score_aggregate)
            g.add_operation(keep_best_aggregate)
            new_sub_paragraphs.append(keep_best_aggregate)
        sub_paragraphs = new_sub_paragraphs

    g.append_operation(operations.GroundTruth(test_keyword_counting))
    return g

def got4_2_nodes(all_potential_countries): 
    return create_proactive_got4(all_potential_countries, intervention_enabled=False)

def got4_python_moe(all_potential_countries): 
    return create_proactive_got4(all_potential_countries, use_llm_judge=False, use_moe=True, validator_fn=check_keyword_validity)


def got4_python_no_moe(all_potential_countries): 
    return create_proactive_got4(all_potential_countries, use_llm_judge=False, use_moe=False, validator_fn=check_keyword_validity)

def got4_llm_no_moe(all_potential_countries): 
    return create_proactive_got4(all_potential_countries, use_llm_judge=True, use_moe=False)

def got4_full(all_potential_countries): 
    return create_proactive_got4(all_potential_countries, use_llm_judge=True, use_moe=True)


def got8(all_potential_countries) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT8 method, which splits the text
    into 8 passages.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    sub_texts = operations.Generate(1, 1)
    operations_graph.append_operation(sub_texts)  # generate the sublists
    sub_paragraphs = []
    for i in range(1, 9):
        paragraph_id = f"Paragraph {i}"
        sub_text = operations.Selector(
            lambda thoughts, list_id=paragraph_id: [
                thought for thought in thoughts if thought.state["part"] == list_id
            ]
        )
        sub_text.add_predecessor(sub_texts)
        operations_graph.add_operation(sub_text)
        count_sub_text = operations.Generate(1, 10)
        count_sub_text.add_predecessor(sub_text)
        operations_graph.add_operation(count_sub_text)
        score_sub_text = operations.Score(
            1, False, partial(num_errors, all_potential_countries)
        )
        score_sub_text.add_predecessor(count_sub_text)
        operations_graph.add_operation(score_sub_text)
        keep_best_sub_text = operations.KeepBestN(1, False)
        keep_best_sub_text.add_predecessor(score_sub_text)
        operations_graph.add_operation(keep_best_sub_text)

        sub_paragraphs.append(keep_best_sub_text)

    while len(sub_paragraphs) > 1:
        new_sub_paragraphs = []
        for i in range(0, len(sub_paragraphs), 2):
            aggregate = operations.Aggregate(3)
            aggregate.add_predecessor(sub_paragraphs[i])
            aggregate.add_predecessor(sub_paragraphs[i + 1])
            operations_graph.add_operation(aggregate)
            val_im_aggregate = operations.ValidateAndImprove(
                1, True, 3, valid_aggregation
            )
            val_im_aggregate.add_predecessor(aggregate)
            operations_graph.add_operation(val_im_aggregate)
            score_aggregate = operations.Score(
                1, False, partial(num_errors, all_potential_countries)
            )
            score_aggregate.add_predecessor(val_im_aggregate)
            operations_graph.add_operation(score_aggregate)
            keep_best_aggregate = operations.KeepBestN(1, False)
            keep_best_aggregate.add_predecessor(score_aggregate)
            operations_graph.add_operation(keep_best_aggregate)
            new_sub_paragraphs.append(keep_best_aggregate)
        sub_paragraphs = new_sub_paragraphs

    operations_graph.append_operation(operations.GroundTruth(test_keyword_counting))

    return operations_graph


def gotx(all_potential_countries) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoTx method, where each sentence
    is considered a different passage.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    sub_texts = operations.Generate(1, 1)
    operations_graph.append_operation(sub_texts)  # generate the sublists
    sub_paragraphs = []
    for i in range(1, 33):
        paragraph_id = f"Sentence {i}"
        sub_text = operations.Selector(
            lambda thoughts, list_id=paragraph_id: [
                thought for thought in thoughts if thought.state["part"] == list_id
            ]
        )
        sub_text.add_predecessor(sub_texts)
        operations_graph.add_operation(sub_text)
        count_sub_text = operations.Generate(1, 10)
        count_sub_text.add_predecessor(sub_text)
        operations_graph.add_operation(count_sub_text)
        score_sub_text = operations.Score(
            1, False, partial(num_errors, all_potential_countries)
        )
        score_sub_text.add_predecessor(count_sub_text)
        operations_graph.add_operation(score_sub_text)
        keep_best_sub_text = operations.KeepBestN(1, False)
        keep_best_sub_text.add_predecessor(score_sub_text)
        operations_graph.add_operation(keep_best_sub_text)

        sub_paragraphs.append(keep_best_sub_text)

    while len(sub_paragraphs) > 1:
        new_sub_paragraphs = []
        for i in range(0, len(sub_paragraphs), 2):
            aggregate = operations.Aggregate(3)
            aggregate.add_predecessor(sub_paragraphs[i])
            aggregate.add_predecessor(sub_paragraphs[i + 1])
            operations_graph.add_operation(aggregate)
            val_im_aggregate = operations.ValidateAndImprove(
                1, True, 3, valid_aggregation
            )
            val_im_aggregate.add_predecessor(aggregate)
            operations_graph.add_operation(val_im_aggregate)
            score_aggregate = operations.Score(
                1, False, partial(num_errors, all_potential_countries)
            )
            score_aggregate.add_predecessor(val_im_aggregate)
            operations_graph.add_operation(score_aggregate)
            keep_best_aggregate = operations.KeepBestN(1, False)
            keep_best_aggregate.add_predecessor(score_aggregate)
            operations_graph.add_operation(keep_best_aggregate)
            new_sub_paragraphs.append(keep_best_aggregate)
        sub_paragraphs = new_sub_paragraphs

    operations_graph.append_operation(operations.GroundTruth(test_keyword_counting))

    return operations_graph


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
    added params for temp and config path
    """

    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "countries_simple.csv") # changed to modified dataset
    data = []
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([int(row[0]), row[1], row[2]])

    all_potential_countries = list(
        set([country for row in data for country in row[2][1:-1].split(", ")])
    )

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # renaming output folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # temps
    temp_val = temperature if temperature is not None else 0.6
    temp_str = f"_T{str(temp_val).replace('.', 'p')}"

    extra_info = f"{lm_name}{temp_str}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}_{os.getpid()}" # add pid
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

    
    logging.info("Loading model(once)")
    # Use the passed 'lm_name' and 'temperature', NOT hardcoded string
    lm = language_models.Llama2HF(config_path, model_name=lm_name, temperature=temperature)
    logging.info(" model loaded.")

    for data in selected_data:
        logging.info(f"Running data {data[0]}")
        if budget <= 0.0:
            logging.error("Budget has been depleted, stopping.")
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            if budget <= 0.0:
                logging.error("Budget has been depleted, stopping.")
                break
           
            operations_graph = method(all_potential_countries)
            executor = controller.Controller(
                lm,
                operations_graph,
                KeywordCountingPrompter(),
                KeywordCountingParser(),
                {
                    "original": data[1],
                    "ground_truth": data[2],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )
            
            # timing
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
            
            executor.output_graph(path)
            # time metric 
            if os.path.exists(path):
                with open(path, "r") as f:
                    graph_data = json.load(f)
                
                if isinstance(graph_data, dict):
                    graph_data["execution_time_seconds"] = execution_time
                elif isinstance(graph_data, list):
                    graph_data.append({"execution_time_seconds": execution_time})
                
                with open(path, "w") as f:
                    json.dump(graph_data, f, indent=4)

            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--model_name", type=str, default="qwen2.5-14b", help="Model name key from config.json")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature from config")
    args = parser.parse_args()

    budget = 100
    samples = [item for item in range(0, 100)] 
    
    approaches =[io, cot, got4_original, got4_2_nodes, got4_python_moe, got4_python_no_moe, got4_llm_no_moe, 
    got4_full]
    spent = run(samples, approaches, budget, args.model_name, args.config_path, args.temperature)

    logging.info(f"Spent {spent} out of {budget} budget.")