# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Ales Kubicek

import os
import torch
from typing import List, Dict, Union
from .abstract_language_model import AbstractLanguageModel


class Llama2HF(AbstractLanguageModel):
    """
    An interface to use LLaMA 2 models through the HuggingFace library.
    """

    def __init__(
    self, config_path: str = "", model_name: str = "llama7b-hf", cache: bool = False, temperature: float = None
    ) -> None:
        """
        Initialize an instance of the Llama2HF class with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to an empty string.
        :type config_path: str
        :param model_name: Specifies the name of the LLaMA model variant. Defaults to "llama7b-hf".
                           Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # Detailed id of the used model.
        self.model_id: str = self.config["model_id"]
        # Costs for 1000 tokens.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # adding temperature as an argument instead of pulling from config file
        if temperature is not None:
            self.temperature = temperature
        # Top K sampling.
        self.top_k: int = self.config["top_k"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]

        # Important: must be done before importing transformers
        os.environ["TRANSFORMERS_CACHE"] = self.config["cache_dir"]
        import transformers

        # change from just meta llama models to add functionality for qwen 
        hf_model_id = self.model_id 
        model_config = transformers.AutoConfig.from_pretrained(hf_model_id)
        # no longer using bnb bc there were issues with loading it on neuronic
        '''
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        '''

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            config=model_config,
            torch_dtype=torch.float16,   # using 16 bit now
            device_map={"": 0},          # force to gpu
            low_cpu_mem_usage=True,      # important to save cpu ram 
        )
        self.model.eval()
        torch.no_grad()

        self.generate_text = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer, task="text-generation"
        )

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        Query the model for responses (using a reformatted proper chat template).

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the model.
        :rtype: List[Dict]
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        #new chat template logic w <|im_start|> and proper instruction tuned formatting 
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format."},
            {"role": "user", "content": query}
        ]
        
        # did NOT work (failed experiment) -- asking for just strict data significantly lowered output accuracy
        '''
        messages =[
            {"role": "system", "content": "You are a strict data processor. You NEVER converse, NEVER apologize, and NEVER explain your reasoning. You only output raw, formatted data exactly as requested."},
            {"role": "user", "content": query}
        ]
        '''
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sequences = []
        for _ in range(num_responses):
            sequences.extend(
                self.generate_text(
                    formatted_prompt,
                    do_sample=True,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.max_tokens,
                    return_full_text=False  # returns only new text now
                )
            )
        # simplified response parsing
        response = [
            {"generated_text": sequence["generated_text"].strip()}
            for sequence in sequences
        ]
        if self.cache:
            self.response_cache[query] = response
        return response

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_responses: The response list of dictionaries generated from the `query` method.
        :type query_responses: List[Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        return [query_response["generated_text"] for query_response in query_responses]
