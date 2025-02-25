from openai import OpenAI
import sys
import subprocess
import os

class ProvidedLLM():
    def __init__(self):
        self.client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.client_deepinfra = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
        self.client_dsapi = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

        # The following are model names for DS models provided via their own API service.
        self.ds_v3_dsapi = "deepseek-chat"
        self.ds_r1_dsapi = "deepseek-reasoner"

        # The following are model names for Large DeepInfra provided models
        self.ds_v3 = "deepseek-ai/DeepSeek-V3"
        self.ds_r1 = "deepseek-ai/DeepSeek-R1" # This model thinks. Cannot use for json output
        self.llama_33_70b = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        self.llama_31_405b = "meta-llama/Meta-Llama-3.1-405B-Instruct"
        self.qw_25_72b = "Qwen/Qwen2.5-72B-Instruct"
        self.ds_distil_llama_70b = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" # This model thinks. Cannot use for json output

        # The following are model names for small DeepInfra provided models
        self.gemma_2 = "google/gemma-2-9b-it"
        self.llama_31_8b = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.qw_25_7b = "Qwen/Qwen2.5-7B-Instruct"
        self.phi_4 = "microsoft/phi-4"

        # The following are the small model names for models provided via the OpenAI API service
        self.gpt_4o_mini = "gpt-4o-mini"
        self.o3_mini = "o3-mini"

        self.model_dictionary = {
                                    "openai_models": [self.gpt_4o_mini, self.o3_mini],
                                    "deepinfra_models": [self.llama_31_8b, 
                                                         self.phi_4,
                                                         self.qw_25_7b
                                                        ] 
                                }