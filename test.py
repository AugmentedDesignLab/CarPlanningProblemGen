# Removal of think tags in output. Test on Deepseek R1 Distill Llama 70B
from client_model_setup import ProvidedLLM
import re

llm = ProvidedLLM()

# Setup the client and the model
client = llm.client_deepinfra
model = llm.ds_distil_llama_70b
model_r1_deepinfra = llm.ds_r1

client_ds = llm.client_dsapi
model_r1 = llm.ds_r1_dsapi

prompt = """
Hi do you have advice for grad school in computer science? Format your response in the following manner:
<open curly bracket>
"advice":
<open curly bracket>
"<advice area>": "<description of advice>",
...
...
<close curly bracket>

Your response needs to strictly be in this format. Please do not write anything else outside this format.
"""
output, thoughts = llm.thinking_llm_call(client=client, model=model_r1_deepinfra, prompt=prompt)

# separated_string = re.split(r"(</think>)", output)
# separated_string_thoughts = re.split(r"(<think>)", separated_string[0])
# separated_string_output = separated_string[1]
# separated_string_thoughts = separated_string_thoughts[1]

print("\n The thoughts are\n")
print(thoughts)
print("\nThe actual output is\n")
print(output)
