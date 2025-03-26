import planner
from client_model_setup import ProvidedLLM

print("Running the PDDL file generation...\nThe domains and problem files will get saved in the apla-planner/generated_pddls_deepseek path within the domains and problems folder.......")

provided_llm = ProvidedLLM()
client = provided_llm.client_oai
model = provided_llm.gpt_45
planner.generate_pddl_with_syntax_check(client, model)
print("PDDL problem generation has been completed!\n")