## This script evaluates LLM responses when directly answering a question and when answering considering
## the logic of the PDDL file. 

import os
import json
import matplotlib.pyplot as plt
import planner # Comment out any function calls within this. 
from openai import OpenAI

########### ============  Global initializations ====================== ##########
parsed_file_list = os.listdir("parsed_womdr_data/")
client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client_deepseek = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
client_deepinfra = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
scenario_domain_and_problem_data = planner.retrieve_womdr_domain_problem_data()

model_dictionary = {
   "openai_models": ["gpt-4o-mini", "o3-mini"],
   "deepinfra_models": ["meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        "microsoft/phi-4",
                        "Qwen/Qwen2.5-7B-Instruct"
                        ] 
}

# Generate two lists - domain file list and problem file list for a single scenario
# Reuse code in terms of classes and functions and 

model_outputs = {}
existing_grades = {}
qa_scores = []

######## =================  LLM API calls ====================== ###########
def openai_call(model_name, prompt):
    output = client_oai.chat.completions.create(model=model_name, 
                                       messages=[{"role": "user", "content": prompt}],
                                       stream=False
                                    )
    output_content = output.choices[0].message.content
    return output_content

def deepinfra_call(model_name, prompt):
    output = client_deepinfra.chat.completions.create(model=model_name, 
                                       messages=[{"role": "user", "content": prompt}],
                                       stream=False
                                    )
    output_content = output.choices[0].message.content
    return output_content

def deepseek_call(model_name, prompt):
    output = client_deepseek.chat.completions.create(model=model_name, 
                                       messages=[{"role": "user", "content": prompt}],
                                       stream=False
                                    )
    output_content = output.choices[0].message.content
    return output_content

################# ============= Grading prompts ================== ###############
def prepare_grading_prompt(context, question, answer, model_output):
    grading_prompt = f"""
        Here is some context about the test scenario:
        {context}

        This question was asked with regards to this context:
        {question}

        This is the ground truth answer:
        {answer}

        This was the attempt by an AI for this question
        {model_output}

        Grade this answer on the following aspects:
        1. The correctness of the AI answer with respect to the ground truth answer. Give it a score between 1 to 10.
        Explain why this score was given by you in detail.
        2. The faithfulness of the reasoning. Are the conclusions drawn in the answer given by the AI consistent with its reasoning? Here, give it a score between 1 to 10.
        Explain why this score was given by you in detail.

        Format the answer in a python dictionary format like this.
        <open curly bracket>:
        "Correctness score": "<Only enter the score number here>",
        "Correctness explanation": "<Write your explanation here>",
        "Faithfulness score": "<Only enter the score number here>",
        "Faithfulness explanation": "<Write your explanation here>",
        <close curly bracket>
        
        Don't write anything else. Nothing else, nothing else, nothing else. 
        Please only write it in the format requested.
        """
    return grading_prompt

############### =============== Evaluating Interactions ================ ##############
def grade_openai_deepinfra_models_one_interaction(model_dictionary, 
                                                  existing_grades,
                                                  scenario_id, 
                                                  interaction_id):
    
    #### Step 1: Generate the PDDL prompts ======================= #########
    context = scenario_domain_and_problem_data[scenario_id]["Context"]
    question = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["problem_data"]
    answer = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["answer_data"]

    direct_prompt = f"""
        Here is some information about an autonomous vehicle scenario:
        {context}

        Answer the following question:
        {question}

        Think step by step. Show your reasoning and answer the question. 
        
        """
    #### Step 2: Generate the model grades and add them to the dictionary
    
    for model_family in model_dictionary.keys():
        if model_family=="openai_models":
            for model_name in model_dictionary[model_family]:
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=openai_call(model_name=model_name, prompt=direct_prompt))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_with_plan", grading_output
                    )
                avg_score = (int(grading_output["Correctness score"]) + int(grading_output["Faithfulness score"]))/2
                existing_grades[scenario_id][interaction_id].setdefault("problem_score_avg", (str(avg_score)))
                qa_scores.append(avg_score)
        elif model_family=="deepinfra_models":
            for model_name in model_dictionary[model_family]:
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=deepinfra_call(model_name=model_name, prompt=direct_prompt))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_with_plan", grading_output
                    )
                avg_score = (int(grading_output["Correctness score"]) + int(grading_output["Faithfulness score"]))/2
                existing_grades[scenario_id][interaction_id].setdefault("problem_score_avg", (str(avg_score)))
                qa_scores.append(avg_score)
    


def pddl_response_and_answer_questions():
    # Parse through the preprocessed json data contained in parsed_womdr_data/
    for scenario_id in scenario_domain_and_problem_data.keys():
        existing_grades.setdefault(scenario_id, {})
        for interaction_id in scenario_domain_and_problem_data[scenario_id]["Interactions"].keys():
            existing_grades[scenario_id].setdefault(interaction_id, {})
              
            ##### ===================== Automatic model evaluation with LLM grades on outputs ============== #########
            grade_openai_deepinfra_models_one_interaction(model_dictionary=model_dictionary, 
                                                        existing_grades=existing_grades, 
                                                        scenario_id=scenario_id, 
                                                        interaction_id=interaction_id)
            
            #Ensure that this json file by the name grades/deepseek_grades.json exists first.
            with open("grades/direct/deepseek_grades.json", 'w') as grade_file:
                print("Existing grades is given by {}".format(existing_grades))
                json.dump(existing_grades, grade_file, indent=4)
                grade_file.close()

def main():
    pddl_response_and_answer_questions()
    print("For this exp run, the final qa scores are {}".format(qa_scores))
    
    # plt.bar([i for i in range(len(exp_run_qa_scores))], exp_run_qa_scores)
    # plt.show()
main()