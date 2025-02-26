## This script evaluates LLM responses when directly answering a question and when answering considering
## the logic of the PDDL file. 

import os
import json
import planner # Comment out any function calls within this. 
from openai import OpenAI
from matplotlib import pyplot as plt

########### ============  Global initializations ====================== ##########
domain_folder_list = os.listdir('apla-planner/generated_pddls_deepseek/dataset/domains')
problem_folder_list = os.listdir('apla-planner/generated_pddls_deepseek/dataset/problems')
client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#client_deepseek = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
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
exp_run_qa_scores = []

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

#def deepseek_call(model_name, prompt):
#    output = client_deepseek.chat.completions.create(model=model_name, 
#                                       messages=[{"role": "user", "content": prompt}],
#                                       stream=False
#                                    )
#    output_content = output.choices[0].message.content
#    return output_content

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
                                                  domain_path,
                                                  problem_path,
                                                  current_plan, 
                                                  scenario_id, 
                                                  interaction_id):
    
    #### Step 1: Generate the PDDL prompts ======================= #########
    context = scenario_domain_and_problem_data[scenario_id]["Context"]
    question = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["problem_data"]
    answer = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["answer_data"]
                
    with open(domain_path, 'r') as file_domain:
        pddl_domain = file_domain.readlines()
    
    with open(problem_path, 'r') as file_problem:
        pddl_problem = file_problem.readlines()

    direct_prompt = f"""
        Here is some information about an autonomous vehicle scenario:
        {context}

        Answer the following question:
        {question}

        Think step by step. Show your reasoning and answer the question. 
        
        """

    pddl_prompt = f"""
        Here is some context about the test scenario:
        {context}
        
        Here is some PDDL domain data:
        {pddl_domain}

        Here is the PDDL problem statement:
        {pddl_problem}

        I ran this through a planner and got the following result:
        {current_plan}

        Think step by step and answer the following question:
        {question}

        Write a short 2 sentence answer only. Show your reasoning.
        
        """
    #### Step 2: Generate the model grades and add them to the dictionary
    
    for model_family in model_dictionary.keys():
        if model_family=="openai_models":
            for model_name in model_dictionary[model_family]:
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=openai_call(model_name=model_name, prompt=pddl_prompt))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_with_plan", grading_output
                    )
                existing_grades[scenario_id][interaction_id].setdefault("problem_score_avg", ((grading_output["Correctness score"] + grading_output["Faithfulness score"])/2))
        elif model_family=="deepinfra_models":
            for model_name in model_dictionary[model_family]:
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=deepinfra_call(model_name=model_name, prompt=pddl_prompt))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_with_plan", grading_output
                    )
                existing_grades[scenario_id][interaction_id].setdefault("problem_score_avg", ((grading_output["Correctness score"] + grading_output["Faithfulness score"])/2))
    


def pddl_response_and_answer_questions(domain_path, problem_path, current_plan, eval_folder):
    # Parse through the preprocessed json data contained in parsed_womdr_data/
    for scenario_id in scenario_domain_and_problem_data.keys():
        existing_grades.setdefault(scenario_id, {})
        for interaction_id in scenario_domain_and_problem_data[scenario_id]["Interactions"].keys():
            existing_grades[scenario_id].setdefault(interaction_id, {})
            if (scenario_id in domain_path) and (interaction_id in problem_path):
                print("Scenario ID that matches is {}".format(scenario_id))
                print("Interaction ID that matches is {}".format(interaction_id))
                eval_complete_path = eval_folder+"LLM_eval_"+interaction_id+".json"
                print("Evaluation file is {}".format(eval_complete_path))
                
                ##### ===================== Automatic model evaluation with LLM grades on outputs ============== #########
                grade_openai_deepinfra_models_one_interaction(model_dictionary=model_dictionary, 
                                                            existing_grades=existing_grades,
                                                            domain_path=domain_path,
                                                            problem_path=problem_path,
                                                            current_plan=current_plan, 
                                                            scenario_id=scenario_id, 
                                                            interaction_id=interaction_id)
               
                #Ensure that this json file by the name grades/deepseek_grades.json exists first.
                with open("grades/deepseek_grades.json", 'w') as grade_file:
                    with open(eval_complete_path, 'r') as eval_file:
                        data = json.load(eval_file)
                        existing_grades[scenario_id][interaction_id].setdefault("LLM_eval_problem_grade", data["Problem coverage"]["Grade"])
                        existing_grades[scenario_id][interaction_id].setdefault("LLM_eval_context_word_count", data["average_context_sentence_word_count"])
                        qa_interaction_score = existing_grades[scenario_id][interaction_id]["problem_score_avg"]*existing_grades[scenario_id][interaction_id]["LLM_eval_problem_grade"]
                        existing_grades[scenario_id][interaction_id].setdefault("qa_interaction_score", qa_interaction_score)
                        exp_run_qa_scores.append(qa_interaction_score)  
                    print("Existing grades is given by {}".format(existing_grades))
                    json.dump(existing_grades, grade_file, indent=4)
                    grade_file.close()

def main():
    # Recover the PDDL domain file, PDDL problem file for a particular scenario and plan file. 
    for scenario_folder in domain_folder_list:
        #Scores for multiple problems (where each problem corresponds to one interaction) within one scenario
        #These grades add up to help us evaluate across all scenarios
        
        scenario_folder_domains_complete_path = 'apla-planner/generated_pddls_deepseek/dataset/domains/'+scenario_folder
        scenario_folder_problems_complete_path = 'apla-planner/generated_pddls_deepseek/dataset/problems/'+scenario_folder
        domains_within_scenario = os.listdir(scenario_folder_domains_complete_path)
        problems_within_scenario = os.listdir(scenario_folder_problems_complete_path)

        # We will traverse the problem list since there will be only one domain per scenario
        plans_for_one_scenario = {}
        problem_coverage_scores = []
        problem_initial_state_sizes = []
        print("Scenario ID is {}".format(scenario_folder))
        pddlproblem_file_name = ""
        plan_file_name = "plan_set.json"
        
        for problem_file_name in problems_within_scenario:
            # If PDDL problem file has been found, then open the plan file and find out the evaluations.
            # There should a plan file by the name of plan_set.json in each problem folder. 
            # Run this after the PDDL problem generation and the plan generation has been done. 
            if ".pddl" in problem_file_name:
                pddlproblem_file_name = problem_file_name
                print("PDDL problem file name is {}".format(pddlproblem_file_name))
                print("problem file name is {}".format(pddlproblem_file_name))
                print("plan file name is {}".format(plan_file_name))
                print("Scenario folder is {}".format(scenario_folder))

                problem_full_path = "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_folder+"/"+pddlproblem_file_name
                domain_full_path = "apla-planner/generated_pddls_deepseek/dataset/domains/"+scenario_folder+"/"+domains_within_scenario[0]
                planfile_full_path = "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_folder+"/"+plan_file_name
                eval_folder = "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_folder+"/"    

                with open(planfile_full_path, 'r') as plan_file:
                    plan_data = json.load(plan_file)
                    try:
                        current_problem_plan = plan_data[pddlproblem_file_name]
                        print("Current problem plan is {}".format(current_problem_plan))
                        pddl_response_and_answer_questions(domain_path=domain_full_path, 
                                                        problem_path=problem_full_path,
                                                    current_plan=current_problem_plan, eval_folder=eval_folder)
                    except:
                        continue
                
            else: pddlproblem_file_name = ""
    
    print("For this exp run, the final qa scores are {}".format(exp_run_qa_scores))
    
    plt.bar([i for i in range(len(exp_run_qa_scores))], exp_run_qa_scores)
    plt.show()
