## This script evaluates LLM responses when directly answering a question and when answering considering
## the logic of the PDDL file. 

import os
import json
import matplotlib.pyplot as plt
import planner # Comment out any function calls within this. 
from openai import OpenAI

domain_folder_list = os.listdir('apla-planner/generated_pddls_deepseek/dataset/domains')
problem_folder_list = os.listdir('apla-planner/generated_pddls_deepseek/dataset/problems')

# Generate two lists - domain file list and problem file list for a single scenario
# Reuse code in terms of classes and functions and 

existing_grades = {}

total_interactions_evaluated = 0
average_correctness_score_one_scenario = 0
incorrect_count_one_scenario = 0
partially_correct_count_one_scenario = 0
correct_count_one_scenario = 0

def pddl_response_and_answer_questions(domain_path, problem_path, current_plan, eval_folder):
    client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    client_deepseek = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    scenario_domain_and_problem_data = planner.retrieve_womdr_domain_problem_data()

    # Process the average grades within these for loops themselves. 

    # Parse through the preprocessed json data contained in parsed_womdr_data/
    for scenario_id in scenario_domain_and_problem_data.keys():
        for interaction_id in scenario_domain_and_problem_data[scenario_id]["Interactions"].keys():
            if (scenario_id in domain_path) and (interaction_id in problem_path):
                print("Scenario ID that matches is {}".format(scenario_id))
                print("Interaction ID that matches is {}".format(interaction_id))
                eval_complete_path = eval_folder+"LLM_eval_"+interaction_id+".json"
                print("Evaluation file is {}".format(eval_complete_path))
                
                context = scenario_domain_and_problem_data[scenario_id]["Context"]
                question = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["problem_data"]
                answer = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["answer_data"]
                
                # Using the OpenAI API and the 
                response_gpt_4o_mini_direct = client_oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": f"""
                    Here is some information about an autonomous vehicle scenario:
                    {scenario_domain_and_problem_data[scenario_id]["Context"]}

                    Answer the following question:
                    {question}

                    Think step by step. Show your reasoning and answer the question. 
                    
                    """},
                    ],
                    stream=False
                )   

                with open(domain_path, 'r') as file_domain:
                    pddl_domain = file_domain.readlines()
                
                with open(problem_path, 'r') as file_problem:
                    pddl_problem = file_problem.readlines()

                response_gpt_4o_mini_with_plan = client_oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": f"""
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
                    
                    """},
                    ],
                    stream=False
                )

                # Deepseek V3 LLM as a judge
                response_deepseek_score_4o_mini_with_plan = client_deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": f"""
                    Here is some context about the test scenario:
                    {context}

                    This question was asked with regards to this context:
                    {question}

                    This is the ground truth answer:
                    {answer}

                    This was the attempt by an AI for this question
                    {response_gpt_4o_mini_with_plan.choices[0].message.content}

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
                    """},
                    ],
                    stream=False
                )

                # Deepseek V3 LLM as a judge
                response_deepseek_score_4o_mini_direct = client_deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": f"""
                    Here is some context about the test scenario:
                    {context}

                    This question was asked with regards to this context:
                    {question}

                    This is the ground truth answer:
                    {answer}

                    This was the attempt by an AI for this question
                    {response_gpt_4o_mini_direct.choices[0].message.content}

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
                    """},
                    ],
                    stream=False
                )

                #Ensure that this json file by the name grades/deepseek_grades.json exists first.
                with open("grades/deepseek_grades.json", 'w') as grade_file:
                    output_dictionary_gpt_4o_mini_with_plan = eval(response_deepseek_score_4o_mini_with_plan.choices[0].message.content)
                    output_dictionary_gpt_4o_mini_direct = eval(response_deepseek_score_4o_mini_direct.choices[0].message.content)
                    
                    # Deepseek LLM as a judge responses to all of GPT-4o-mini responses will be stored 
                    # in the existing_grades dictionary. 
                    existing_grades.setdefault(scenario_id, {})
                    existing_grades[scenario_id].setdefault(interaction_id, {})
                    existing_grades[scenario_id][interaction_id].setdefault("GPT_4o_mini_with_plan_grades", output_dictionary_gpt_4o_mini_with_plan)
                    existing_grades[scenario_id][interaction_id].setdefault("GPT_4o_mini_direct_grades", output_dictionary_gpt_4o_mini_direct)
                    with open(eval_complete_path, 'r') as eval_file:
                        data = json.load(eval_file)
                        existing_grades[scenario_id][interaction_id].setdefault("LLM_eval_problem_grade", data["Problem coverage"]["Grade"])
                        existing_grades[scenario_id][interaction_id].setdefault("LLM_eval_context_word_count", data["context_word_count"])
                    print("Existing grades is given by {}".format(existing_grades))
                    json.dump(existing_grades, grade_file, indent=4)
                    grade_file.close()
                print("Deepseek score grading response\n")
                print(response_deepseek_score_4o_mini_with_plan.choices[0].message.content)


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
