## This script evaluates LLM responses when directly answering a question and when answering considering
## the logic of the PDDL file. 

import os
import json
import planner # Comment out any function calls within this. 
from openai import OpenAI
from matplotlib import pyplot as plt

######## =================  LLM API calls ====================== ###########
def openai_call(model_name, prompt):
    client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    output = client_oai.chat.completions.create(model=model_name, 
                                       messages=[{"role": "user", "content": prompt}],
                                       stream=False
                                    )
    output_content = output.choices[0].message.content
    return output_content

def deepinfra_call(model_name, prompt):
    client_deepinfra = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
    output = client_deepinfra.chat.completions.create(model=model_name, 
                                       messages=[{"role": "user", "content": prompt}],
                                       stream=False
                                    )
    output_content = output.choices[0].message.content
    return output_content

def deepseek_call(model_name, prompt):
    client_deepseek = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
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
                                                  domain_path,
                                                  problem_path,
                                                  current_plan, 
                                                  scenario_id, 
                                                  interaction_id,
                                                  scenario_domain_and_problem_data):
    
    #### Step 1: Generate the PDDL prompts ======================= #########
    context = scenario_domain_and_problem_data[scenario_id]["Context"]
    question = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["problem_data"]
    answer = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["answer_data"]
                
    with open(domain_path, 'r') as file_domain:
        pddl_domain = file_domain.readlines()
    
    with open(problem_path, 'r') as file_problem:
        pddl_problem = file_problem.readlines()

    direct_prompt = f"""
        I want you to think step by step carefully and answer questions about autonomous vehicle test scenarios. Each question
        has a descriptive answer.

        Here is some contextual information about one specific scenario:
        {context}

        Now, given all of this information, think step by step and answer the following question:
        {question}

        Write a short answer only. Think step by step carefully and show your reasoning and how you reached a solution. 
        """
    
    #create a set of variables called
    #direct_prompt_cot_2shot, direct_prompt_cot_4shot, direct_prompt_cot_6shot,
    #direct_prompt_cot_8shot (#shot refers to number of interactions(qa pair) within the same scenario ID/context)
    #(chain of thought) before asking the question
    #provide examples of q and a pairs. obtain the pairs
    #from the results of the parse script. scenario using
    #for examples should not be used for evaluation.
    # <context from parsed data>
    # <q and a from one interaction>
    # <q and a from another interaction>
    #how to create these examples: 
    #direct_prompt = f"""
    #     I want you to answer some questions about an autonomous vehicle test scenario. Here are some examples for some scenarios:
    #     <context from parsed data>
    #     <q and a from one interaction>
    #     <q and a from another interaction>
    #     Here is some information about an autonomous vehicle scenario:
    #     {context}

    #     Answer the following question:
    #     {question}

    #     Think step by step. Show your reasoning and answer the question. 
        
    #     """

    pddl_prompt = f"""
        I want you to think step by step carefully and answer questions about autonomous vehicle test scenarios. Each question
        has a descriptive answer.

        Here is some contextual information about one specific scenario:
        {context}
        
        Here is some generated PDDL (Planning Domain Definition Language) domain data corresponding to this contextual information. This information
        is to give you a more specific, explicit sense of the logic behind some of the car driving behaviors
        in this scenario:
        {pddl_domain}

        Each scenario consists of interactions between vehicles and intentionality regarding driving behavior intent of vehicles.
        Regarding one such interaction, here is a PDDL problem statement generated. Connect it carefully with the 
        domain information given above.

        Here is the PDDL problem data:
        {pddl_problem}

        Since this data is in the PDDL format, I ran this through a planner which carried out 
        breadth first search to try and answer the PDDL problem data above in light of the PDDL domain
        data and got the following result:
        {current_plan}

        Now, given all of this information, think step by step and answer the following question:
        {question}

        Write a short answer only. Think step by step carefully and show your reasoning and how you reached a solution.
        
        """
    #### Step 2: Generate the model grades and add them to the dictionary
    
    for model_family in model_dictionary.keys():
        if model_family=="openai_models":
            for model_name in model_dictionary[model_family].keys():
                print("Model name is {}".format(model_name))
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=openai_call(model_name=model_name, prompt=pddl_prompt))
                grading_output = eval(deepseek_call(model_name="deepseek-chat", prompt=grading_prompt))
                print(grading_output)
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_modelname", grading_output
                    )
                avg_score = (int(grading_output["Correctness score"]) + int(grading_output["Faithfulness score"]))/2
                
                existing_grades[scenario_id][interaction_id][model_family+"_"+model_name+"_modelname"].setdefault("problem_score_avg", (str(avg_score)))
                model_dictionary[model_family][model_name].append((str(avg_score)))
        elif model_family=="deepinfra_models":
            for model_name in model_dictionary[model_family].keys():
                print("Model name is {}".format(model_name))
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=deepinfra_call(model_name=model_name, prompt=pddl_prompt)) #when creating new var replace pddl prompt w my var name (ie 4shot)
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_with_plan", grading_output #replace _with_plan with _for_(insert variable name)
                    )
                existing_grades[scenario_id][interaction_id].setdefault("problem_score_avg", ((grading_output["Correctness score"] + grading_output["Faithfulness score"])/2))
        print("Retrieving grades")


def pddl_response_and_answer_questions(domain_path, 
                                       problem_path, 
                                       current_plan, 
                                       eval_folder,
                                       scenario_domain_and_problem_data,
                                       existing_grades,
                                       model_dictionary):
    # Parse through the preprocessed json data contained in parsed_womdr_data/
    print(f"Data is {scenario_domain_and_problem_data}")
    for scenario_id in scenario_domain_and_problem_data.keys():
        existing_grades.setdefault(scenario_id, {})
        for interaction_id in scenario_domain_and_problem_data[scenario_id]["Interactions"].keys():
            print(f"Interaction considered is {interaction_id}")
            print(f"Domain path is {domain_path}")
            print(f"Problem path is {problem_path}")
            existing_grades[scenario_id].setdefault(interaction_id, {})
            if (scenario_id in domain_path) and (interaction_id in problem_path):
                print("Scenario ID that matches is {}".format(scenario_id))
                print("Interaction ID that matches is {}".format(interaction_id))
                eval_complete_path = eval_folder+"LLM_eval_"+interaction_id+".json"
                print("Evaluation file is {}".format(eval_complete_path))
                
                ##### ===================== Automatic model evaluation with LLM grades on outputs ============== #########
                existing_grades = grade_openai_deepinfra_models_one_interaction(model_dictionary=model_dictionary, 
                                                            existing_grades=existing_grades,
                                                            domain_path=domain_path,
                                                            problem_path=problem_path,
                                                            current_plan=current_plan, 
                                                            scenario_id=scenario_id, 
                                                            interaction_id=interaction_id,
                                                            scenario_domain_and_problem_data=scenario_domain_and_problem_data)
               
                #Ensure that this json file by the name grades/deepseek_grades.json exists first.
                print("Editing grades")
                with open("grades/deepseek_grades.json", 'w') as grade_file:
                    with open(eval_complete_path, 'r') as eval_file:
                        data = json.load(eval_file)
                        existing_grades[scenario_id][interaction_id].setdefault("LLM_eval_problem_grade", data["Problem coverage"]["Grade"])
                        existing_grades[scenario_id][interaction_id].setdefault("LLM_eval_context_word_count", data["average_context_sentence_word_count"])
                        qa_interaction_score = existing_grades[scenario_id][interaction_id]["problem_score_avg"]*existing_grades[scenario_id][interaction_id]["LLM_eval_problem_grade"]
                        existing_grades[scenario_id][interaction_id].setdefault("qa_interaction_score", qa_interaction_score)  
                    print("Existing grades is given by {}".format(existing_grades))
                    json.dump(existing_grades, grade_file, indent=4)
                    grade_file.close()

def run_evaluations():
    # Recover the PDDL domain file, PDDL problem file for a particular scenario and plan file. 
    domain_folder_list = os.listdir('apla-planner/generated_pddls_deepseek/dataset/domains')
    problem_folder_list = os.listdir('apla-planner/generated_pddls_deepseek/dataset/problems')
    scenario_domain_and_problem_data = planner.retrieve_womdr_domain_problem_data()

    model_dictionary = {
    "openai_models": {
        "o3-mini": []
        },
    "deepinfra_models": {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": []
    } 
    }

    # Generate two lists - domain file list and problem file list for a single scenario
    # Reuse code in terms of classes and functions and 

    model_outputs = {}
    existing_grades = {}


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
            if ".pdd"==problem_file_name[-5:-1]:
                pddlproblem_file_name = problem_file_name
                print("PDDL problem file name is {}".format(pddlproblem_file_name))
                print("problem file name is {}".format(pddlproblem_file_name))
                print("plan file name is {}".format(plan_file_name))
                print("Scenario folder is {}".format(scenario_folder))

                problem_full_path = "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_folder+"/"+pddlproblem_file_name
                domain_full_path = "apla-planner/generated_pddls_deepseek/dataset/domains/"+scenario_folder+"/"+domains_within_scenario[0]
                planfile_full_path = "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_folder+"/"+pddlproblem_file_name+"_"+plan_file_name
                eval_folder = "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_folder+"/"    

                with open(planfile_full_path, 'r') as plan_file:
                    plan_data = json.load(plan_file)
                    try:
                        current_problem_plan = plan_data[pddlproblem_file_name]
                        print("Current problem plan is {}".format(current_problem_plan))
                        existing_grades = pddl_response_and_answer_questions(domain_path=domain_full_path, 
                                                        problem_path=problem_full_path,
                                                    current_plan=current_problem_plan, 
                                                    eval_folder=eval_folder,
                                                    scenario_domain_and_problem_data=scenario_domain_and_problem_data,
                                                    model_dictionary=model_dictionary,
                                                    existing_grades=existing_grades)
                    except:
                        continue
                
            else: pddlproblem_file_name = ""
    
    # After everything is done, plot a single bar chart per model output
    for model_provider in model_dictionary.keys():
        for model in model_dictionary[model_provider].keys():
            plt.bar([i for i in range(len(model_dictionary[model_provider][model]))], [float(model_dictionary[model_provider][model][i]) for i in range(len(model_dictionary[model_provider][model]))])
            plt.title(f"{model}")
            plt.xlabel("Interaction number (across all scenarios)")
            plt.ylabel("Average correctness/faithfulness scores")
            plt.show()
