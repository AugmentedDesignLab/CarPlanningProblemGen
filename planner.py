
import sys
import subprocess
import os
import json
from pathlib import Path
import pddlpy
from openai import OpenAI
from tqdm import tqdm
from pddl_gen import PDDLGen
from client_model_setup import ProvidedLLM
import shutil

provided_llm = ProvidedLLM()

def retrieve_womdr_domain_problem_data():

    parsed_womdr_files = os.listdir("parsed_womdr_data/")
    scenario_domain_problem_data = {}

    for i in parsed_womdr_files:
        with open("parsed_womdr_data/"+i, 'r') as scenario_file:
            scenario_data = json.load(scenario_file) 
            for key in scenario_data.keys():
                # Indices here have been planned based on the Waymo Reasoning dataset files
                scenario_domain_problem_data.setdefault(i[:-5], {
                    "Context": ""
                })
                scenario_domain_problem_data[i[:-5]]["Context"] = scenario_data[key]["Context"]
                scenario_domain_problem_data[i[:-5]]["Word Count"] = scenario_data[key]["Word Count"]
                print(f"""
                      number of interactions in this scenario are {scenario_data[key]["Interactions"].keys()}
                      """)
                for interaction_key in scenario_data[key]["Interactions"].keys():
                    scenario_domain_problem_data[i[:-5]].setdefault("Interactions", {})
                    scenario_domain_problem_data[i[:-5]]["Interactions"].setdefault(interaction_key, {
                    "problem_data": "",
                    "answer_data": ""
                    }) 
                    scenario_domain_problem_data[i[:-5]]["Interactions"][interaction_key]["problem_data"] = scenario_data[key]["Interactions"][interaction_key]["reference_question"]
                    scenario_domain_problem_data[i[:-5]]["Interactions"][interaction_key]["answer_data"] = scenario_data[key]["Interactions"][interaction_key]["reference_answer"]
       
    return scenario_domain_problem_data

# Client and model type are resolved in the function call itself and remain fixed throughout
def generate_pddl_with_syntax_check(client, model):
    scenario_domain_problem_data = retrieve_womdr_domain_problem_data()  
    for id in tqdm(scenario_domain_problem_data.keys()):
        val_error_detected = False
        blocklist_ids = []
        with open("apla-planner/generated_pddls_deepseek/dataset/blocklist.json", 'r') as block_file:
            blocklist_ids = json.load(block_file)
        if blocklist_ids.count(id)>0: continue # Scenario ID detected in blocklist, continue        
        
        # Client and model fixed for all PDDL generation. 
        pddl_gen = PDDLGen(client=client, model=model)
        print(f"\nDomain generation, generating domain file for scenario id {id}....\n")
        scenario_domain_problem_data_context = scenario_domain_problem_data[id]["Context"]
        domain, attempted_overwrite, generated_actions = pddl_gen.generate_pddl_domain(scenario_domain_problem_data_context=scenario_domain_problem_data_context,
                                      scenario_id=id)
        print(domain)
        print(generated_actions)
        if attempted_overwrite==True: continue # Attempted PDDL domain overwrite, move on to the next scenario. 

        # Given one domain file based on a context, generate multiple problem files.
        for interaction_id in tqdm(scenario_domain_problem_data[id]["Interactions"].keys()):    
            if val_error_detected: break
            print(f"\nProblem generation, generating problem file for interaction {interaction_id}....\n")
            scenario_domain_problem_problem_data = scenario_domain_problem_data[id]["Interactions"][interaction_id]["problem_data"]
            problem = pddl_gen.generate_pddl_problem(domain=domain,
                                           scenario_domain_problem_data_context=scenario_domain_problem_data_context,
                                           generated_actions=generated_actions,
                                           scenario_problem_data=scenario_domain_problem_problem_data,
                                           scenario_id=id,
                                           interaction_id=interaction_id)
            print(problem)
            # Take each domain and problem file pair and run val through it, write it to the corresponding text file.
            output_val_deepseek_chat = subprocess.run(["Parser", "apla-planner/generated_pddls_deepseek/dataset/domains/"+id+"/domain_deepseek_chat_"+id+".pddl", "apla-planner/generated_pddls_deepseek/dataset/problems/"+id+"/problem_deepseek_chat_"+interaction_id+".pddl"], stdout=subprocess.PIPE).stdout
            string_output_round2 = str(output_val_deepseek_chat, encoding='utf-8')
            if string_output_round2.find("Errors: 0,")==-1:
                val_error_detected = True
                print("\nOh no val error detected!\n")
                blocklist_ids.append(id)
                with open("apla-planner/generated_pddls_deepseek/dataset/blocklist.json", 'w') as block_file:
                    json.dump(blocklist_ids, block_file, indent=4)
                    block_file.close()
                    print("\nAdding scenario to blocklist\n")
                break #Exit the for loop for this set of interactions.
            with open("apla-planner/generated_pddls_deepseek/dataset/problems/"+id+"/val_output_"+interaction_id+".txt", "w", encoding='utf-8') as file:
                    file.write(string_output_round2) # We want to read the article as a single string, so that we can feed it to gpt.
                    file.close()
            
            ######### ============== Syntax verification feedback loop ============== ############
            # print("Considering syntax check, reviewing and updating domain file....\n")
            # response_domain_final = client_deepinfra.chat.completions.create(
            #     model=model_llama_33,
            #     messages=[
            #         {"role": "user", "content": f"""
            #         Here is some information about an autonomous vehicle scenario:
            #         {scenario_domain_problem_data[id]["Context"]}
                     
            #         Please have a look at the PDDL domain file provided:
            #         {response_domain_initial.choices[0].message.content}.

            #         Please have a look at the PDDL problem file provided:
            #         {response_problem_final.choices[0].message.content}

            #         Now please have a look at the output from a syntax checker:
            #         {string_output_round2}

            #         Are there any errors that the syntax checker points out? 
            #         Can you describe them and connect them to the given domain and problem file?

            #         Think step by step and update the domain file. I only want the domain file for now. 
            #         Double check that everything is clear and it does in fact have a solution.
            #         Do not write anything else other than what is asked. Only Only Only write what has been asked. Only write pure PDDL as asked. 
            #         Only write pure PDDL as asked. Only write pure PDDL as asked.

            #         Do not write ```pddl or ``` or ```lisp or the corresponding closing tags since I'm going to parse these outputs. 
            #         """},
            #     ],
            #     stream=False
            # )

            # print("Considering syntax check, reviewing and updating problem file....\n")
            # response_problem_final_final = client_deepinfra.chat.completions.create(
            #     model=model_llama_33,
            #     messages=[
            #         {"role": "user", "content": f"""
            #         Here is some information about an autonomous vehicle scenario:
            #         {scenario_domain_problem_data[id]["Context"]}
                    
            #         Please have a look at the PDDL domain file provided:
            #         {response_domain_initial.choices[0].message.content}.

            #         Please have a look at the PDDL problem file provided:
            #         {response_problem_final.choices[0].message.content}

            #         Now please have a look at the output from a syntax checker:
            #         {string_output_round2}

            #         In response to this, the following domain file was created:
            #         {response_domain_final.choices[0].message.content}

            #         Are there any errors pointed out by the syntax checker above? 
            #         Can you describe them and connect them to the given domain and problem file?

            #         Think step by step and update the problem file now. I only want the problem file now. 
            #         Double check that everything is clear and it does in fact have a solution.
            #         Do not write anything else other than what is asked. Only Only Only write what has been asked. Only write pure PDDL as asked. 
            #         Only write pure PDDL as asked. Only write pure PDDL as asked.

            #         Do not write ```pddl or ``` or ```lisp or the corresponding closing tags since I'm going to parse these outputs. 
            #         """},
            #     ],
            #     stream=False
            # )

            # with open(dir_path_text+"/domain_deepseek_chat_"+id+".pddl", "w", encoding='utf-8') as file:
            #     file.write(response_domain_final.choices[0].message.content) # We want to read the article as a single string, so that we can feed it to gpt.
            #     file.close() 
            
            # with open(dir_path_text_problem+"/problem_deepseek_chat_"+interaction_id+".pddl", "w", encoding='utf-8') as file:
            #         file.write(response_problem_final_final.choices[0].message.content) # We want to read the article as a single string, so that we can feed it to gpt.
            #         file.close()

            print("\nLLM grading for PDDL file generation....\n")
            llm_eval = pddl_gen.generate_llm_eval(scenario_domain_problem_data_problem_data=scenario_domain_problem_problem_data,
                                       domain=domain,
                                       problem_final=problem,
                                       scenario_id=id,
                                       interaction_id = interaction_id,
                                       scenario_domain_problem_data_context=scenario_domain_problem_data_context)
            
            print(f"\nPDDL problem generation complete for interaction with id {interaction_id}. Progress with interactions shown below\n")
        
        if val_error_detected==True:
            #Delete the domain folder for this scenario id completely.
            delete_path_domain = "apla-planner/generated_pddls_deepseek/dataset/domains/"+id
            shutil.rmtree(delete_path_domain)
            delete_path_problem = "apla-planner/generated_pddls_deepseek/dataset/problems/"+id
            shutil.rmtree(delete_path_problem)
            continue # Move on to the next scenario id
        print(f"\nPDDL generation complete for scenario with id {id}. Progress with scenarios shown below\n")

def pddl_response_and_answer_questions():
    client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    client_deepinfra = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")

    scenario_domain_and_problem_data = retrieve_womdr_domain_problem_data()
    for scenario_id in scenario_domain_and_problem_data.keys():
        for interaction_id in scenario_domain_and_problem_data[scenario_id]["Interactions"].keys():
            context = scenario_domain_and_problem_data[scenario_id]["Context"]
            question = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["problem_data"]
            answer = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["answer_data"]
            response_direct = client_oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"""
                Here is some information about an autonomous vehicle scenario:
                {scenario_domain_and_problem_data[scenario_id]["Context"]}

                Answer the following question:
                {question}

                Think step by step. Write a short 2 sentence answer only. Show your reasoning. 
                
                """},
                ],
                stream=False
            )

            domain_path = "generated_pddls/domain_deepseek_chat_"+scenario_id+".pddl"
            problem_file_path = "problem_deepseek_chat_"+scenario_id+"_"+interaction_id+".pddl"
            problem_path = "generated_pddls/"+problem_file_path

            with open(domain_path, 'r') as file_domain:
                pddl_domain = file_domain.readlines()
            
            with open(problem_path, 'r') as file_problem:
                pddl_problem = file_problem.readlines()

            with open("generated_pddls/plan_set.json", 'r') as plan_file:
                plan_dictionary = json.load(plan_file)

            response_gpt_4o_mini = client_oai.chat.completions.create(
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
                {plan_dictionary[problem_file_path]}

                Think step by step and answer the following question:
                {question}

                Write a short 2 sentence answer only. Show your reasoning.
                
                """},
                ],
                stream=False
            )

            response_deepseek_score = client_deepinfra.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "user", "content": f"""
                Here is some context about the test scenario:
                {context}

                This question was asked with regards to this context:
                {question}

                This is the ground truth answer:
                {answer}

                This was the attempt by an AI for this question
                {response_gpt_4o_mini.choices[0].message.content}

                Grade this answer on the following aspects:
                1. The correctness of the AI answer with respect to the ground truth answer. Give it a score between 1 to 10.
                Explain why this score was given by you in detail.
                2. The faithfulness of the reasoning. Are the conclusions drawn in the answer given by the AI consistent with its reasoning? Here, give it a score between 1 to 10.
                Explain why this score was given by you in detail.
                
                """},
                ],
                stream=False
            )

            print(response_direct.choices[0].message.content)
            print("\n")

            print("GPT 4o mini answer after reading the PDDL is:\n")
            print(response_gpt_4o_mini.choices[0].message.content)
            print("\n")
            print("Ground truth answer is:\n")
            print(answer)
            print("\n")
            with open("generated_pddls/deepseek_grades.txt", 'w') as grade_file:
                grade_file.writelines(response_deepseek_score.choices[0].message.content)
            print("Deepseek score grading response\n")
            print(response_deepseek_score.choices[0].message.content)
