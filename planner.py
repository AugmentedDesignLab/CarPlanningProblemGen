import guidance
import sys
import subprocess
import os
import json
from pathlib import Path
import pddlpy
from openai import OpenAI
from client_model_setup import ProvidedLLM
from tqdm import tqdm

provided_llm = ProvidedLLM() #Object contains all the client setup and the model names for that client.

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
                for interaction_key in scenario_data[key]["Interactions"].keys():
                    scenario_domain_problem_data[i[:-5]].setdefault("Interactions", {})
                    scenario_domain_problem_data[i[:-5]]["Interactions"].setdefault(interaction_key, {
                    "problem_data": "",
                    "answer_data": ""
                    }) 
                    scenario_domain_problem_data[i[:-5]]["Interactions"][interaction_key]["problem_data"] = scenario_data[key]["Interactions"][interaction_key]["reference_question"]
                    scenario_domain_problem_data[i[:-5]]["Interactions"][interaction_key]["answer_data"] = scenario_data[key]["Interactions"][interaction_key]["reference_answer"]
       
    return scenario_domain_problem_data

def resolve_client_and_model(api_type, model_name):
    # API_type parameter must be from the following names:
    # 1. ds_api
    # 2. deepinfra_api
    # 3. oai_api
    
    # For ds models, model names should be from the following:
    # 1. ds_v3_dsapi
    # 2. ds_r1_dsapi

    # For deepinfra models, model names should be from the following:
    # 1. ds_v3,
    # 2. llama_33_70b
    # 3. llama_31_405b
    # 4. qw_25_72b
    # 5. ds_distil_llama_70b
    # 6. gemma_2
    # 7. llama_31_8b
    # 8. qw_25_7b
    # 9. phi_4
    
    # For OpenAI models, model names should be from the following:
    # 1. gpt_4o_mini
    # 2. o3_mini
    
    if api_type=="ds_api":
        client = provided_llm.client_dsapi
        if model_name=="ds_v3_dsapi": 
            selected_model = provided_llm.ds_v3_dsapi
        elif model_name=="ds_r1_dsapi":
            selected_model = provided_llm.ds_r1_dsapi
        else:
            print("Model name is incompatible with DS api or invalid")
    elif api_type=="deepinfra_api":
        client = provided_llm.client_deepinfra
        if model_name=="ds_v3":
            selected_model = provided_llm.ds_v3
        elif model_name=="llama_33_70b":
            selected_model = provided_llm.llama_33_70b
        elif model_name=="ds_distil_llama_70b":
            selected_model = provided_llm.ds_distil_llama_70b
        else:
            print("model name either incompatible with DeepInfra API or invalid.") 
    elif api_type=="oai_api":
        client = provided_llm.client_oai
        selected_model = provided_llm.gpt_4o_mini
    else: 
        print("API type invalid")
    
    return client, selected_model

def generate_pddl_with_syntax_check(api_type, model_name):
    client, selected_model = resolve_client_and_model(api_type=api_type, model_name=model_name)
    scenario_domain_problem_data = retrieve_womdr_domain_problem_data()  
    for id in tqdm(scenario_domain_problem_data.keys()):
        print("\nDomain generation, generating action suggestions....\n")
        response_action_json = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "user", "content": f"""
            
            Based on the information detailed in {scenario_domain_problem_data[id]["Context"]}, 
            * Write down a list of actions that map between states in natural language. 
            * Each action has some causal states (predicates) and some effect states that will be true or false.
            * Each action is a cause and effect mapping between any number of causal states and any number of effect states.
            * Actions and states must not contradict each other.
            * Action names must be descriptive and the action can be understood just by looking at the name.
            * The state names within each action are also descriptive. The cause and effect statements and the state names must have the same information.
            * There must be separate states regarding the environment, ego and the respective surrounding agents.
            * In each action and state, the ego agent or the surrounding agent must be identified as <EGO> or <SURROUNDING AGENT #0> or <SURROUNDING AGENT #1> as needed.
            * For distances, positions and speeds do not use specific numbers but words instead such as front, left, right, near, far, fast, slow, medium (or combinations such as front-left and so on) or other similar descriptive words. 
            * The action itself will only become true when the causal states and the effect states are in the specific states that this description details.
            * Write them in the following format:  
            <open curly bracket>
                "<action name>": 
                <open curly bracket> 
                    "<state name>": <open curly bracket> 
                        "statement": "<the assertion in natural language. Use the fewest words possible for maximum clarity>
                        "value": <Whether this value is true for false>,
                        "state_type": <whether this state is a cause or effect for the current action>
                    <close curly bracket>, 
                    "<state name>": <curly bracket> 
                        "statement": "<the assertion in natural language. Use the fewest words possible for maximum clarity>
                        "value": <Whether this value is true for false>,
                        "state_type": <whether this state is a cause or effect for the current action>
                    <close curly bracket>
                <close curly bracket>, 
                ... 
            <close curly bracket>

            No json tags to be used. Just the dictionary in the output. Nothing else, nothing else, nothing else.   
            """},
            ],
            stream=False
        )
        print(f"\nDomain generation, generating domain file for scenario id {id}....\n")
        response_domain_initial = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "user", "content": f"""
                We need you to write specific driving behaviors to accomplish certain goals. A behavior is defined as actions taken in response to certain conditions. Conditions are provided as an environment state. 
                Think about some states by yourself that you believe is necessary. 
                Vehicles navigate in the action space and the state space provided to them.

                Now generate a PDDL domain file for the scenario: {response_action_json.choices[0].message.content}. Domain file only only only for now.
                Think about the STRIPS PDDL for different popular domains such as gripper and sokoban.
                Verify whether all the suggested states and actions makes sense and are correct.
                If it feels correct, write it down as a PDDL domain file. I only only want the PDDL domain file contents.
                
                Please keep things really clear. Do not repeat names. Do not repeat names. Do not redefine anything. Ensure that everything is very very clear and correct. Check and double check correctness.
                Do not write anything else other than what is asked. Only Only Only write what has been asked. No tags of any sort. Only pure PDDL. Only write what has been asked. Only write what has been asked.
                Nothing other than pure PDDL as asked. Nothing other than pure PDDL as asked. Please make sure it is correct.
                Do not write ```pddl or ``` or the corresponding closing tags since I'm going to parse these outputs. 
                
                I repeat, do not write ```pddl or ``` or ```lisp or the corresponding closing tags since I'm going to parse these outputs. 
                I repeat again, do not write ```pddl or ``` or ```lisp or the corresponding closing tags since I'm going to parse these outputs. 
                """},
            ],
            stream=False
        )

        dir_path_text = "apla-planner/generated_pddls_deepseek/dataset/domains/"+id
        try: 
            dir_path = Path(dir_path_text)
            dir_path.mkdir()
            with open(dir_path_text+"/domain_deepseek_chat_"+id+".pddl", "w", encoding='utf-8') as file:
                file.write(response_domain_initial.choices[0].message.content) # We want to read the article as a single string, so that we can feed it to gpt.
                file.close()
        except FileExistsError:
            with open(dir_path_text+"/domain_deepseek_chat_"+id+".pddl", "w", encoding='utf-8') as file:
                file.write(response_domain_initial.choices[0].message.content) # We want to read the article as a single string, so that we can feed it to gpt.
                file.close()

        # Given one domain file based on a context, generate multiple problem files.
        for interaction_id in tqdm(scenario_domain_problem_data[id]["Interactions"].keys()):    
            print(f"\nProblem generation, generating problem file for interaction {interaction_id}....\n")
            response_problem_initial = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "user", "content": f"""
                    Now carefully write the PDDL problem file for the corresponding domain file provided:
                    {response_domain_initial.choices[0].message.content}.

                    Consider in addition some problem specific data: {scenario_domain_problem_data[id]["Interactions"][interaction_id]["problem_data"]}
                    First repeat the types, states (predicates) and actions in this file as a list in natural language. 
                    Then think step by step about a problem for this domain. Think about whether this problem does indeed have a solution plan.
                    Double check that everything is clear and it does in fact have a solution. Then write the PDDL problem file contents. I only want the problem file contents. 
                    Do not repeat names. Do not repeat names. Only the problem file contents nothing more. Only the problem file contents nothing more. I'm pasting this in a pddl problem file just letting you know. 
                    Do not write anything else other than what is asked. Only Only Only write what has been asked. Only write pure PDDL as asked. 
                    Only write pure PDDL as asked. Only write pure PDDL as asked.

                    Do not write ```pddl or ``` or ```lisp or the corresponding closing tags since I'm going to parse these outputs. 
                    """},
                ],
                stream=False
            )

            print("\nProblem generation, reviewing and updating problem file....\n")
            response_problem_final = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "user", "content": f"""
                    Carefully read this PDDL problem file:
                    {response_problem_initial.choices[0].message.content}.

                    It is really important that the ```pddl or ``` or ```lisp opening tags 
                    or the corresponding closing tags do not exist. Do these tags exist in the given PDDL problem file?
                    Do not write your answer in the output. But if the answer is yes, can you remove the lines with these tags 
                    and rewrite the rest of the PDDL file exactly as it is? The lines with the tags should definitely not be there in the final output. 
                    If the answer is no however, please rewrite the file exactly as it is. Thank you! 
                    
                    Again, remember that the final output should only have lines of PDDL as instructed above, nothing else, nothing else, nothing else.
                    """},
                ],
                stream=False
            )

            dir_path_text_problem = "apla-planner/generated_pddls_deepseek/dataset/problems/"+id
            try:
                # Try creating folder if it doesn't exist. Create only file if it does. 
                dir_path_problem = Path(dir_path_text_problem)
                dir_path_problem.mkdir()
                with open(dir_path_text_problem+"/problem_deepseek_chat_"+interaction_id+".pddl", "w", encoding='utf-8') as file:
                        file.write(response_problem_final.choices[0].message.content) # We want to read the article as a single string, so that we can feed it to gpt.
                        file.close()
            except FileExistsError:
                with open(dir_path_text_problem+"/problem_deepseek_chat_"+interaction_id+".pddl", "w", encoding='utf-8') as file:
                        file.write(response_problem_final.choices[0].message.content) # We want to read the article as a single string, so that we can feed it to gpt.
                        file.close()


            # Take each domain and problem file pair and run val through it, write it to the corresponding text file.
            output_val_deepseek_chat = subprocess.run(["Parser", "apla-planner/generated_pddls_deepseek/dataset/domains/"+id+"/domain_deepseek_chat_"+id+".pddl", "apla-planner/generated_pddls_deepseek/dataset/problems/"+id+"/problem_deepseek_chat_"+interaction_id+".pddl"], stdout=subprocess.PIPE).stdout
            string_output_round2 = str(output_val_deepseek_chat, encoding='utf-8')
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
            response_LLM_judgement = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "user", "content": f"""
                    First, read the context information for the given scenario:
                    {scenario_domain_problem_data[id]["Context"]}
                    
                    Now, carefully read the generated domain file:
                    {response_domain_initial.choices[0].message.content}

                    Now, carefully review the problem data in the scenario:
                    {scenario_domain_problem_data[id]["Interactions"][interaction_id]["problem_data"]}
                    
                    Carefully read this PDDL problem file:
                    {response_problem_final.choices[0].message.content}.

                    Now score the generated domain and problem PDDL files according to the given rubric:

                    1. Consistency: Are the facts in the context information above consistently and correctly presented in the domain and problem files? Rate this output on a scale of 1 to 10. Explain your rating. 
                    2. Domain coverage: Does the generated domain PDDL domain file adequately cover the information in the context above? Rate this output on a scale of 1 to 10. Explain your rating.
                    3. Problem coverage: Does the generated problem PDDL file adequately cover the given problem data as presented above? The problem data asks specific questions with respect to the context. 
                    Therefore, you must rate the coverage with respect to this specific question only. Rate this output on a scale of 1 to 10. Explain your rating.

                    Format your output exactly in the following manner:
                    <open curly bracket>
                    "Context": "<Initial contextual information of the scenario exactly as it is above.>",
                    "Consistency":
                        <open curly bracket>
                        "Score explanation": "<Detailed explanation here.>", 
                        "Grade": "<Only a score here between 1 and 10.>" 
                        <close curly bracket>,
                    "Domain coverage":
                        <open curly bracket>
                        "Score explanation": "<Detailed explanation here.>", 
                        "Grade": "<Only a score here between 1 and 10.>" 
                        <close curly bracket>,
                    "Problem coverage":
                        <open curly bracket>
                        "Problem data provided": "<Problem data given exactly as it is above.>"
                        "Score explanation": "<Detailed explanation here.>", 
                        "Grade": "<Only a score here between 1 and 10.>" 
                        <close curly bracket>
                    <close curly bracket>

                    No tags. Just the dictionary in the output. Nothing else, nothing else.
                    """},
                ],
                stream=False
            )

            LLM_eval_dictionary = eval(response_LLM_judgement.choices[0].message.content)
            # Each sentence in the scenario context pertains to a fact.
            # We can split the context by sentence and count the word count per sentence to get a sense of how difficult the facts are.
            # Longer individual sentences would mean more complex facts.
            context_sentence_list = scenario_domain_problem_data[id]["Context"].split(". ")
            total_word_count_sentence = 0
            for sentence_index in range(len(context_sentence_list)): 
                 total_word_count_sentence += len(context_sentence_list[sentence_index].split())
            
            average_word_count_sentence = total_word_count_sentence / len(context_sentence_list)
            
            LLM_eval_dictionary.setdefault("average_context_sentence_word_count", average_word_count_sentence)

            domain_problem_files = pddlpy.DomainProblem("apla-planner/generated_pddls_deepseek/dataset/domains/"+id+"/domain_deepseek_chat_"+id+".pddl", 
                                                        "apla-planner/generated_pddls_deepseek/dataset/problems/"+id+"/problem_deepseek_chat_"+interaction_id+".pddl")
            LLM_eval_dictionary.setdefault("domain_action_count", len(list(domain_problem_files.operators()))) # List of actions written in the domain.
            LLM_eval_dictionary.setdefault("initial_state_size", len(domain_problem_files.initialstate())) # Initial state in the problem file.    

            with open(dir_path_text_problem+"/LLM_eval_"+interaction_id+".json", "w", encoding='utf-8') as file_eval:
                        json.dump(LLM_eval_dictionary, file_eval, indent=4) # We want to read the article as a single string, so that we can feed it to gpt.
                        file.close()
            print(f"\nPDDL problem generation complete for interaction with id {interaction_id}. Progress with interactions shown below\n")
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
