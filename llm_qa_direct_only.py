## This script evaluates LLM responses when directly answering a question and when answering considering
## the logic of the PDDL file. 

import os
import json
import matplotlib.pyplot as plt 
import planner
from openai import OpenAI
from parsed_data_retrieval import retrieve_parsed_data
import argparse


########### ============  Global initializations ====================== ##########
parsed_file_list = os.listdir("parsed_womdr_data/")
client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#client_deepseek = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
client_deepinfra = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
scenario_domain_and_problem_data = retrieve_parsed_data()

parser = argparse.ArgumentParser(prog="Cruzway Reasoner 2.0",
                        description="Evaluate LLMs for autonomous vehicle test scenario reasoning")
parser.add_argument("-ns", "--nshot", type=str)
parser.add_argument("-sindex", "--scenario_index", type=int)


# The following are model names for DeepInfra provided models
# "deepseek-ai/DeepSeek-V3"
# "deepseek-ai/DeepSeek-R1" # This model thinks. 
# "meta-llama/Llama-3.3-70B-Instruct-Turbo"
# "meta-llama/Meta-Llama-3.1-405B-Instruct"
# "Qwen/Qwen2.5-72B-Instruct"
# "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" # This model thinks.
# "google/gemma-2-9b-it" *
# "meta-llama/Meta-Llama-3.1-8B-Instruct" *
# "Qwen/Qwen2.5-7B-Instruct" *
# "microsoft/phi-4" *

# # The following are the small model names for models provided via the OpenAI API service
# "gpt-4o-mini"
# "o3-mini"

model_dictionary = {
   "openai_models": {
       "gpt-4o-mini": [],
       #"o3-mini": []
       },
   "deepinfra_models":{
    #   "google/gemma-2-9b-it": [],
    #    "meta-llama/Meta-Llama-3.1-8B-Instruct": [],
    #    "Qwen/Qwen2.5-7B-Instruct": [],
    #    "microsoft/phi-4": []
   } 
}

# Generate two lists - domain file list and problem file list for a single scenario
# Reuse code in terms of classes and functions and 

model_outputs = {}
existing_grades = {} #This dictionary goes into the grades file.
scenario_qa_score = {}

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

################# ============== QA prompts =====================
def generate_qa_prompt(context, question, answer, prompt_type="4shot"):
    direct_prompt = f"""
        Here is some information about an autonomous vehicle scenario:
        {context}

        Answer the following question:
        {question}

        Think step by step. Show your reasoning and answer the question. 
        
        """
    #Reference for this prompt is scenario ID 10471914b8bb79a1 using interactions 0 and 7
    direct_cot_prompt_2shot = f"""
        I want you to answer some questions from the world of autonomous vehicle testing. 

        First, some information about the context: "Can you describe the type of intersection present in the current driving scenario? The intersection is a 4 way intersection.What is the status of the traffic light for the ego agent at this moment? The traffic light for the ego agent is green.Is there any information about the presence of stop signs, crosswalks, or speed bumps in the current scenario? There is no information about stop signs, crosswalks, or speed bumps in the current scenario.What is the ego agent's current action within the intersection? The ego agent is turning left and exiting the intersection.Could you specify the ego agent's current speed and whether it is increasing or decreasing? The ego agent's current speed is 13 m/s and it is accelerating.How does the current traffic light affect the ego agent's movement? The traffic light is green, which allows the ego agent to proceed.What type of agent is surrounding agent #0 and what is its current motion status? Surrounding agent #0 is a vehicle and it is accelerating.How is surrounding agent #0 positioned relative to the ego agent and the intersection? Surrounding agent #0 is on the left of the ego agent, in front of it, and is departing from the intersection.What is the current speed of surrounding agent #0? The current speed of surrounding agent #0 is 8 m/s.What type of agent is surrounding agent #1 and what is its current motion status? Surrounding agent #1 is a vehicle and it is not moving.How is surrounding agent #1 positioned relative to the ego agent and the intersection? Surrounding agent #1 is on the left of the ego agent, in front of it, and is heading towards the intersection.What type of agent is surrounding agent #2 and what is its current motion status? Surrounding agent #2 is a vehicle and it is decelerating.How is surrounding agent #2 positioned relative to the ego agent and the intersection? Surrounding agent #2 is on the left of the ego agent, in front of it, and is heading towards the intersection.What is the current speed of surrounding agent #2? The current speed of surrounding agent #2 is 3 m/s.What type of agent is surrounding agent #4 and what is its current motion status? Surrounding agent #4 is a vehicle and it is not moving.How is surrounding agent #4 positioned relative to the ego agent and the intersection? Surrounding agent #4 is on the left of the ego agent, behind it, and is on the same side of the intersection as the ego agent.What type of agent is surrounding agent #5 and what is its current motion status? Surrounding agent #5 is a vehicle and it is not moving.How is surrounding agent #5 positioned relative to the ego agent and the intersection? Surrounding agent #5 is on the left of the ego agent, behind it, and is departing from the intersection.What type of agent is surrounding agent #6 and what is its current motion status? Surrounding agent #6 is a vehicle and it is moving at a constant speed.How is surrounding agent #6 positioned relative to the ego agent and the intersection? Surrounding agent #6 is on the left of the ego agent, behind it, and is heading towards the intersection.Is there any traffic control affecting surrounding agent #6? Surrounding agent #6 is approaching a crosswalk 4 meters ahead.What type of agent is surrounding agent #7 and what is its current motion status? Surrounding agent #7 is a vehicle and it is not moving.How is surrounding agent #7 positioned relative to the ego agent and the intersection? Surrounding agent #7 is on the left of the ego agent, in front of it, and is on the same side of the intersection as the ego agent.What type of agent is surrounding agent #8 and what is its current motion status? Surrounding agent #8 is a vehicle and it is not moving.How is surrounding agent #8 positioned relative to the ego agent and the intersection? Surrounding agent #8 is on the left of the ego agent, in front of it, and is on the same side of the intersection as the ego agent." 

        Here is the first example of a question being answered:
        Question: "What interactions are expected between the ego agent and surrounding agent #0?"
        Answer: "Surrounding agent #0 will have no interaction with the ego agent as it is departing from the intersection and their paths do not conflict."

        Here is the second and last example of a question being answered:
        Question: "What is the ego agent's plan in the immediate future?"
        Answer: "The ego agent intends to complete its left turn and exit the intersection. It will proceed with the turn as the traffic light is green and it has the right of way. Surrounding agents #1 and #2 will yield to the ego agent, and surrounding agent #6 will likely stop at the crosswalk, so the ego agent does not need to alter its course in response to these agents."

        When analyzing and responding to driving scenarios, please do not, at any cost, conclude your reasoning with statements 1 through 11:
        1) Misinterpret or incorrectly assess the key scenario dynamics, agent positions, motion statuses, relative movements, spatial relationships, or right-of-way. Because of the aforementioned, do not incorrectly assume no interaction.
        2) Make unnecessary or speculative assumptions regarding potential interactions, collisions, evasive maneuvers, or conflicts not supported by the explicit context or ground truth. Prioritize explicit details in the context and ground truth.
        3) Elaborate on hypothetical, potential, or unwarranted interactions, trajectory adjustments, yielding, or collision avoidance unless directly stated in the scenario or ground truth. Prioritize explicit details in the context and ground truth.
        4) Overcomplicate the reasoning, introduce extraneous details, or speculate about future events, monitoring requirements, or passive observation that are not specified or relevant to the scenario. Prioritize explicit details in the context and ground truth.
        5) Provide overly detailed, verbose, redundant, or partially off-topic explanations; keep answers concise and focused on the central point. Prioritize key takeaways.
        6) Add information, nuanced explanations, or background reasoning beyond what is present in and supported by the ground truth. Prioritize explicit details in the ground truth.
        7) Fail to identify, mention, or precisely describe critical details explicitly present in the ground truth (e.g., yielding, overtaking, leading/following dynamics, acceleration, established right-of-way, explicit actions). Prioritize explicit details in the ground truth.
        8) Underestimate or overemphasize the importance of specific interactions compared to what is stated in the ground truth. Prioritize explicit details in the context and ground truth.
        9) Deviate from the ground truth by introducing unsupported, irrelevant, or misleading information, or by omitting key details prescribed in the scenario or answer. Prioritize explicit details in the context and ground truth.
        10) Present logical but incomplete, imprecise, or misaligned conclusions (with respect to the ground truth). Prioritize explicit details in the context and ground truth.
        11) Discuss possible but nonexistent proximity risks, interaction points, or minimal interactions not warranted by the scenario. Prioritize explicit details in the context and ground truth.
        Please do not make the mistakes listed above when analyzing and responding to driving scenarios.
        
        Given these examples now please have a look at the following new context and try to answer the following question pertaining to the new context:
        Here is the context: {context}

        Here is the question: {question}

        """
    
    #using interactions 0 2 4 6
    direct_cot_prompt_4shot = f"""
        I want you to answer some questions from the world of autonomous vehicle testing. 

        First, some information about the context: "Can you describe the type of intersection present in the current driving scenario? The intersection is a 4 way intersection.What is the status of the traffic light for the ego agent at this moment? The traffic light for the ego agent is green.Is there any information about the presence of stop signs, crosswalks, or speed bumps in the current scenario? There is no information about stop signs, crosswalks, or speed bumps in the current scenario.What is the ego agent's current action within the intersection? The ego agent is turning left and exiting the intersection.Could you specify the ego agent's current speed and whether it is increasing or decreasing? The ego agent's current speed is 13 m/s and it is accelerating.How does the current traffic light affect the ego agent's movement? The traffic light is green, which allows the ego agent to proceed.What type of agent is surrounding agent #0 and what is its current motion status? Surrounding agent #0 is a vehicle and it is accelerating.How is surrounding agent #0 positioned relative to the ego agent and the intersection? Surrounding agent #0 is on the left of the ego agent, in front of it, and is departing from the intersection.What is the current speed of surrounding agent #0? The current speed of surrounding agent #0 is 8 m/s.What type of agent is surrounding agent #1 and what is its current motion status? Surrounding agent #1 is a vehicle and it is not moving.How is surrounding agent #1 positioned relative to the ego agent and the intersection? Surrounding agent #1 is on the left of the ego agent, in front of it, and is heading towards the intersection.What type of agent is surrounding agent #2 and what is its current motion status? Surrounding agent #2 is a vehicle and it is decelerating.How is surrounding agent #2 positioned relative to the ego agent and the intersection? Surrounding agent #2 is on the left of the ego agent, in front of it, and is heading towards the intersection.What is the current speed of surrounding agent #2? The current speed of surrounding agent #2 is 3 m/s.What type of agent is surrounding agent #4 and what is its current motion status? Surrounding agent #4 is a vehicle and it is not moving.How is surrounding agent #4 positioned relative to the ego agent and the intersection? Surrounding agent #4 is on the left of the ego agent, behind it, and is on the same side of the intersection as the ego agent.What type of agent is surrounding agent #5 and what is its current motion status? Surrounding agent #5 is a vehicle and it is not moving.How is surrounding agent #5 positioned relative to the ego agent and the intersection? Surrounding agent #5 is on the left of the ego agent, behind it, and is departing from the intersection.What type of agent is surrounding agent #6 and what is its current motion status? Surrounding agent #6 is a vehicle and it is moving at a constant speed.How is surrounding agent #6 positioned relative to the ego agent and the intersection? Surrounding agent #6 is on the left of the ego agent, behind it, and is heading towards the intersection.Is there any traffic control affecting surrounding agent #6? Surrounding agent #6 is approaching a crosswalk 4 meters ahead.What type of agent is surrounding agent #7 and what is its current motion status? Surrounding agent #7 is a vehicle and it is not moving.How is surrounding agent #7 positioned relative to the ego agent and the intersection? Surrounding agent #7 is on the left of the ego agent, in front of it, and is on the same side of the intersection as the ego agent.What type of agent is surrounding agent #8 and what is its current motion status? Surrounding agent #8 is a vehicle and it is not moving.How is surrounding agent #8 positioned relative to the ego agent and the intersection? Surrounding agent #8 is on the left of the ego agent, in front of it, and is on the same side of the intersection as the ego agent."

        Here is the first example of a question being answered:
        Question: "What interactions are expected between the ego agent and surrounding agent #0?"
        Answer: "Surrounding agent #0 will have no interaction with the ego agent as it is departing from the intersection and their paths do not conflict." 

        Here is the second example of a question being answered:
        Question: "What is the nature of the interaction between the ego agent and surrounding agent #2?"
        Answer: "Surrounding agent #2 will yield to the ego agent as it is decelerating and heading towards the intersection while the ego agent is exiting the intersection with a green light."

        Here is the third example of a question being answered:
        Question: "What interaction will occur between the ego agent and surrounding agent #6?"
        Answer: "Surrounding agent #6 will yield to the ego agent as it is on the right of the intersection and is approaching a crosswalk, indicating it may need to stop, while the ego agent is actively exiting the intersection."

        Here is the fourth and last example of a question being answered:
        Question:  "What kind of interaction will take place between the ego agent and surrounding agent #8?"
        Answer: "Surrounding agent #8 will have no interaction with the ego agent as it is not moving and is on the same side of the intersection as the ego agent."

        scienceqa lecturing...explain what an interaction is

        When analyzing and responding to driving scenarios, please do not, at any cost, conclude your reasoning with points 1 through 8:
        1) Speculate about potential or hypothetical interactions, maneuvers, conflicts, or yielding behavior that are not supported by the provided context or ground truth. Please prioritize explicit details in the context and ground truth.
        2) Overcomplicate the analysis with unnecessary details, verbose explanations, or elaborate on irrelevant possible actions or future trajectories not mentioned in the scenario. Please prioritize explicit details in the context.
        3) Misinterpret or incorrectly assess the key scenario dynamics, agent positions, motion statuses, relative movements, spatial relationships, or right-of-way. Because of the aforementioned, never assume no interaction.
        4) Introduce unwarranted scenarios (e.g., unnecessary speed or trajectory adjustments, collision risks, overtaking, or yielding) that detract from the actual dynamics and facts described. Please prioritize explicit details in the context.
        5) Omit or fail to align explanations with critical details or key points from the ground truth, especially regarding the presence or absence of interactions and required behaviors. Please prioritize explicit details in the context and ground truth.
        6) Offer conclusions or reasoning that contradict the context, such as stating there is (or is not) an interaction when the ground truth indicates the opposite, or introduce unsupported assumptions about right-of-way, vehicle intent, or agent relationships. Please prioritize explicit details in the ground truth.
        7) Add irrelevant commentary or speculative concerns that dilute the core answer. Please prioritize explicit details in the context and ground truth.
        8) Focus your answer on concise, accurate, and context-supported analysis aligned with the scenario's ground truth. Do not expand with hypothetical, unnecessary, or unsupported information. Please prioritize explicit details in the context and ground truth.
        Please do not make the mistakes listed above when analyzing and responding to driving scenarios. 

        Given these examples now please have a look at the following new context and try to answer the following question pertaining to the new context:
        Here is the context: {context}

        Here is the question: {question}

        """

    #using interactions 0 1 2 3 5 7
    direct_cot_prompt_6shot = f"""
        I want you to answer some questions from the world of autonomous vehicle testing. 

        Here are some examples of questions being answered:
        First, some information about the context: "Can you describe the type of intersection present in the current driving scenario? The intersection is a 4 way intersection.What is the status of the traffic light for the ego agent at this moment? The traffic light for the ego agent is green.Is there any information about the presence of stop signs, crosswalks, or speed bumps in the current scenario? There is no information about stop signs, crosswalks, or speed bumps in the current scenario.What is the ego agent's current action within the intersection? The ego agent is turning left and exiting the intersection.Could you specify the ego agent's current speed and whether it is increasing or decreasing? The ego agent's current speed is 13 m/s and it is accelerating.How does the current traffic light affect the ego agent's movement? The traffic light is green, which allows the ego agent to proceed.What type of agent is surrounding agent #0 and what is its current motion status? Surrounding agent #0 is a vehicle and it is accelerating.How is surrounding agent #0 positioned relative to the ego agent and the intersection? Surrounding agent #0 is on the left of the ego agent, in front of it, and is departing from the intersection.What is the current speed of surrounding agent #0? The current speed of surrounding agent #0 is 8 m/s.What type of agent is surrounding agent #1 and what is its current motion status? Surrounding agent #1 is a vehicle and it is not moving.How is surrounding agent #1 positioned relative to the ego agent and the intersection? Surrounding agent #1 is on the left of the ego agent, in front of it, and is heading towards the intersection.What type of agent is surrounding agent #2 and what is its current motion status? Surrounding agent #2 is a vehicle and it is decelerating.How is surrounding agent #2 positioned relative to the ego agent and the intersection? Surrounding agent #2 is on the left of the ego agent, in front of it, and is heading towards the intersection.What is the current speed of surrounding agent #2? The current speed of surrounding agent #2 is 3 m/s.What type of agent is surrounding agent #4 and what is its current motion status? Surrounding agent #4 is a vehicle and it is not moving.How is surrounding agent #4 positioned relative to the ego agent and the intersection? Surrounding agent #4 is on the left of the ego agent, behind it, and is on the same side of the intersection as the ego agent.What type of agent is surrounding agent #5 and what is its current motion status? Surrounding agent #5 is a vehicle and it is not moving.How is surrounding agent #5 positioned relative to the ego agent and the intersection? Surrounding agent #5 is on the left of the ego agent, behind it, and is departing from the intersection.What type of agent is surrounding agent #6 and what is its current motion status? Surrounding agent #6 is a vehicle and it is moving at a constant speed.How is surrounding agent #6 positioned relative to the ego agent and the intersection? Surrounding agent #6 is on the left of the ego agent, behind it, and is heading towards the intersection.Is there any traffic control affecting surrounding agent #6? Surrounding agent #6 is approaching a crosswalk 4 meters ahead.What type of agent is surrounding agent #7 and what is its current motion status? Surrounding agent #7 is a vehicle and it is not moving.How is surrounding agent #7 positioned relative to the ego agent and the intersection? Surrounding agent #7 is on the left of the ego agent, in front of it, and is on the same side of the intersection as the ego agent.What type of agent is surrounding agent #8 and what is its current motion status? Surrounding agent #8 is a vehicle and it is not moving.How is surrounding agent #8 positioned relative to the ego agent and the intersection? Surrounding agent #8 is on the left of the ego agent, in front of it, and is on the same side of the intersection as the ego agent."
        
        Here is the first example of a question being answered:
        Question: "What interactions are expected between the ego agent and surrounding agent #0?"
        Answer: "Surrounding agent #0 will have no interaction with the ego agent as it is departing from the intersection and their paths do not conflict." 

        Here is the second example of a question being answered:
        Question: "How will the ego agent and surrounding agent #1 interact as they are both near the intersection?"
        Answer: "Surrounding agent #1 will yield to the ego agent because the ego agent has the right of way with a green traffic light and is already exiting the intersection while surrounding agent #1 is not moving."

        Here is the third example of a question being answered:
        Question: "What is the nature of the interaction between the ego agent and surrounding agent #2?"
        Answer: "Surrounding agent #2 will yield to the ego agent as it is decelerating and heading towards the intersection while the ego agent is exiting the intersection with a green light." 

        Here is the fourth example of a question being answered:
        Question: "Can you describe the interaction between the ego agent and surrounding agent #4?"
        Answer: "There will be no interaction between the ego agent and surrounding agent #4 as surrounding agent #4 is not moving and is on the same side of the intersection as the ego agent."

        Here is the fifth example of a question being answered:
        Question: "What will be the interaction between the ego agent and surrounding agent #7?"
        Answer: "Surrounding agent #7 will have no interaction with the ego agent as it is not moving and is on the same side of the intersection as the ego agent."

        Here is the sixth and last example of a question being answered:
        Question: "What is the ego agent's plan in the immediate future?"
        Answer: "The ego agent intends to complete its left turn and exit the intersection. It will proceed with the turn as the traffic light is green and it has the right of way. Surrounding agents #1 and #2 will yield to the ego agent, and surrounding agent #6 will likely stop at the crosswalk, so the ego agent does not need to alter its course in response to these agents."

        When analyzing driving scenarios, please do not, at any cost, conclude your reasoning with points 1 through 11:
        1) Speculate about potential or hypothetical interactions unless clearly indicated in the scenario. Please prioritize explicit details in the context.
        2) Provide overly detailed analyses or introduce unnecessary complexity not present in the ground truth. Please prioritize explicit details in the ground truth.
        3) Assume or infer yielding behavior, collisions, or path conflicts unless explicitly described. Please prioritize explicit details in the context and ground truth.
        4) Include information about right of way, speed bumps, or surrounding agents unless it is directly supported by the scenario and ground truth. Please prioritize explicit details in the context and ground truth.
        5) Misinterpret the relative positions, motion statuses, or the spatial/directional dynamics of the agents. Because of the aforementioned, never assume no interaction. Do not ignore differences in relative speed and position between agents. Do not state "no interaction" just because there is no immediate collision or lane change. Do not overlook that a decelerating ego agent falling behind a constant-speed leader is still an interaction. Do not disregard lead-follow dynamics as unimportant to agent interactions. Do not provide shallow or incomplete analysis of agent relationships.
        6) Overanalyze or focus on possible but unsupported outcomes or secondary details. Please prioritize explicit details in the context and ground truth.
        7) Omit key points stated in the ground truth, such as explicit yielding, leading/following dynamics, or the order of actions. Please analyze the potential of the aforementioned, specifically how decelerating and constant speed combined cause no interaction.
        8) Use vague conclusions like "minimal" or "negligible" interaction instead of directly stating "no interaction" where appropriate. Please prioritize explicit details in your reasoning.
        9) Address passive observation, secondary potential scenarios, or future interactions unless specified by the scenario. Please prioritize explicit details in the context.
        10) Ignore or fail to directly address the main interaction or key behavioral details described in the ground truth. Please prioritize explicit details in your reasoning.
        11) Keep responses concise, directly address the core scenario as described, and strictly adhere to the information provided in the ground truth. Please prioritize explicit details in the ground truth.
        Please do not make the mistakes listed above when analyzing and responding to driving scenarios. 

        Given these examples now please have a look at the following new context and try to answer the following question pertaining to the new context:
        Here is the context: {context}

        Here is the question: {question}

        """

    if prompt_type=="4shot":
        return direct_cot_prompt_4shot
    elif prompt_type=="0shot":
        return direct_prompt
    elif prompt_type=="2shot":
        return direct_cot_prompt_2shot
    elif prompt_type=="6shot":
        return direct_cot_prompt_6shot

################# ============= Grading via LLM as a judge prompts ================== ###############
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

        Grade this answer on the following aspect:
        The correctness of the AI answer with respect to the ground truth answer. Give it a score between 1 to 10.
        Explain why this score was given by you in detail.
        Format the answer in a python dictionary format like this.
        <open curly bracket>:
        "Correctness score": "<Only enter the score number here>",
        "Correctness explanation": "<Write your explanation here>"
        <close curly bracket>
        
        Don't write anything else. Nothing else, nothing else, nothing else. 
        Please only write it in the format requested.

        """

    return grading_prompt

############### =============== Evaluating Interactions ================ ##############
def grade_openai_deepinfra_models_one_interaction(model_dictionary, 
                                                  existing_grades,
                                                  scenario_id, 
                                                  interaction_id,
                                                  prompt_type):
    
    #### Step 1: Generate the PDDL prompts ======================= #########
    context = scenario_domain_and_problem_data[scenario_id]["Context"]
    question = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["problem_data"]
    answer = scenario_domain_and_problem_data[scenario_id]["Interactions"][interaction_id]["answer_data"]
    context_word_count = scenario_domain_and_problem_data[scenario_id]["Word Count"]

    generated_prompt = generate_qa_prompt(context, question, answer, prompt_type)
    #### Step 2: Generate the model grades and add them to the dictionary
    
    for model_family in model_dictionary.keys():
        if model_family=="openai_models":
            for model_name in model_dictionary[model_family]:
                predicted_answer = openai_call(model_name=model_name, prompt=generated_prompt)
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=openai_call(model_name=model_name, prompt=generated_prompt))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_modelname", grading_output
                    )
                correctness = int(grading_output["Correctness score"])
                #existing_grades[scenario_id][interaction_id][model_family+"_"+model_name+"_modelname"].setdefault("Correctness Average", (str(correctness)))

                existing_grades[scenario_id][interaction_id][model_family+"_"+model_name+"_modelname"].setdefault("Word Count", (str(context_word_count)))
                model_dictionary[model_family][model_name].append(correctness)
        elif model_family=="deepinfra_models":
            for model_name in model_dictionary[model_family]:
                predicted_answer = deepinfra_call(model_name=model_name, prompt=generated_prompt)
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=deepinfra_call(model_name=model_name, prompt=generated_prompt))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_modelname", grading_output
                    )
                correctness = int(grading_output["Correctness score"])

                #existing_grades[scenario_id][interaction_id][model_family+"_"+model_name+"_modelname"].setdefault("Correctness Score", (str(total_correctness_average)))

                existing_grades[scenario_id][interaction_id][model_family+"_"+model_name+"_modelname"].setdefault("Word Count", (str(context_word_count)))
                model_dictionary[model_family][model_name].append(correctness)
    

def pddl_response_and_answer_questions(prompt_type="4shot", arguments=None):
    # Parse through the preprocessed json data contained in parsed_womdr_data/
    for scenario_id in scenario_domain_and_problem_data.keys():
        existing_grades = {}
        existing_grades.setdefault(scenario_id, {})
        current_scenario_index = scenario_domain_and_problem_data[scenario_id]["Scenario Index"]
        if (arguments.scenario_index!=None) and (current_scenario_index!=arguments.scenario_index):
            continue
        existing_grades[scenario_id].setdefault("Scenario Index", current_scenario_index)
        for interaction_id in scenario_domain_and_problem_data[scenario_id]["Interactions"].keys():
            existing_grades[scenario_id].setdefault(interaction_id, {})
              
            ##### ===================== Automatic model evaluation with LLM grades on outputs ============== #########
            grade_openai_deepinfra_models_one_interaction(model_dictionary=model_dictionary, 
                                                        existing_grades=existing_grades, 
                                                        scenario_id=scenario_id, 
                                                        interaction_id=interaction_id,
                                                        prompt_type=prompt_type)
            
            #Ensure that this json file by the name grades/deepseek_grades.json exists first.
        with open("grades/grades_"+"scenario_index_"+str(current_scenario_index)+"_prompt_type_"+prompt_type+"_"+scenario_id+".json", 'w') as grade_file:
            print("Existing grades is given by {}".format(existing_grades))
            json.dump(existing_grades, grade_file, indent=4)
            grade_file.close()

def main():
    # Change parameter here depending on the prompt.
    arguments = parser.parse_args()
    print(f"The argument is {arguments.nshot}")
    if arguments.nshot=="4shot" or arguments.nshot=="0shot" or arguments.nshot=="2shot" or arguments.nshot=="6shot":
        prompt_type = arguments.nshot
    else: prompt_type = "2shot"
    print(f"Prompt type is {prompt_type}") 
    print(f"The scenario index argument is {arguments.scenario_index}")
    pddl_response_and_answer_questions(prompt_type=prompt_type, arguments=arguments)
    for model_provider in model_dictionary.keys():
        for model in model_dictionary[model_provider].keys():
            plt.bar([i for i in range(len(model_dictionary[model_provider][model]))], model_dictionary[model_provider][model])
            plt.title("Correctness Scores for All Scenarios of Current Exp.")
            plt.xlabel("Interactions")
            plt.ylabel("Correctness Scores")
            plt.show()
main()