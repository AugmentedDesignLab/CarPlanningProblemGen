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
   "openai_models": ["gpt-4o-mini", "gpt-4.5-preview"],
   "deepinfra_models": [] 
}

# Generate two lists - domain file list and problem file list for a single scenario
# Reuse code in terms of classes and functions and 

model_outputs = {}
existing_grades = {}
qa_scores_o3_mini = []
qa_scores_qwen25_7b = []

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
    
    direct_cot_prompt_4shot = f"""
    I want you to answer some questions from the world of autonomous vehicle testing. 

    Here are some examples of questions being answered:
    First, some information about the context: "Can you describe the current road configuration in terms of lanes? The road has three lanes.What traffic controls are present in the current driving scene? There are no traffic controls present in the current driving scene.What is the ego agent's current velocity? The ego agent's current speed is 6 meters per second.Is the ego agent's speed constant or changing? The ego agent is accelerating.Could you specify the ego agent's current lane position? The ego agent is on the first lane from the right.What is the ego agent's current direction of travel? The ego agent is heading in the same direction as its current lane.What type of agent is surrounding agent #0? Surrounding agent #0 is a vehicle.How fast is surrounding agent #0 moving at the moment? Surrounding agent #0's current speed is 5 meters per second.What is the motion status of surrounding agent #0? Surrounding agent #0 is accelerating.Where is surrounding agent #0 in relation to the ego agent? Surrounding agent #0 is 4 meters on the left and 1 meter in front of the ego agent.What direction is surrounding agent #0 facing compared to the ego agent? Surrounding agent #0 is heading in the same direction as the ego agent.What type of agent is surrounding agent #1? Surrounding agent #1 is a vehicle.What is the current speed of surrounding agent #1? Surrounding agent #1's current speed is 4 meters per second.Is surrounding agent #1 accelerating or maintaining its speed? Surrounding agent #1 is moving at a constant speed.Can you describe the position of surrounding agent #1 relative to the ego agent? Surrounding agent #1 is 24 meters behind and 3 meters on the left of the ego agent.In which direction is surrounding agent #1 moving with respect to the ego agent? Surrounding agent #1 is heading in the same direction as the ego agent.What type of agent is surrounding agent #3? Surrounding agent #3 is a vehicle.What is the current velocity of surrounding agent #3? Surrounding agent #3 is not moving.Where is surrounding agent #3 located in relation to the ego agent? Surrounding agent #3 is 4 meters in front and 4 meters on the right of the ego agent.What direction is surrounding agent #3 facing in relation to the ego agent? Surrounding agent #3 is heading in the same direction as the ego agent.What type of agent is surrounding agent #4? Surrounding agent #4 is a vehicle.What is the motion status of surrounding agent #4? Surrounding agent #4 is not moving.Can you describe the position of surrounding agent #4 with respect to the ego agent? Surrounding agent #4 is 9 meters on the right and 1 meter behind the ego agent.In which direction is surrounding agent #4 heading compared to the ego agent? Surrounding agent #4 is heading the opposite direction as the ego agent.What type of agent is surrounding agent #5? Surrounding agent #5 is a vehicle.Is surrounding agent #5 currently in motion? Surrounding agent #5 is not moving.Where is surrounding agent #5 situated in relation to the ego agent? Surrounding agent #5 is 11 meters in front and 2 meters on the right of the ego agent.What direction is surrounding agent #5 facing with respect to the ego agent? Surrounding agent #5 is heading right of the ego agent.What type of agent is surrounding agent #6? Surrounding agent #6 is a vehicle.What is the current speed of surrounding agent #6? Surrounding agent #6 is not moving.Can you describe the position of surrounding agent #6 relative to the ego agent? Surrounding agent #6 is 14 meters in front and 7 meters on the right of the ego agent.In which direction is surrounding agent #6 moving with respect to the ego agent? Surrounding agent #6 is heading right of the ego agent."  
    
    Question: "What interactions are anticipated between the ego agent and surrounding agent #0?"
    Answer: "Surrounding agent #0 will overtake the ego agent as it is accelerating and will be further ahead in the future."

    Question: "Can you predict the interaction between the ego agent and surrounding agent #4?"
    Answer: "There will be no interaction between the ego agent and surrounding agent #4 as they are heading in opposite directions and not affecting each other's path."

    Question: "What is the ego agent's plan for the immediate future?"
    Answer: "The ego agent intends to continue on its current path and lane while accelerating. It will overtake surrounding agent #3 and pass surrounding agents #5 and #6, as they are not moving. It will also be overtaken by surrounding agent #0, which is accelerating on the left side."

    Question: "What will be the nature of the interaction between the ego agent and surrounding agent #6?"
    Answer: "The ego agent will pass surrounding agent #6 since surrounding agent #6 is stationary and the ego agent is accelerating."

    Given these examples now please have a look at the following new context and try to answer the following question:
    Here is the context: {context}

    Here is the question: {question}
    
    """
    #### Step 2: Generate the model grades and add them to the dictionary
    
    for model_family in model_dictionary.keys():
        if model_family=="openai_models":
            for model_name in model_dictionary[model_family]:
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=openai_call(model_name=model_name, prompt=direct_cot_prompt_4shot))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_with_plan", grading_output
                    )
                avg_score = (int(grading_output["Correctness score"]) + int(grading_output["Faithfulness score"]))/2
                existing_grades[scenario_id][interaction_id].setdefault("problem_score_avg", (str(avg_score)))
                qa_scores_o3_mini.append(avg_score)
        elif model_family=="deepinfra_models":
            for model_name in model_dictionary[model_family]:
                grading_prompt = prepare_grading_prompt(context=context, question=question, 
                                       answer=answer, model_output=deepinfra_call(model_name=model_name, prompt=direct_cot_prompt_4shot))
                grading_output = eval(deepinfra_call(model_name="deepseek-ai/DeepSeek-V3", prompt=grading_prompt))
                existing_grades[scenario_id][interaction_id].setdefault(
                    model_family+"_"+model_name+"_with_plan", grading_output
                    )
                avg_score = (int(grading_output["Correctness score"]) + int(grading_output["Faithfulness score"]))/2
                existing_grades[scenario_id][interaction_id].setdefault("problem_score_avg", (str(avg_score)))
                qa_scores_qwen25_7b.append(avg_score)
    


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
    
    plt.bar([i for i in range(len(qa_scores_o3_mini))], qa_scores_o3_mini)
    plt.show()

    plt.bar([i for i in range(len(qa_scores_qwen25_7b))], qa_scores_qwen25_7b)
    plt.show()
main()