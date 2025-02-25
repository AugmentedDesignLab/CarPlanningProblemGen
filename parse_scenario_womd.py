import json
from guidance import models, gen, user, assistant, system
from openai import OpenAI
import os
from rouge import Rouge

scenario_files = os.listdir("../car_beh_gen/datasets/training.tar/training/training/")
scenario_blocklist = ['3e9622a454291617']

def generate_womd_reasoning_datapoint(filename):
    with open('../car_beh_gen/datasets/training.tar/training/training/'+filename, 'r') as file:
        data = json.loads(file.read())
        new_data_no_interactions = {
            'environment questions': data['env_q'],
            'environment answers': data['env_a'],
            'ego vehicle questions': data['ego_q'],
            'ego_vehicle answers': data['ego_a'],
            'surronding vehicle questions': data['sur_q'],
            'surronding vehicle answers': data['sur_a']}  
        
        new_data_interactions = {
            'vehicle interaction questions': data['int_q'],
            'vehicle interaction answers': data['int_a']}
    
    return data


def process_womd_datapoint_for_mcq_gen(womd_datapoint):
    environment_facts = ''
    ego_facts = ''
    surr_facts = ''
    
    for index in range(len(womd_datapoint['env_q'])):
        environment_facts += womd_datapoint['env_q'][index]
        environment_facts += " "
        environment_facts += womd_datapoint['env_a'][index]
    
    for index in range(len(womd_datapoint['ego_q'])):
        ego_facts += womd_datapoint['ego_q'][index]
        ego_facts += " "
        ego_facts += womd_datapoint['ego_a'][index]
    
    for index in range(len(womd_datapoint['sur_q'])):
        surr_facts += womd_datapoint['sur_q'][index]
        surr_facts += " "
        surr_facts += womd_datapoint['sur_a'][index]

    facts = {
                "Facts about the static environment": environment_facts,
                "Facts about the ego vehicle in this environment": ego_facts,
                "Facts about the agents surrounding the ego vehicle in this environment": surr_facts 
            
            } # The facts set up the scenario for us. The MCQs are generated regarding the interactions.
    
    mcq_qa_information = []
    
    # Sequence of questions. Each question is a single string. 
    for i in range(len(womd_datapoint['int_q'])):
        mcq_qa_information.append(womd_datapoint['int_q'][i])

    return facts, mcq_qa_information


def obtain_and_write_mcq_data(start, end):
    for filename in scenario_files[start:end]:
        blocklist_match = False
        final_preprocessed_data = {}
        womd_datapoint = generate_womd_reasoning_datapoint(filename=filename)
        id = womd_datapoint['sid']

        # Add bad scenarios to the blocklist
        for blocklist_id in scenario_blocklist:
            if blocklist_id==id: 
                blocklist_match = True
        if blocklist_match==True:
            continue #skip this iteration
        
        facts, mcq_info = process_womd_datapoint_for_mcq_gen(womd_datapoint=womd_datapoint)
        reference_context = facts["Facts about the static environment"]+facts["Facts about the ego vehicle in this environment"]+facts["Facts about the agents surrounding the ego vehicle in this environment"]
        preprocessed_data = {}
        preprocessed_data["Context"] = reference_context
        preprocessed_data["Interactions"] = {}
        for i in range(len(mcq_info)): #Iterate over the mcqs generated
            original_qa_data = {}
            reference_question = womd_datapoint['int_q'][i]
            reference_answer = womd_datapoint['int_a'][i]
             
            original_qa_data["reference_question"] = reference_question
            original_qa_data["reference_answer"] = reference_answer
            preprocessed_data["Interactions"]["Interactions_"+str(i)] = original_qa_data

        final_preprocessed_data[str(id)] = preprocessed_data

        with open("parsed_womdr_data/"+str(id)+".json", 'w') as file:
            json.dump(final_preprocessed_data, file, indent=4)

