import json
from openai import OpenAI
import os
import re

scenario_files = os.listdir("../car_beh_gen/datasets/training.tar/training_2/training/")
scenario_blocklist = []

def generate_womd_reasoning_datapoint(filename):
    print("File size is {}".format(os.path.getsize('../car_beh_gen/datasets/training.tar/training_2/training/'+filename)))
    with open('../car_beh_gen/datasets/training.tar/training_2/training/'+filename, 'r') as file:
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
        environment_facts += " "
    
    for index in range(len(womd_datapoint['ego_q'])):
        ego_facts += womd_datapoint['ego_q'][index]
        ego_facts += " "
        ego_facts += womd_datapoint['ego_a'][index]
        ego_facts += " "
    
    for index in range(len(womd_datapoint['sur_q'])):
        surr_facts += womd_datapoint['sur_q'][index]
        surr_facts += " "
        surr_facts += womd_datapoint['sur_a'][index]
        if index != (len(womd_datapoint['sur_q']) - 1):
            surr_facts += " "

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


def obtain_and_write_data_smallest(start, end):
    filesize = 100000
    smallest_filename = ""
    for filename in scenario_files[start:end]:
        file_size = os.path.getsize('../car_beh_gen/datasets/training.tar/training_2/training/'+filename)
        if file_size < filesize: 
            filesize = file_size
            smallest_filename = filename
    #for filename in scenario_files[start:end]:
    blocklist_match = False
    final_preprocessed_data = {}
    womd_datapoint = generate_womd_reasoning_datapoint(filename=smallest_filename)
    id = womd_datapoint['sid']

    # Add bad scenarios to the blocklist
    for blocklist_id in scenario_blocklist:
        if blocklist_id==id: 
            blocklist_match = True
    if blocklist_match==True:
        return #skip this iteration
    
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

def transform_datapoint_to_qa_list(wmo_reasoning_datapoint):
    combined_qa_list = [] # A list that will combine all the QA lists in one datapoint. This 
    # will result in one large list of strings. 
            
    datapoint_qa_keys = ["env_q", "env_a", "ego_q", "ego_a", "sur_q", "sur_a", "int_q", "int_a"]
    for qa_key in datapoint_qa_keys: combined_qa_list += wmo_reasoning_datapoint[qa_key]

    return combined_qa_list 


# Find scenarios similar to the given scenario
def find_similar_data(start, end, search_range_start, search_range_end):
    # Ensure that only one scenario is used as the reference for comparison.
    if end-start>1: return

    highest_similarity_score = 0
    highest_similarity_index = 0
    for filename in scenario_files[start:end]:
        datapoint_reference = generate_womd_reasoning_datapoint(filename=filename)
        combined_qa_list_reference = transform_datapoint_to_qa_list(datapoint_reference)
        
        # Within the datapoint above, we have keys about the following concepts:
        # scene id, ego id, start time, end time, main dataset ids for surrounding agents, current dataset ids for 
        # surrounding agents. env_q, env_a for environment questions. ego_q, ego_a for ego questions.
        # sur_q, sur_a for surrounding agent questions. int_q and int_a for interaction related questions. 
        # It is observed that the questions and answers are NOT consistent across all datapoints. 

        # Using the search_range_start and search_range_end indices, we will search all these scenario file indices to find
        # the most similar scenario. Firstly, the questions under each category need to be similar and then the corresponding answers need to be similar.
        # We will compute a match score based on the following factors:
        # number of elements in the current dataset ids list mentioned above. If it's within +-2, we add one point.
        # QA similarity for each of the 3 question categories. Add one point for each question being matched and each answer being matched.  

        number_of_surrounding_reference = len(datapoint_reference["rel_qa_id"])
        scenario_similarity_score_collection = {} # IDs mapped to similarity scores.
        scenario_similarity_score_search_candidate = 0
        
        for i in range(len(scenario_files[search_range_start: search_range_end])):
            datapoint_search_candidate = generate_womd_reasoning_datapoint(filename=scenario_files[search_range_start+i])
            number_of_surrounding_search_candidate = len(datapoint_search_candidate["rel_qa_id"])

            # Check if the number of surrounding agents in the search candidate are close enough to the reference scenario.
            if abs((number_of_surrounding_reference - number_of_surrounding_search_candidate)) <= 2: scenario_similarity_score_search_candidate += 1

            combined_qa_list = transform_datapoint_to_qa_list(datapoint_search_candidate)

            # Turn the search candidate into one string where we can search for similarities.
            combined_qa_search_candidate = ""
            combined_qa_search_candidate.join(combined_qa_list)

            # Using the regex library to search for matching patterns.
            # Iterate through each question and answer in the reference and add a point each time there is a string match in the candidate.  
            for search_term in combined_qa_list_reference:
                if re.search(search_term, combined_qa_search_candidate) is not None: scenario_similarity_score_search_candidate += 1

            # Modifying the highest score value
            if scenario_similarity_score_search_candidate > highest_similarity_score: 
                highest_similarity_score = scenario_similarity_score_search_candidate
                highest_similarity_index = search_range_start + i
            
            scenario_similarity_score_collection.setdefault("Index_number_"+str(search_range_start+i), [datapoint_search_candidate["sid"], str(scenario_similarity_score_search_candidate)])

    with open("scenario_similarity_ref_"+str(start)+"_"+str(end)+"_search_"+str(search_range_start)+"_"+str(search_range_end)+".json", 'w') as file:
        json.dump(scenario_similarity_score_collection, file, indent=4)
    
    print("\n The highest similarity index is {}".format(highest_similarity_index))
    print("\n The highest similarity score for this index is {}".format(highest_similarity_score))

    return scenario_similarity_score_collection



def obtain_and_write_data(start, end):
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
        preprocessed_data["Size"] = os.path.getsize('../car_beh_gen/datasets/training.tar/training_2/training/'+filename)
        preprocessed_data["Context"] = reference_context

        context_word_count = len(reference_context.split(" "))
        #print(reference_context.split(" "))
        preprocessed_data["Word Count"] = context_word_count
        #total_word_count_sentence = 0
        # for sentence_index in range(len(context_sentence_list)): 
        #          total_word_count_sentence += len(context_sentence_list[sentence_index].split(" "))
            
        # average_word_count_sentence = total_word_count_sentence / len(context_sentence_list)
        #print(average_word_count_sentence)
        #print(context_word_count)
        if preprocessed_data['Size'] > 10000:
            preprocessed_data['Scenario Category'] = "C"
        elif 6000 <= preprocessed_data['Size'] < 9999:
            preprocessed_data['Scenario Category'] = "B"
        elif preprocessed_data['Size'] < 6000:
            preprocessed_data['Scenario Category'] = "A"

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

obtain_and_write_data(482, 483)

