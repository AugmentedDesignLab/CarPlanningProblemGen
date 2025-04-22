import json
from openai import OpenAI
import os
import re
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pathlib import Path

scenario_files = os.listdir("../training/")
scenario_blocklist = []

def generate_womd_reasoning_datapoint(filename):
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

# Function to implement k nearest neighbour prompting 
# The notion of nearest is based on the OpenAI and Gemini embedding representation of the 
# (1) The context (2) Both the context and one interaction question.  
def find_similar_data_emb_based(search_range_start=100, 
                                search_range_end=200,
                                consider_scenario_context=True,
                                consider_scenario_interactions=True,
                                embedding_model_type="small",
                                number_of_clusters=5):
    
    client = OpenAI()
    if embedding_model_type==("small"):
       embedding_dimensions = 1536
       embedding_model_name = "text-embedding-3-small"
    elif embedding_model_type==("large"):
        embedding_dimensions = 3072
        embedding_model_name = "text-embedding-3-large"
    
    # Storing embeddings in a single numpy array for dimensionality reduction later.
    embedding_vector_array = np.zeros((len(scenario_files[search_range_start: search_range_end]), embedding_dimensions)) # text embedding small generates embedding vectors of the dimension of 1536

    for i in tqdm(range(len(scenario_files[search_range_start: search_range_end]))):
        datapoint_search_candidate = generate_womd_reasoning_datapoint(filename=scenario_files[search_range_start+i])
        
        facts, qa_info = process_womd_datapoint_for_mcq_gen(datapoint_search_candidate)
        context_and_interactions = ""
        
        if consider_scenario_interactions==False and consider_scenario_context==False: 
            print("No scenario information to work with, please try again!")
            break

        if consider_scenario_context==True:
            context_and_interactions += facts["Facts about the static environment"]
            context_and_interactions += facts["Facts about the ego vehicle in this environment"]
            context_and_interactions += facts["Facts about the agents surrounding the ego vehicle in this environment"]
        elif consider_scenario_interactions==True:
            for i in range(len(datapoint_search_candidate['int_q'])):
                context_and_interactions += datapoint_search_candidate['int_q'][i]
                context_and_interactions += datapoint_search_candidate['int_a'][i]
        
        emb_response = client.embeddings.create(
            input=context_and_interactions,
            model=embedding_model_name # Experiment with the large embedding model as well as the Gemini model.
        )
        embedding_vector = emb_response.data[0].embedding
        embedding_vector_array[i] = embedding_vector
    
    # embedding_vector_array[len(scenario_files[search_range_start: search_range_end])] = ref_scenario_embedding_final #n-1th index.

    # Reduce dimensionality using PCA. This allows for observations of the embeddings on a graph. 
    pca = PCA(n_components=2)
    low_dim_embeddings = pca.fit_transform(embedding_vector_array)

    plot_x_list = []
    plot_y_list = []
    
    print(f"\n Low dimension embeddings are {len(low_dim_embeddings)}\n") # Should be 1 greater than high dimension embeddings if we are adding the reference scenario from above as well.
    print(f"\n High dimension embeddings rows are {len(scenario_files[search_range_start: search_range_end])}\n")
     
    # K-means clustering
    kmeans = KMeans(n_clusters=number_of_clusters, init="k-means++", random_state=42)
    kmeans.fit(low_dim_embeddings) # K-means clustering of the low dimension representation of the embeddings
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    color_list = ["blue", "orange", "green", "red", "purple"]

    # Low dimension embeddings will have corresponding indices compared to high dimension embeddings.
    for i in range(len(low_dim_embeddings)):
        plt.scatter(low_dim_embeddings[i][0], low_dim_embeddings[i][1], color=color_list[labels[i]])
        plt.annotate(text=str(search_range_start+i), 
                    xy=(low_dim_embeddings[i][0], low_dim_embeddings[i][1]),
                    xytext=(low_dim_embeddings[i][0]+0.001, low_dim_embeddings[i][1]+0.001),
                    fontname='monospace',
                    fontsize='xx-small')

    plt.scatter([i[0] for i in cluster_centers], [i[1] for i in cluster_centers], marker="X")
    plt.title(f"Embeddings between the indices {search_range_start} and {search_range_end}")
    plt.show()
        

# Find scenarios similar to the given scenario
def find_similar_data(scenario_index, search_range_start, search_range_end):
    # Ensure that only one scenario is used as the reference for comparison.

    highest_similarity_score = 0
    second_highest_similarity_score = 0

    highest_similarity_index = 0
    second_highest_similarity_index = 0

    lowest_similarity_score = 10000
    second_lowest_similarity_score = 10000

    lowest_similarity_index = 0
    second_lowest_similarity_index = 0

    for filename in scenario_files[scenario_index: scenario_index+1]:
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
            if i==scenario_index: continue
            datapoint_search_candidate = generate_womd_reasoning_datapoint(filename=scenario_files[search_range_start+i])
            number_of_surrounding_search_candidate = len(datapoint_search_candidate["rel_qa_id"])

            # Check if the number of surrounding agents in the search candidate are close enough to the reference scenario.
            if abs((number_of_surrounding_reference - number_of_surrounding_search_candidate)) <= 2: scenario_similarity_score_search_candidate += 1

            combined_qa_list = transform_datapoint_to_qa_list(datapoint_search_candidate)

            # Turn the search candidate into one string where we can search for similarities.
            combined_qa_search_candidate = ""
            combined_qa_search_candidate = combined_qa_search_candidate.join(combined_qa_list)

            # Using the regex library to search for matching patterns.
            # Iterate through each question and answer in the reference and add a point each time there is a string match in the candidate.  
            for search_term in combined_qa_list_reference:
                matches = re.search(search_term, combined_qa_search_candidate)
                
                # Add one point each time a match to either a question or an answer is provided.
                if (matches is not None): 
                    scenario_similarity_score_search_candidate += 1

            # Modifying the highest score value
            if scenario_similarity_score_search_candidate > highest_similarity_score: 
                second_highest_similarity_score = highest_similarity_score
                second_highest_similarity_index = highest_similarity_index

                highest_similarity_score = scenario_similarity_score_search_candidate
                highest_similarity_index = search_range_start+i
            
            # Modifying the lowest score value
            elif scenario_similarity_score_search_candidate < lowest_similarity_score:
                second_lowest_similarity_score = lowest_similarity_score
                second_lowest_similarity_index = lowest_similarity_index

                lowest_similarity_score = scenario_similarity_score_search_candidate
                lowest_similarity_index = search_range_start+i
            
            # Add to dictionary anyway
            scenario_similarity_score_collection.setdefault("Index_number_"+str(search_range_start+i), [datapoint_search_candidate["sid"], str(scenario_similarity_score_search_candidate)])
            scenario_similarity_score_search_candidate = 0

    with open("scenario_similarity_ref_"+str(scenario_index)+"_search_"+str(search_range_start)+"_"+str(search_range_end)+".json", 'w') as file:
        json.dump(scenario_similarity_score_collection, file, indent=4)
    
    print("\n The highest similarity index is {}".format(highest_similarity_index))
    print("\n The highest similarity score for this index is {}".format(highest_similarity_score))

    print("\n The second highest similarity index is {}".format(second_highest_similarity_index))
    print("\n The second highest similarity score for this index is {}".format(second_highest_similarity_score))

    obtain_and_write_data_single_scenario(highest_similarity_index)
    print("\n This index has been parsed and is in the parsed/... folder")

    obtain_and_write_data_single_scenario(second_highest_similarity_index)
    print("\n This index has also been parsed and is in the parsed/... folder")

    print("\n The lowest similarity index is {}".format(lowest_similarity_index))
    print("\n The lowest similarity score for this index is {}".format(lowest_similarity_score))

    print("\n The second lowest similarity index is {}".format(second_lowest_similarity_index))
    print("\n The second lowest similarity score for this index is {}".format(second_lowest_similarity_score))

    return scenario_similarity_score_collection

def obtain_and_write_data(start, end):
    for i in range(len(scenario_files[start:end])):
        blocklist_match = False
        final_preprocessed_data = {}
        womd_datapoint = generate_womd_reasoning_datapoint(filename=scenario_files[start+i])
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
        preprocessed_data["Size"] = os.path.getsize('../training/'+scenario_files[start+i])
        preprocessed_data["Context"] = reference_context

        context_word_count = len(reference_context.split(" "))
        preprocessed_data["Word Count"] = context_word_count
        preprocessed_data["Scenario Index"] = start+i
        
        if preprocessed_data['Size'] > 10000:
            preprocessed_data['Scenario Category'] = "C"
        elif 6000 <= preprocessed_data['Size'] <= 9999:
            preprocessed_data['Scenario Category'] = "B"
        elif preprocessed_data['Size'] < 6000:
            preprocessed_data['Scenario Category'] = "A"

        preprocessed_data["Interactions"] = {}
        for mcq_info_i in range(len(mcq_info)): #Iterate over the mcqs generated
            original_qa_data = {}
            reference_question = womd_datapoint['int_q'][mcq_info_i]
            reference_answer = womd_datapoint['int_a'][mcq_info_i]
                
            original_qa_data["reference_question"] = reference_question
            original_qa_data["reference_answer"] = reference_answer
            preprocessed_data["Interactions"]["Interactions_"+str(mcq_info_i)] = original_qa_data

        final_preprocessed_data[str(id)] = preprocessed_data

        # Do not change the id and json extension at the end since this is used by the parser
        with open("parsed_womdr_data/"+"scenario_index_"+str(start+i)+"_scenario_id_"+str(id)+".json", 'w') as file:
            json.dump(final_preprocessed_data, file, indent=4)

# In case you're working with one scenario index at a time.
def obtain_and_write_data_single_scenario(scenario_index):
    obtain_and_write_data(scenario_index, scenario_index+1)

# Comment out lines as necessary

# find_similar_data(254, 400, 1000) # This includes the obtain function below btw

# From the initial 60 experiments:

scenario_index_list_small = [239, 562, 999, 2827, 475]
scenario_index_list_medium = [6, 254, 622, 136, 182]
scenario_index_list_large = [52, 13, 41, 102, 600]

scenario_indices_all = scenario_index_list_small+scenario_index_list_medium+scenario_index_list_large


for scenario_index in scenario_index_list_small:
    obtain_and_write_data_single_scenario(scenario_index)


# find_similar_data_emb_based(search_range_start=1000, 
#                             search_range_end=1200,
#                             consider_scenario_context=True,
#                             consider_scenario_interactions=False,
#                             embedding_model_type="small",
#                             number_of_clusters=5)