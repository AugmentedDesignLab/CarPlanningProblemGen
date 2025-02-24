import json
from guidance import models, gen, user, assistant, system
from openai import OpenAI
import os
from rouge import Rouge

scenario_files = os.listdir("../car_beh_gen/datasets/training.tar/training/training/")

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

def transform_to_mcq(facts, mcq_info):
    client = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"])

    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {"role": "user", "content": f"""
            Here are some facts about a scenario where a self driving vehicle or an autonomous vehicle referrred here as an ego vehicle is driving on the roads:
            {facts}

            Here is some relevant question and answer information pertaining interactions between the vehicles
            {mcq_info}

            Transform each question in the given information into a multiple choice question. The number of multiple choice questions should be equal to the given questions here. 
            
            Here is some information regarding multiple choice questions:
            A multiple-choice question is a collection of three components aimed at testing a student’s understanding of a 
            certain topic, given a particular context of what the student is expected to know. The topic, as well as the context 
            of the topic, will be provided in order to generate effective multiple-choice questions. The three components of a 
            multiple-choice question are as follows: a Stem, a Correct Answer, and two Distractors. There must always be only 
            one correct answer and only two distractors.

            The stem refers to the question the student will attempt to answer, as well as the relevant context necessary in order 
            to answer the question. It may be in the form of a question, an incomplete statement, or a scenario. The stem should 
            focus on assessing the specific knowledge or concept the question aims to evaluate.

            The Correct Answer refers to the correct, undisputable answer to the question in the stem.

            A Distractor is an incorrect answer to the question in the stem and adheres to the following properties. […] Use 
            “None of the Above” or “All of the Above” style answer choices sparingly. These answer choices have been shown to, in
            general, be less effective at measuring or assessing student understanding.

            Multiple-choice questions should be clear, concise, and grammatically correct statements. Make sure the questions are 
            worded in a way that is easy to understand and does not introduce unnecessary complexity or ambiguity. Students should 
            be able to understand the questions without confusion. The question should not be too long, and allow most students to 
            finish in less than five minutes. This means adhering to the following properties.

            Generate the multiple choice questions exactly in the format shown in these examples.
            <open curly bracket. This will be the start of the questions regarding a single scenario>
            "context": <Statements about the facts given initially. Read carefully. They should cover each question and answer. Just write this as a sequence of sentences.>,
            "q_1": <open curly bracket>               
            "difficulty": 9,
            "question": <An exact copy of the question given initially>,
            "a": "...",
            "b": "...",
            "c": "...",
            "correct_answer": <just the option label>
            <close curly bracket>
            <close curly bracket. This reflects the end of the questions regarding a single scenario.>

            Keep in mind that the multiple choice questions must be consistent with the questions provided initially. Nothing else. No tags. No carriage returns. Just the dictionary starting from the curly bracket and ending in the curly bracket.
            I need it to be exactly in that form. No json tags in your output. No json tags in your output. I'm grateful, thank you!

            """},
        ],
        stream=False
    )

    return response.choices[0].message.content


def transform_to_mcq_deepseek(facts, mcq_info):
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    response_ds_v3 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are very intelligent (as usual, not surprising) and helpful."},
            {"role": "user", "content": f"""
            Here are some facts about a scenario where a self driving vehicle or an autonomous vehicle referrred here as an ego vehicle is driving on the roads:
            {facts}

            Here is some relevant question and answer information pertaining interactions between the vehicles
            {mcq_info}

            Transform each question in the given information into a multiple choice question. The number of multiple choice questions should be equal to the given questions here. 
            
            Here is some information regarding multiple choice questions:
            A multiple-choice question is a collection of three components aimed at testing a student’s understanding of a 
            certain topic, given a particular context of what the student is expected to know. The topic, as well as the context 
            of the topic, will be provided in order to generate effective multiple-choice questions. The three components of a 
            multiple-choice question are as follows: a Stem, a Correct Answer, and two Distractors. There must always be only 
            one correct answer and only two distractors.

            The stem refers to the question the student will attempt to answer, as well as the relevant context necessary in order 
            to answer the question. It may be in the form of a question, an incomplete statement, or a scenario. The stem should 
            focus on assessing the specific knowledge or concept the question aims to evaluate.

            The Correct Answer refers to the correct, undisputable answer to the question in the stem.

            A Distractor is an incorrect answer to the question in the stem and adheres to the following properties. […] Use 
            “None of the Above” or “All of the Above” style answer choices sparingly. These answer choices have been shown to, in
            general, be less effective at measuring or assessing student understanding.

            Multiple-choice questions should be clear, concise, and grammatically correct statements. Make sure the questions are 
            worded in a way that is easy to understand and does not introduce unnecessary complexity or ambiguity. Students should 
            be able to understand the questions without confusion. The question should not be too long, and allow most students to 
            finish in less than five minutes. This means adhering to the following properties.

            Generate the multiple choice questions exactly in the format shown in these examples.
            <open curly bracket. This will be the start of the questions regarding a single scenario>
            "context": <Rewrite scenario facts mentioned above as a single paragraph.
            It must not be in the question and answer format, but should be >,
            "q_1": <open curly bracket>               
            "difficulty": 9,
            "question": <An exact copy of the question given initially>,
            "a": "...",
            "b": "...",
            "c": "...",
            "correct_answer": <just the option label>
            <close curly bracket>
            <close curly bracket. This reflects the end of the questions regarding a single scenario.>

            Keep in mind that the multiple choice questions must be consistent with the questions provided initially. Nothing else. No tags. No carriage returns. Just the dictionary starting from the curly bracket and ending in the curly bracket.
            I need it to be exactly in that form. No json tags in your output. No json tags in your output. I'm grateful, thank you!
            """},
        ],
        stream=False
    )

    # response_ds_v3 = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": "You are very intelligent (as usual, not surprising) and helpful."},
    #         {"role": "user", "content": f"""
    #         Now consider the output presented here: {response_ds_r1.choices[0].message.content} 
    #         Format and generate the outputs above exactly in the format shown below.
    #         <open curly bracket. This will be the start of the questions regarding a single scenario>
    #         "context": <An exact copy of the scenario facts mentioned initially>,
    #         "q_1": <open curly bracket>               
    #         "difficulty": 9,
    #         "question": <An exact copy of the question given initially>,
    #         "a": "...",
    #         "b": "...",
    #         "c": "...",
    #         "correct_answer": <just the option label>
    #         <close curly bracket>
    #         <close curly bracket. This reflects the end of the questions regarding a single scenario.>

    #         Keep in mind that the multiple choice questions must be consistent with the questions provided initially. Nothing else. No tags. No carriage returns. Just the dictionary starting from the curly bracket and ending in the curly bracket.
    #         I need it to be exactly in that form. No json tags in your output. No json tags in your output. I'm grateful, thank you!

    #         The output needs to be exactly in the format shown above.
    #         """},
    #     ],
    #     stream=False
    # )

    return response_ds_v3.choices[0].message.content

def evaluate_llm(question, model_name):
    gpt_scenario = models.OpenAI(model=model_name, echo=False)

    with system():
        lm_scenario = gpt_scenario

    with user():
        lm_scenario += f"""
        Given the questions here: 
        {question}

        Choose the correct answer. The option descriptions are only supposed to be present in the question, not the answer. 
        Only write the correct option character label, not the description.
        """

    with assistant():
        lm_scenario += gen("mcq_response", temperature=0.5)
    
    #print("The scenario responses are {}".format(lm_scenario["scenario_response"]))
    return lm_scenario["mcq_response"]

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

def generate_womd_mcq(filenames):
    final_questions = []
    for i in range(len(filenames)):
        womd_datapoint = generate_womd_reasoning_datapoint(filenames[i]) # Put another computation in between here which only uses the answers as a set of facts. 
        facts, mcq_info = process_womd_datapoint_for_mcq_gen(womd_datapoint=womd_datapoint)
        mcq = transform_to_mcq_deepseek(facts, mcq_info)
        final_questions.append(eval(mcq))
    return final_questions

def eval_llm():
   # Obtain a list of MCQs
   question_list = generate_womd_mcq(filenames=scenario_files[0:10]) #6 filenames
   print("Total WOMD-Reasoning questions are {}".format(len(question_list))) 
   question_counter = 0
   correct_answer_counter = 0
   # Loop through the question list, evaluating the response to each dictionary entry (different scenario context)
   for i in range(len(question_list)):
    for j in range(len(question_list[i]["questions"])):
        response = evaluate_llm(question_list[i]["questions"][j]["standalone_scenario"], "gpt-4o")
        question_counter += 1
        print("The total questions asked so far are {}".format(question_counter))
        if str(response)==question_list[i]["questions"][j]["correct_answer"]: #Response is sometimes a string sometimes it is not.
            correct_answer_counter += 1
            print("The total correct answers so far are {}".format(correct_answer_counter))
        else:
            print("The question was {}".format(question_list[i]["questions"][j]["standalone_scenario"]))
            print("The response {} is incorrect. The correct response is {}".format(response, question_list[i]["questions"][j]["correct_answer"]))
    
    print("Final accuracy percentage is {}".format((correct_answer_counter/question_counter)*100))

def obtain_and_write_mcq_data(start, end):
    
    for filename in scenario_files[start:end]:
        mcq_data = {}
        womd_datapoint = generate_womd_reasoning_datapoint(filename=filename)
        id = womd_datapoint['sid']
        facts, mcq_info = process_womd_datapoint_for_mcq_gen(womd_datapoint=womd_datapoint)
        mcq_set = eval(transform_to_mcq_deepseek(facts=facts, mcq_info=mcq_info)) # Dictionary in the same format as the prompt. 
        
        #mcq_set = eval(transform_to_mcq(facts=facts, mcq_info=mcq_info))
        rouge = Rouge()
        reference_context = facts["Facts about the static environment"]+facts["Facts about the ego vehicle in this environment"]+facts["Facts about the agents surrounding the ego vehicle in this environment"]
        generated_context = mcq_set["context"]
        print("Reference context is {}".format(reference_context))
        print("Generated context is {}".format(generated_context))
        
        mcq_dictionary = {}
        mcq_dictionary["Reference_Context"] = reference_context
        mcq_dictionary["Context"] = generated_context
        mcq_dictionary["Interactions"] = {}
        for i in range(len(mcq_info)): #Iterate over the mcqs generated
            mcq = {}
            reference_question = womd_datapoint['int_q'][i]
            generated_question = mcq_set["q_"+str(i+1)]["question"]
             
            reference_answer = womd_datapoint['int_a'][i]
            generated_answer_a = mcq_set["q_"+str(i+1)]["a"]
            generated_answer_b = mcq_set["q_"+str(i+1)]["b"]
            generated_answer_c = mcq_set["q_"+str(i+1)]["c"]
            
            mcq["reference_question"] = reference_question
            mcq["reference_answer"] = reference_answer
            mcq["corresponding_mcq"] = mcq_set
            mcq_dictionary["Interactions"]["Interactions_"+str(i)] = mcq

        mcq_data[str(id)] = mcq_dictionary

        with open("parsed_womdr_data/"+str(id)+".json", 'w') as file:
            json.dump(mcq_data, file, indent=4)
    

#obtain_and_write_mcq_data(0, 1)

