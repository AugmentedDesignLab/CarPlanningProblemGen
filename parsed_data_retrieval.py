import os
import json

def retrieve_parsed_data():

    parsed_womdr_files = os.listdir("parsed_womdr_data/")
    scenario_domain_problem_data = {}

    for i in parsed_womdr_files:
        with open("parsed_womdr_data/"+i, 'r') as scenario_file:
            scenario_data = json.load(scenario_file) 
            print(f"number of scenarios are {scenario_data.keys()}")
            current_scenario_id = i[-21:-5]
            for key in scenario_data.keys():
                # Indices here have been planned based on the Waymo Reasoning dataset files
                scenario_domain_problem_data.setdefault(current_scenario_id, {
                    "Context": ""
                })
                scenario_domain_problem_data[current_scenario_id].setdefault("Scenario Index", scenario_data[key]["Scenario Index"])
                scenario_domain_problem_data[current_scenario_id]["Context"] = scenario_data[key]["Context"]
                scenario_domain_problem_data[current_scenario_id]["Word Count"] = scenario_data[key]["Word Count"]
                #print(f"number of interactions in this scenario are {scenario_data[key]["Interactions"].keys()}")
                for interaction_key in scenario_data[key]["Interactions"].keys():
                    scenario_domain_problem_data[current_scenario_id].setdefault("Interactions", {})
                    scenario_domain_problem_data[current_scenario_id]["Interactions"].setdefault(interaction_key, {
                    "problem_data": "",
                    "answer_data": ""
                    }) 
                    scenario_domain_problem_data[current_scenario_id]["Interactions"][interaction_key]["problem_data"] = scenario_data[key]["Interactions"][interaction_key]["reference_question"]
                    scenario_domain_problem_data[current_scenario_id]["Interactions"][interaction_key]["answer_data"] = scenario_data[key]["Interactions"][interaction_key]["reference_answer"]
       
    return scenario_domain_problem_data