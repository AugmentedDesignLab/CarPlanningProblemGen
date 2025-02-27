from jupyddl import AutomatedPlanner # Comment this line along with the other planner lines if running from outside WSL
import os
import json

## There is one context per scenario. Each context has a corresponding PDDL domain file.
## Each scenario has multiple interactions. Each interaction will have one PDDL problem file. 
## This script should be run via WSL with all the APLA toolbox Python PDDL installation instructions fulfilled.
## Also enable the conda environment with the pip library installed and one having Python 3.7.5 as written in the APLA toolbox PythonPDDL Github page.

domain_folder_list = os.listdir('dataset/domains')
problem_folder_list = os.listdir('dataset/problems')

# Generate two lists - domain file list and problem file list for a single scenario
# Reuse code in terms of classes and functions and 

for scenario_folder in domain_folder_list:
    scenario_folder_domains_complete_path = 'dataset/domains/'+scenario_folder
    scenario_folder_problems_complete_path = 'dataset/problems/'+scenario_folder
    domains_within_scenario = os.listdir(scenario_folder_domains_complete_path)
    problems_within_scenario = os.listdir(scenario_folder_problems_complete_path)

    # We will traverse the problem list since there will be only one domain per scenario

    plans_for_one_scenario = {}
    problem_coverage_scores = []
    problem_initial_state_sizes = []
    print("Scenario ID is {}".format(scenario_folder))
    for problem_file_name in problems_within_scenario:
        problem_full_path = "dataset/problems/"+scenario_folder+"/"+problem_file_name
        domain_full_path = "dataset/domains/"+scenario_folder+"/"+domains_within_scenario[0]
        print("Planner is now running for the problem {}".format(problem_file_name))
        try:
            planner = AutomatedPlanner(domain_full_path, problem_full_path)
            print("the planner is {}".format(planner))
            path = planner.breadth_first_search()
            # Path is a tuple with a list object as the first element. 
            plans_for_one_scenario.setdefault(problem_file_name, str(planner.get_actions_from_path(path[0])))
        except:
            continue


    with open("dataset/problems/"+scenario_folder+"/plan_set.json", 'w') as file:
        json.dump(plans_for_one_scenario, file)
