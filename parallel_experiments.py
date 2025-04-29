import subprocess

print("Running nshot prompting experiments \n")
print("All scripts requested will run in the background and will return results in the grades folder soon!")

scenario_index_list_small = [239, 562, 999, 2827, 475]
scenario_index_list_medium = [6, 254, 622, 136, 182]
scenario_index_list_large = [52, 13, 41, 102, 600]

scenario_indices_all = scenario_index_list_small + scenario_index_list_medium + scenario_index_list_large

for index in scenario_index_list_large:
    process_1 = subprocess.Popen(["C:\\Users\\ishaa\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\ishaa\\anaconda3",
                    "&&", "conda", "activate", "car_beh_gen",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "0shot",
                    "--scenario_index", str(index), "--ground_truth", "False",
                    "--enable_basic_refined_prompt", "False", "--enable_lecturing_refined_prompt", "True"])
    process_2 = subprocess.Popen(["C:\\Users\\ishaa\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\ishaa\\anaconda3",
                    "&&", "conda", "activate", "car_beh_gen",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "2shot",
                    "--scenario_index", str(index), "--ground_truth", "False",
                    "--enable_basic_refined_prompt", "False", "--enable_lecturing_refined_prompt", "True"])
    process_3 = subprocess.Popen(["C:\\Users\\ishaa\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\ishaa\\anaconda3",
                    "&&", "conda", "activate", "car_beh_gen",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "4shot",
                    "--scenario_index", str(index), "--ground_truth", "False",
                    "--enable_basic_refined_prompt", "False", "--enable_lecturing_refined_prompt", "True"])
    process_4 = subprocess.Popen(["C:\\Users\\ishaa\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\ishaa\\anaconda3",
                    "&&", "conda", "activate", "car_beh_gen",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "6shot",
                    "--scenario_index", str(index), "--ground_truth", "False",
                    "--enable_basic_refined_prompt", "False", "--enable_lecturing_refined_prompt", "True"])