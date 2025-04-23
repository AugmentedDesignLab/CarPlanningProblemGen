import subprocess

print("Running nshot prompting experiments \n")
print("All scripts requested will run in the background and will return results in the grades folder soon!")

scenario_index_list_small = [239]
scenario_index_list_medium = [6, 254, 622, 136, 182]
scenario_index_list_large = [52, 13, 41, 102, 600]
scenario_index_list_all = scenario_index_list_small + scenario_index_list_medium + scenario_index_list_large

for index in scenario_index_list_small:
    process_1 = subprocess.Popen(["C:\\Users\\denis\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\denis\\anaconda3",
                    "&&", "conda", "activate", "womdr_conda",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "0shot",
                    "--scenario_index", str(index)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_2 = subprocess.Popen(["C:\\Users\\denis\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\denis\\anaconda3",
                    "&&", "conda", "activate", "womdr_conda",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "2shot",
                    "--scenario_index", str(index)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_3 = subprocess.Popen(["C:\\Users\\denis\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\denis\\anaconda3",
                    "&&", "conda", "activate", "womdr_conda",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "4shot",
                    "--scenario_index", str(index)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_4 = subprocess.Popen(["C:\\Users\\denis\\anaconda3\\Scripts\\activate.bat", "C:\\Users\\denis\\anaconda3",
                    "&&", "conda", "activate", "womdr_conda",
                    "&&", "python", "llm_qa_direct_only.py", "--nshot", "6shot",
                    "--scenario_index", str(index)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)