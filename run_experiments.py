# Please make sure that the WSL is installed to run this script. 
# In WSL, please make sure that Anaconda is installed with changes to the .bashrc file in
# the wsl home ~ folder as stated in this project's README. 

import subprocess
import parse_scenario_womd
import planner
import llm_qa

print("Running the data preprocessing .... \nThe data will be in the parsed_womdr_data dictionary.")
parse_scenario_womd.obtain_and_write_mcq_data(3, 4)
print("Completed data preprocessing!\n")

print("""Running the PDDL file generation.\nThe domains and problem files will get saved in the apla-planner/generated_pddls_deepseek/ path
      within the domains and problems folder.""")
planner.generate_pddl_with_syntax_check_deepseek()
print("PDDL problem generation has been completed!\n")

print("Running the planner within WSL... \n")
subprocess.run(["wsl", "-e", "bash", "-ic", "cd apla-planner/generated_pddls_deepseek ; python planner_test.py"], stdout=subprocess.PIPE).stdout
print("Plan generation has been completed!\n")

print("Running the LLM evaluations ... \nThe results will be in the grades folder.")
llm_qa.main()
print("LLM evaluations have been completed!")

