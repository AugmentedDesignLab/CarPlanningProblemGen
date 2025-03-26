import subprocess

print("Running the planner within WSL... \n")
subprocess.run(["wsl", "-e", "bash", "-ic", "cd apla-planner/generated_pddls_deepseek ; python planner_test.py"], stdout=subprocess.PIPE).stdout
print("Plan generation has been completed!\n")