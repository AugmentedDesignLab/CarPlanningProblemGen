import unified_planning as up
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.plot import plot_sequential_plan
import guidance
import sys
import subprocess

from guidance import models, gen, user, assistant, system
import basic_scenario_gpt
import parse_scenario_womd

scenario_data = parse_scenario_womd.new_data_no_interactions
scenario = basic_scenario_gpt.generate_scenario_actions(concepts=basic_scenario_gpt.generate_scenario_concepts(number_of_concepts=3, granularity=6, scenario_data=scenario_data), granularity=2)

gpt = models.OpenAI(model="gpt-4o", echo=False)
solution_found = False

while(True):

    with system():
        lm = gpt

    with user():
        lm += f"""
        We need you to write specific driving behaviors to accomplish certain goals. A behavior is defined as actions taken in response to certain conditions. Conditions are provided as an environment state. 
        Think about some states by yourself that you believe is necessary. 
        Vehicles navigate in the action space and the state space provided to them.

        Now generate a PDDL domain file for the scenario: {scenario}. Domain file only only only for now.
        
        Think about the STRIPS PDDL for different popular domains such as gripper and sokoban.
        Verify whether it all the suggested states and actions makes sense and is correct.
        Then, if it feels correct, write it down as a PDDL domain file. I only only want the PDDL domain file contents.
        
        Keep things simple and clear. Do not repeat names. Do not repeat names. Do not redefine anything. Ensure that everything is very very clear and correct. Check and double check correctness.
        Do not write anything else other than what is asked. Only Only Only write what has been asked. Do not write any ``` statement as well. Only write what has been asked. Only write what has been asked.
        """

    with assistant():
        domain_gen_status=True
        while(domain_gen_status):
            try: 
                lm += gen("domain", temperature=0.5)
                break
            except:
                continue    
        print("Domain draft has been generated!")

    with open("domain0.pddl", "w", encoding='utf-8') as file:
        file.write(lm["domain"]) # We want to read the article as a single string, so that we can feed it to gpt.
        file.close()

    with user():
        lm += f"""
        Now carefully write the PDDL problem file for the corresponding domain file provided:
        {lm["domain"]}.
        First repeat the types, states (predicates) and actions in this file as a list in natural language. 
        Then think step by step about a problem for this domain. Think about whether this problem does indeed have a solution plan.
        Double check that everything is clear and it does in fact have a solution. Then write the PDDL problem file contents. I only want the problem file contents. 
        Do not repeat names. Do not repeat names. Only the problem file contents nothing more. Only the problem file contents nothing more. I'm pasting this in a pddl problem file just letting you know. 
        Do not write anything else other than what is asked. Only Only Only write what has been asked. Do not write any ``` statement as well. Only write what has been asked. Only write what has been asked. Only write what has been asked.
        Do not write any html in the output.
        """
    with assistant():
        problem_gen_status=True
        while(problem_gen_status):
            try: 
                lm += gen("problem", temperature=0.5)
                break
            except:
                continue
        print("Problem draft has been generated!")

    with open("problem0.pddl", "w", encoding='utf-8') as file:
        file.write(lm["problem"]) # We want to read the article as a single string, so that we can feed it to gpt.
        file.close()
    
    with user():
        lm += """
        Consider the PDDL domain generated above, think step by step and 
        carefully write some constructive, insightful analysis as a list that for each of the domain and problem files. 
        Is the domain have a good state space and action space for the given scenario? Is the syntax consistent between the domain and the problem files?
        What needs to be fixed? Do you believe that the domain and problem have a good solution?
        What should be addressed for a solvable domain and problem? Give clear advice and instructions on fixing these issues. Point to the respective parts of the domain and problem files while giving advice.
        """

    with assistant():
        lm+= gen("feedback_initial", temperature=0.5)
    
    print(lm["feedback_initial"])
    
    with user():
        lm += """
        Consider the PDDL domain above and the PDDL problem files generated above. 
        Now based on your feedback and selected domain file, write down a more polished, high quality PDDL domain file. I only only want the PDDL domain file contents.
        
        Keep things very very simple and very very clear. Do not repeat names. Do not repeat names. Do not redefine anything. Ensure that everything is very very clear and correct. Check and double check correctness.
        Do not write anything else other than what is asked. Only Only Only write what has been asked. Do not write any ``` statement as well. Only write what has been asked. Only write what has been asked.
        """

    with assistant():
        domain_gen_status=True
        while(domain_gen_status):
            try: 
                lm += gen("domain_final", temperature=0.1)
                break
            except:
                continue    
        print("Final Domain file has been generated!")

    with open("domain_final.pddl", "w", encoding='utf-8') as file:
        file.write(lm["domain_final"]) # We want to read the article as a single string, so that we can feed it to gpt.
        file.close()
    
    with user():
        lm += """
        Consider the PDDL domains above and the PDDL problems generated above. Consider the feedback provided above as well. Consider the PDDL domain and problems that were selected. 
        Now write the PDDL problem file contents. I only want the updated, polished problem file contents. 
        Do not repeat names. Do not repeat names. Only the problem file contents nothing more. Only the problem file contents nothing more. I'm pasting this in a pddl problem file just letting you know. 
        Do not write anything else other than what is asked. Only Only Only write what has been asked. Do not write any ``` statement as well. Only write what has been asked. Only write what has been asked. Only write what has been asked.
        Do not write any html in the output.
        """
    with assistant():
        problem_gen_status=True
        while(problem_gen_status):
            try: 
                lm += gen("problem_final", temperature=0.1)
                break
            except:
                continue
        print("Final problem file has been generated!")

    with open("problem_final.pddl", "w", encoding='utf-8') as file:
        file.write(lm["problem_final"]) # We want to read the article as a single string, so that we can feed it to gpt.
        file.close()

    #Remove credit streaming from the output
    up.shortcuts.get_environment().credits_stream = None

    try:
        # Run the generated domain and problem files via the Parser application in VAL (BSD 3 License) - https://github.com/KCL-Planning/VAL/tree/master 
        output = subprocess.run(["Parser", "domain_final.pddl", "problem_final.pddl"], stdout=subprocess.PIPE).stdout
        string_output = str(output, encoding='utf-8')
        print(string_output)
        with user():
            lm+=f"""
            The PDDL domain files generated above were given to the popular planning validator VAL. Within this tool, I'm using the PlanRec tool which explains which action comes before or after which action.
            This output is given in {string_output}. Based on this output, can you explain what the output means. Please explain clearly so that I can understand.  
            """
        with assistant():
            lm+=gen("validity_final", temperature=0.2)
        
        # with user():
        #     lm+="""
        #     Now write the updated PDDL domain file contents. I only want the domain file contents. 
        #     Do not repeat names. Do not repeat names. DO NOT write feedback here, write the actual PDDL domain file contents. 
        #     Only the domain file contents nothing more. Only the domain file contents nothing more.
        #     Only Only Only the new updated domain file contents.
        #     Do not write anything else other than what is asked. Only Only Only write what has been asked. Do not write any ``` statement as well. Only write what has been asked. Only write what has been asked. Only write what has been asked.
        #     Do not write any html in the output. 
        #     """
        
        # with assistant():
        #     domain_gen_status=True
        #     while(domain_gen_status):
        #         try: 
        #             lm += gen("updated_domain", temperature=0.05)
        #             break
        #         except:
        #             continue    
        #     print("Planner feedback has been considered and the new domain file has been generated!")

        # with open("updated_domain.pddl", "w", encoding='utf-8') as file:
        #     file.write(lm["updated_domain"]) # We want to read the article as a single string, so that we can feed it to gpt.
        #     file.close()
        
        # with user():
        #     lm += """
        #     Consider the PDDL domains, the PDDL problems and the planner feedback generated above. Now carefully write the PDDL problem file for the corresponding domain file provided by you above.
        #     Write some constructive, insightful feedback that would improve the problem file, with close attention on the planner results. 
        #     Is the problem looking for some important planning skills? What should be addressed in the problem files? Be specific, and focus on the 
        #     problem file contents.
        #     """

        # with assistant():
        #     lm+= gen("problem_critique", temperature=0.2)
        
        # print(lm["problem_critique"])

        # with user():
        #     lm+="""
        #     Now write the PDDL problem file contents. I only want the problem file contents. 
        #     Do not repeat names. Do not repeat names. Only the problem file contents nothing more. Only the problem file contents nothing more. 
        #     Do not write anything else other than what is asked. Only Only Only write what has been asked. Do not write any ``` statement as well. Only write what has been asked. Only write what has been asked. Only write what has been asked.
        #     Do not write any html in the output.
        #     """
        # with assistant():
        #     problem_gen_status=True
        #     while(problem_gen_status):
        #         try: 
        #             lm += gen("problem_after_planner_feedback", temperature=0.05)
        #             break
        #         except:
        #             continue
        #     print("Problem after planner feedback has been generated!")

        # with open("problem_after_planner_feedback.pddl", "w", encoding='utf-8') as file:
        #     file.write(lm["problem_after_planner_feedback"]) # We want to read the article as a single string, so that we can feed it to gpt.
        #     file.close()
        
        # problem=reader.parse_problem("updated_domain.pddl", "problem_after_planner_feedback.pddl")
        # result = planner.solve(problem)
        # print("The round 2 results for the planner are...")
        # print(result)
        

    except:
        print("Exception! Trying again...")
        continue # Try again

    # Solution found, break the loop
    break






