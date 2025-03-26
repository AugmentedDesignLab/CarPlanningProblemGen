# Script to define functions that return pddl prompts
from client_model_setup import ProvidedLLM
from pathlib import Path
import pddlpy
import json

class PDDLGen():
    def __init__(self, client, model):
        self.provided_llm = ProvidedLLM()
        self.pddl_domain = ""
        self.pddl_problem = ""
        self.client = client
        self.model = model

    def llm_call(self, prompt, dictionary_mode=False, output_thoughts=False):
        output_, thoughts = self.provided_llm.llm_call(client=self.client, model=self.model, prompt=prompt)
        if output_thoughts==True:
            return thoughts
        else: 
            if dictionary_mode==True: # Asked LLM to generate dictionary output in the prompt. 
                output = eval(output_)
                return output
        return output_
    
    def write_pddls(self, write_domain=False, write_problem=False, 
                    scenario_id="", interaction_id="",
                    domain_info="",
                    problem_info=""):
        attempted_overwrite = False
        dir_path_text = "apla-planner/generated_pddls_deepseek/dataset/domains/"+scenario_id
        if write_domain==True:
            pddl_domain_path = dir_path_text+"/domain_deepseek_chat_"+scenario_id+".pddl"
            try: 
                dir_path = Path(dir_path_text)
                dir_path.mkdir()
                with open(pddl_domain_path, "w", encoding='utf-8') as file:
                    file.write(domain_info) # We want to read the article as a single string, so that we can feed it to gpt.
                    file.close()
            except FileExistsError:
                print(f"""
                      Attempted domain file overwrite for scenario id {scenario_id}.
                      This is to reduce repitition in PDDL generation. If you want to regenerate domain files for this scenario,
                      please delete the domains folder for this scenario first. 

                      Skipping PDDL gen for scenario id {scenario_id}
                      """)
                attempted_overwrite=True
                return attempted_overwrite
        elif write_problem==True:
            dir_path_text_problem = "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_id
            pddl_problem_path = dir_path_text_problem+"/problem_deepseek_chat_"+interaction_id+".pddl"
            print("PDDL problem path is {}".format(pddl_problem_path))
            try:
                # Try creating folder if it doesn't exist. Create only file if it does. 
                dir_path_problem = Path(dir_path_text_problem)
                dir_path_problem.mkdir()
                with open(pddl_problem_path, "w", encoding='utf-8') as file:
                        file.write(problem_info) # We want to read the article as a single string, so that we can feed it to gpt.
                        file.close()
            except FileExistsError:
                with open(pddl_problem_path, "w", encoding='utf-8') as file:
                        file.write(problem_info) # We want to read the article as a single string, so that we can feed it to gpt.
                        file.close()
        return attempted_overwrite
    
    def generate_action_prompt(self, scenario_domain_problem_data_context):
        action_prompt = f""" 
            Based on the information detailed in {scenario_domain_problem_data_context}, 
            * Write down a list of actions that map between states in natural language. 
            * Each action has some causal states (predicates) and some effect states that will be true or false.
            * Each action has a strong logical connection between any number of causal states and any number of effect states.
            * States in an action description must not contradict each other.
            * Action names must be descriptive and the action can be understood just by looking at the name.
            * The state names within each action are also descriptive. The cause and effect statements and the state names must have the same information.
            * There must be separate states regarding the environment, ego and the respective surrounding agents.
            * In each action and state, the ego agent or the surrounding agent must be identified as <EGO> or <SURROUNDING AGENT #0> or <SURROUNDING AGENT #1> as needed.
            * For distances, positions and speeds do not use specific numbers but words instead such as front, left, right, near, far, fast, slow, medium (or combinations such as front-left and so on) or other similar descriptive words. 
            * The action itself will only become true when the causal states and the effect states are in the specific states that this description details.
            * Write them in the following format:  
            <open curly bracket>
                "<action name>": 
                <open curly bracket> 
                    "<state name>": <open curly bracket> 
                        "statement": "<the assertion in natural language. Use the fewest words possible for maximum clarity>
                        "value": <Whether this value is true for false>,
                        "state_type": <whether this state is a cause or effect for the current action>
                    <close curly bracket>, 
                    "<state name>": <curly bracket> 
                        "statement": "<the assertion in natural language. Use the fewest words possible for maximum clarity>
                        "value": <Whether this value is true for false>,
                        "state_type": <whether this state is a cause or effect for the current action>
                    <close curly bracket>
                <close curly bracket>, 
                ... 
            <close curly bracket>

            No json tags to be used. Just the dictionary in the output. Nothing else, nothing else, nothing else.   
            """
        return action_prompt
        
    
    def generate_domain_prompt(self, scenario_domain_problem_data_context, generated_actions):
        domain_prompt = f"""
                I have an autonomous vehicle test scenario described here: {scenario_domain_problem_data_context}. I want you to formalize the 
                information to more explicitly write the logic regarding the various driving behaviors in this information. Regarding this information, I first generated 
                action descriptions here: {generated_actions} 
                
                Now for these action descriptions, please generate a PDDL (Planning Domain Definition Language) domain file. I only want the contents that would be in
                such a file, no other information in your writing. Keep in mind that this content will be entered into a file with a .pddl extension and saved.
                Please ensure that all the generated states and actions are absolutely correct with respect to the given information.
                
                Please ensure that everything is very clear and correct. Please make use of good names that are readable. Please check and double check your work before writing.
                No pddl, lisp or any other tags to be used. Just the pddl lines in the output. Please do not write anything else other than precisely what has been asked. 
                """
        return domain_prompt
    
    def generate_pddl_domain(self, scenario_domain_problem_data_context, scenario_id):
        action_prompt = self.generate_action_prompt(scenario_domain_problem_data_context=scenario_domain_problem_data_context)
        action_json = self.llm_call(action_prompt)
        print("Action json is {}".format(action_json))
        domain_prompt = self.generate_domain_prompt(scenario_domain_problem_data_context=scenario_domain_problem_data_context, generated_actions=action_json)
        self.pddl_domain = self.llm_call(domain_prompt)
        print("PDDL domain is {}".format(self.pddl_domain))
        attempted_overwrite = self.write_pddls(write_domain=True, domain_info=self.pddl_domain, scenario_id=scenario_id)
        return self.pddl_domain, attempted_overwrite, action_json

    def generate_initial_problem_prompt(self, scenario_domain_problem_data_context,
                                        generated_actions,
                                        scenario_domain_problem_data_problem_data, domain):
        problem_initial_prompt = f"""
            I have an autonomous vehicle test scenario described here: {scenario_domain_problem_data_context}. 
            
            I wanted you to formalize the information to more explicitly and write the logic regarding driving behaviors contained in this information. 
            Regarding this information, I first generated action descriptions here: 
            {generated_actions}. 
            
            From all of this information, I generated the following PDDL (Planning Domain Definition Language) Domain model here: 
            {domain}.

            In addition to everything else above, I have some more pertinent information here regarding the PDDL problem corresponding to the PDDL domain above: 
            {scenario_domain_problem_data_problem_data}
            
            First, please repeat the types, states (predicates) and actions in this file in your mind. 
            Then think step by step about a PDDL problem for this PDDL domain. Please think about whether this problem does indeed have a solution. In other words, whether a plan exists for this problem.
            Now, please generate a PDDL (Planning Domain Definition Language) problem file. I only want the contents that would be in
            such a file, no other information in your writing. Keep in mind that this content will be entered into a file with a .pddl extension and saved so no extra information should be contained.
            
            No pddl, lisp or any other tags to be used. Just the pddl lines in the output. No tags. No tags. No tags.
            """
        return problem_initial_prompt
    
    def generate_final_problem_prompt(self, 
                                      scenario_domain_problem_data_context,
                                      generated_actions,
                                      domain,
                                      scenario_domain_problem_data_problem_data,
                                      initial_problem):
        final_problem_prompt = f"""
            I have an autonomous vehicle test scenario described here: {scenario_domain_problem_data_context}. 
            
            I want you to formalize the information to more explicitly write the logic regarding the various driving behaviors in this information. Regarding this information, I first generated 
            action descriptions here: {generated_actions}. Then I generated the following PDDL (Planning Domain Definition Language) Domain model here: {domain}.

            In addition to everything else above, regarding the PDDL problem file, I have some pertinent information here: {scenario_domain_problem_data_problem_data}
            
            From all of this information, I generated the PDDL problem file. Carefully read this PDDL problem file:
            {initial_problem}.

            Please consider all the information above and generate a refined, correct and better quality PDDL problem file. Thank you! 
            
            Again, I only want the contents that would be in such a file, no other information in your writing. Keep in mind that this content will be entered into a file with a .pddl extension and saved so no extra information should be contained.
            
            No pddl, lisp or any other tags to be used. Just the pddl lines in the output. No tags. No tags. No tags.
            """
        return final_problem_prompt
    
    def generate_pddl_problem(self, scenario_domain_problem_data_context, generated_actions, domain, scenario_problem_data, scenario_id, interaction_id):
        initial_problem_prompt = self.generate_initial_problem_prompt(scenario_domain_problem_data_context=scenario_domain_problem_data_context, 
                                                                      generated_actions=generated_actions, 
                                                                      scenario_domain_problem_data_problem_data=scenario_problem_data, 
                                                                      domain=domain)
        initial_pddl_problem = self.llm_call(initial_problem_prompt)
        final_problem_prompt = self.generate_final_problem_prompt(scenario_domain_problem_data_context,
                                                                  generated_actions,
                                                                  domain,
                                                                  scenario_domain_problem_data_problem_data=scenario_problem_data,
                                                                  initial_problem=initial_pddl_problem)
        self.pddl_problem = self.llm_call(final_problem_prompt)
        print("PDDL problem is {}".format(self.pddl_problem))
        self.write_pddls(write_problem=True, scenario_id=scenario_id, problem_info=self.pddl_problem, interaction_id=interaction_id)
        return self.pddl_problem
        
    
    def generate_llm_pddl_judge_prompt(self, 
                                       scenario_domain_problem_data_context,
                                       domain,
                                       scenario_domain_problem_data_problem_data,
                                       problem_final):
        llm_pddl_judge_prompt = f"""
            First, read the context information for the given scenario:
            {scenario_domain_problem_data_context}
            
            Now, carefully read the generated domain file:
            {domain}

            Now, carefully review the problem data in the scenario:
            {scenario_domain_problem_data_problem_data}
            
            Carefully read this PDDL problem file:
            {problem_final}.

            Now score the generated domain and problem PDDL files according to the given rubric:

            1. Consistency: Are the facts in the context information above consistently and correctly presented in the domain and problem files? Rate this output on a scale of 1 to 10. Explain your rating. 
            2. Domain coverage: Does the generated domain PDDL domain file adequately cover the information in the context above? Rate this output on a scale of 1 to 10. Explain your rating.
            3. Problem coverage: Does the generated problem PDDL file adequately cover the given problem data as presented above? The problem data asks specific questions with respect to the context. 
            Therefore, you must rate the coverage with respect to this specific question only. Rate this output on a scale of 1 to 10. Explain your rating.

            Format your output exactly in the following manner:
            <open curly bracket>
            "Consistency":
                <open curly bracket>
                "Score explanation": "<Detailed explanation here.>", 
                "Grade": "<Only a score here between 1 and 10.>" 
                <close curly bracket>,
            "Domain coverage":
                <open curly bracket>
                "Score explanation": "<Detailed explanation here.>", 
                "Grade": "<Only a score here between 1 and 10.>" 
                <close curly bracket>,
            "Problem coverage":
                <open curly bracket>
                "Problem data provided": "<Problem data given exactly as it is above.>"
                "Score explanation": "<Detailed explanation here.>", 
                "Grade": "<Only a score here between 1 and 10.>" 
                <close curly bracket>
            <close curly bracket>

            No tags. Just the dictionary in the output. Nothing else, nothing else.
            """
        return llm_pddl_judge_prompt
    
    def generate_llm_eval(self, scenario_domain_problem_data_context="",
                                domain="",
                                scenario_domain_problem_data_problem_data="",
                                problem_final="",
                                scenario_id="",
                                interaction_id=""):
        
        llm_pddl_judge_prompt = self.generate_llm_pddl_judge_prompt(scenario_domain_problem_data_context=scenario_domain_problem_data_context,
                                domain=self.pddl_domain,
                                scenario_domain_problem_data_problem_data=scenario_domain_problem_data_problem_data,
                                problem_final=self.pddl_problem)
        llm_eval = self.llm_call(llm_pddl_judge_prompt, dictionary_mode=True)
        # Each sentence in the scenario context pertains to a fact.
        # We can split the context by sentence and count the word count per sentence to get a sense of how difficult the facts are.
        # Longer individual sentences would mean more complex facts.
        context_sentence_list = scenario_domain_problem_data_context.split(". ")
        total_word_count_sentence = 0
        for sentence_index in range(len(context_sentence_list)): 
                total_word_count_sentence += len(context_sentence_list[sentence_index].split())

        average_word_count_sentence = int(total_word_count_sentence / len(context_sentence_list))
        llm_eval.setdefault("average_context_sentence_word_count", average_word_count_sentence)
        llm_eval.setdefault("total_word_count", total_word_count_sentence)

        domain_problem_files = pddlpy.DomainProblem("apla-planner/generated_pddls_deepseek/dataset/domains/"+scenario_id+"/domain_deepseek_chat_"+scenario_id+".pddl", 
                                                    "apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_id+"/problem_deepseek_chat_"+interaction_id+".pddl")
        llm_eval.setdefault("domain_action_count", len(list(domain_problem_files.operators()))) # List of actions written in the domain.
        llm_eval.setdefault("initial_state_size", len(domain_problem_files.initialstate())) # Initial state in the problem file.    

        with open("apla-planner/generated_pddls_deepseek/dataset/problems/"+scenario_id+"/LLM_eval_"+interaction_id+".json", "w", encoding='utf-8') as file_eval:
                    json.dump(llm_eval, file_eval, indent=4) # We want to read the article as a single string, so that we can feed it to gpt.
                    file_eval.close()
        return llm_eval
        
