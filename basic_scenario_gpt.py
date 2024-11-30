from guidance import models, gen, user, assistant, system
import parse_scenario_womd 


def generate_scenario_concepts(number_of_concepts, granularity, scenario_data):
    gpt_scenario = models.OpenAI(model="gpt-4o", echo=False)

    with system():
        lm_scenario = gpt_scenario

    with user():
        lm_scenario += f"""
        Think deeply about scenarios for testing autonomous vehicles. 

        I need some states of the world that would be relevant for logically describing this traffic scenario:
        {scenario_data}

        A state is just an assertion with a true or false value that's representing the world in that particular moment. 
        This is similar to the concept of a turn in a turn based game.

        There must be states regarding the following concepts:
        * Relative distance: Distance from landmarks in the scene such as relative to the sidewalk, traffic light, lane markings, intersection area, crosswalks, other vehicles and pedestrians and more.
        * Relative velocity: Velocity is a vector representing the rate of change of position and the direction of this change. The direction aspect of the velocity vector must also be described relative to landmarks mentioned above.  
        * Relative acceleration: This is a vector representing the rate of change of velocity and the direction of this change.  The direction aspect of the acceleration vector must also be described relative to landmarks mentioned above.
        * Mental state of the driver: The driver can be fatigued, drunk, distracted or just plain reckless. In addition, the driver can have other attributes. These can affect the movement of the vehicle.
        
        Can you think of {str(number_of_concepts)} more concepts that can be represented as states? 
        Rewrite the important and unique information for this given scenario. 
        These concepts that you need to generate must be relevant to the given scenario. 
        At the same time, they must not refer to very specific scenario information and must be general enough to be used in other scenarios. 
        Increase the granularity of the concepts in proportion to the granularity level.
        The granularity level is {str(granularity)} on a scale of 1 to 10 with 1 being the least and 10 being the most granular
        Granularity pertains to how specific the information is.

        Make sure to rewrite the concepts given in the generated list of concepts in addition to your concepts. 
        """

    with assistant():
        lm_scenario += gen("concepts", temperature=0.8)
    
    print("The scenario concepts are {}".format(lm_scenario["concepts"]))
    return lm_scenario["concepts"]

def generate_scenario_states(concepts):
    gpt_scenario = models.OpenAI(model="gpt-4o", echo=False)

    with system():
        lm_scenario = gpt_scenario

    with user():
        lm_scenario += f"""
        Based on the concepts detailed in {concepts}, 
        Write down a list of states pertaining to these concepts in natural language. Write them in the following format: 
        ```json 
        <curly bracket> 
            "<state name>": <curly bracket> 
                "statement": "<the assertion in natural language. Use the fewest words possible for maximum clarity>
            <close curly bracket>, 
            "<state name>": <open curly bracket> 
                "statement": "<the assertion in natural language>,
            <close curly bracket>, 
            ... 
        <close curly bracket>
        json```

        Be very very very specific and granular. Very granual,  fine details and specific.    
        """

    with assistant():
        lm_scenario += gen("state_dictionary", temperature=0.8)
    
    return lm_scenario["state_dictionary"]

def generate_scenario_actions(concepts, granularity=2):
    gpt_scenario = models.OpenAI(model="gpt-4o", echo=False)

    with system():
        lm_scenario = gpt_scenario

    with user():
        lm_scenario += f"""
        Based on the concepts detailed in {concepts}, 
        * Write down a list of actions that map between these states in natural language. 
        * Each action has some causal states (predicates) and some effect states that will be true or false.
        * Each action is a cause and effect mapping between any number of causal states and any number of effect states.
        * Actions and states must not contradict each other.
        * The action itself will only become true when the causal states and the effect states are in the specific states that this description details.
        * Write them in the following format: 
        ```json 
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
        json```

        Increase the granularity of these actions in proportion to the granularity level.
        Granularity pertains to how specific the information is. 
        While the actions must be relevant to the given scenario, they must be general enough to be used for other scenarios as well.
        The granularity level is {str(granularity)} on a scale of 1 to 10 with 1 being the least and 10 being the most granular

        """

    with assistant():
        lm_scenario += gen("action_dictionary", temperature=0.8)
    
    print("The scenario actions are {}".format(lm_scenario["action_dictionary"]))
    return lm_scenario["action_dictionary"]

# Removed from this project after consideration
def generate_scenario_states(concepts):
    gpt_scenario = models.OpenAI(model="gpt-4o", echo=False)

    with system():
        lm_scenario = gpt_scenario

    with user():
        lm_scenario += f"""
        Based on the concepts detailed in {concepts}, 
        Write down a list of states pertaining to these concepts in natural language. Write them in the following format: 
        ```json 
        <curly bracket> 
            "<state name>": <curly bracket> 
                "statement": "<the assertion in natural language. Use the fewest words possible for maximum clarity>
            <close curly bracket>, 
            "<state name>": <open curly bracket> 
                "statement": "<the assertion in natural language>,
            <close curly bracket>, 
            ... 
        <close curly bracket>
        json```

        Be very very very specific and granular. Very granual,  fine details and specific.    
        """

    with assistant():
        lm_scenario += gen("state_dictionary", temperature=0.8)
    
    return lm_scenario["state_dictionary"]

def respond_scenario_query(concepts, actions, questions):
    gpt_scenario = models.OpenAI(model="gpt-4o", echo=False)

    with system():
        lm_scenario = gpt_scenario

    with user():
        lm_scenario += f"""
        Based on the concepts detailed in {concepts} and actions detailed in {actions}, respond to the following questions:
        {questions}
        Be very pecific and very granular. Very granual,  fine details and specific.    
        """

    with assistant():
        lm_scenario += gen("scenario_response", temperature=0.8)
    
    #print("The scenario responses are {}".format(lm_scenario["scenario_response"]))
    return lm_scenario["scenario_response"]