import json

system_template = """You are an expert synthetic chemist.  Your task is to modify the planning value function for retrosynthesis planning to improve its ability to find routes that are optimal across many objectives.
You will be provided with the previous (existing) value function, the current routes sampled using MCTS with the previous (existing) value function, and the critiques of each route.
Synthetic routes will be presented as a JSON object containing a list of forward reactions (reactants...>>>product), but the order of these reactions within the route is not important.  You may assume that all SMILES strings provided are valid.

Your responses should always take the following format:

Thought: (your reflection on the problem and thoughts on what to do next)
Action: `Tool(arguments)`<PAUSE>

Always take great care with formatting and the validity of inputs, especially SMILES strings.  Always include the <PAUSE> token.

You may use the following tools to aid in your reasoning:

SetValueFunction("Weight1*Component1('arg1', 'arg2', ...) + Weight2*Component2(...) - Weight3*Component3(...)") : Sets the value function using the following components:

Route-level components (added once for the entire route):
- Synth(): Returns a pretrained synthesis planning value for the route.
- Depth(): Returns the depth of the route up to the current reaction step.  During planning, the depth is the number of reaction steps taken so far.

Reaction-level components (value is the sum over all reactions in the route, consider normalizing by depth depending on your needs):
- BBPrice(): Returns the monetary cost of the reaction based on its purchasable building block reactants.
- GHS('HXXX', ...): Returns 1 if the reaction involves a molecule with the specified GHS hazard code, otherwise returns 0.  You may use multiple GHS components to capture multiple hazard codes.  Can be slow.
- FastCarc(): Returns 1 if the reaction involves a molecule flagged as a potential carcinogen, otherwise returns 0.  Uses a fast, lightweight set of structural alerts.
- MaxSim('SMILES', ...): Returns a score from 0 to 1 indicating the maximum Tanimoto similarity of any molecule in the reaction to the input SMILES, using Morgan fingerprints.  This can be used to bias the planner towards or away from certain chemotypes.
- MinSim('SMILES', ...): Returns a score from 0 to 1 indicating the minimum Tanimoto similarity of any molecule in the reaction to the input SMILES, using Morgan fingerprints.
- Pyro(): Returns 1 if the reaction involves a molecule flagged as potentially pyrophoric, otherwise returns 0.

Use single quotes within the arguments of components, and double quotes to enclose the entire value function string.  Weights may be positive or negative.  You may combine multiple components using addition, subtraction, multiplication, and division.  You may NOT use any other operations (e.g., exponentiation (^ or **), thresholding (><), etc.).  You may not invent new components.  You must use only the components provided above.
After each SetValueFunction call, the value function will be updated to the new value function you provided, and you will be shown how the current routes would be reranked using the new value function.  Keep in mind that complex value functions may incur high latency during future planning.
Try to minimize latency while still achieving your goals.

Finalize(): Completes the value function modification task and returns the final value function.

Your action must be one of [SetValueFunction, Finalize].  Do not invent new actions or modify existing ones.  Do not invent new tools or modify existing ones.  Always wrap your tool call in backticks.  Also, do not invent new components or operations for the Value Function.
"""

rejected_tools = """
- SlowCarc(): Returns a probabilistic score from 0 to 1 indicating the likelihood that the reaction involves a molecule that is a carcinogen.  Uses a computationally intensive ADMET prediction model. (Caution: Very slow, consider FastCarc instead.)
"""

def initial_prompt(memory):
    prompt = f"""The target molecule to be synthesized is `{memory['product']}`.\nThe current routes for retrosynthesis planning are, from highest to lowest value:\n\n"""
    for i, route in enumerate(sorted(memory['routes'].values(), key=lambda r: r['value'], reverse=True)):
        prompt += f"{i+1}. `{json.dumps(route, indent=1)}`\n"
    if memory.get("context", None) is not None and len(memory["context"]) > 0:
        prompt += f"""\n\nUser context: {memory['context']}."""
    prompt += f"""\n\nThe current value function is: `{memory['value_function']}`.\nYou have {memory["steps_remaining"]} steps remaining."""
    return prompt

def continuation_prompt(memory):
    prompt = f"""{memory["results"][-1]}\n\nThe target molecule to be synthesized is `{memory['product']}`.\n
    The current routes for retrosynthesis planning are, from highest to lowest value:\n\n"""
    for i, route in enumerate(sorted(memory['routes'].values(), key=lambda r: r['value'], reverse=True)):
        prompt += f"{i+1}. `{json.dumps(route, indent=1)}`\n"
    if memory.get("context", None) is not None and len(memory["context"]) > 0:
        prompt += f"""\n\nUser context: {memory['context']}."""
    prompt += f"""\n\nThe current value function is: `{memory['value_function']}`. User context: {memory['context']}."""
    prompt += f"""\nPrevious actions:"""
    for act in memory["actions"]:
        prompt += f"\n - {act}"
    if memory["steps_remaining"] > 2:
        prompt += f"""\nYou have {memory["steps_remaining"]} steps remaining.  Examine the updated route values to decide if you are satisfied with the current value function or if you want to modify it further."""
    elif memory["steps_remaining"] == 2:
        prompt += f"""\nThis is your final opportunity to change the value function.  Change it to the best value function you have explored, or finalize the current value function."""
    elif memory["steps_remaining"] == 1:
        prompt += f"""\nYou have no steps remaining. You must finalize your value function."""
    return prompt