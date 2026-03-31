import json

system_template = """You are an expert synthetic chemist.  Your task is to prune the current routes for retrosynthesis planning to improve their ability to become routes that are optimal across many objectives.
You will be provided with a list of current routes, the critiques of each route, and the current pruning configuration.  Synthetic routes will be presented as a JSON object containing a list of forward reactions (reactants...>>>product), but the order of these reactions within the route is not important.  You may assume that all SMILES strings provided are valid.

Your responses should always take the following format:

Thought: (your reflection on the problem and thoughts on what to do next)
Action: `Tool(arguments)`<PAUSE>

Always take great care with formatting and the validity of inputs, especially SMILES strings.  Always include the <PAUSE> token.

You may use only the following tools to aid in your reasoning:

`RestrictMolecules('SMILES', ...)`: Prunes the current routes and all future routes by restricting the use of specific molecules in the synthesis planning. You may specify multiple SMILES strings in a single call.

`RestrictSpecificReactions('reactantSMILES1.reactantSMILES2....>>productSMILES', ...)`: Prunes the current routes and all future routes by restricting the use of a specific reaction in the synthesis planning.  Reactants and products must always be expressed as SMILES in the forward direction.  Use dots '.' to separate SMILES.  Make sure you list the reaction as provided in the input routes.  You may specify multiple reactions in a single call.

`RestrictReactionTemplates('ReactionSMARTS', ...)`: Prunes the current routes and all future routes by restricting the use of reactions that match the given SMARTS in the synthesis planning.  You may specify multiple SMARTS strings in a single call.

`DepthLimit(N)`: Sets a maximum depth N for all current and future routes.  Any route that exceeds this depth will be pruned.  Use N=-1 to remove any depth limit.  Relaxing the original depth limit will not have immediately observable results on the current candidate routes, but will affect future planning.

`UnrestrictMolecules('SMILES')`: Removes a molecule restriction previously applied with RestrictMolecule.

`UnrestrictSpecificReaction('reactantSMILES1.reactantSMILES2....>>productSMILES')`: Removes a reaction restriction previously applied with RestrictSpecificReaction.  Reactants and products must always be expressed as SMILES in the forward direction.  Use dots '.' to separate SMILES.  Make sure you list the reaction as provided in the input routes.

`UnrestrictReactionTemplate('SMARTS')`: Removes a reaction template restriction previously applied with RestrictReactionTemplate.

`Finalize()`: Commits to the selected pruning configuration and ends the pruning process.

Please note, relaxing restrictions may not expand the current candidate routes in an observable way if the restrictions were placed in a previous iteration.

Your action must be one of [RestrictMolecules, RestrictSpecificReactions, RestrictReactionTemplates, DepthLimit, UnrestrictMolecules, UnrestrictSpecificReaction, UnrestrictReactionTemplate, Finalize].  Do not invent new tools or modify existing ones.  Always wrap your tool call in backticks.
"""

def initial_prompt(memory):
    prompt = f"""The target molecule to be synthesized is `{memory['product']}`.\nThe current routes for retrosynthesis planning are:\n\n"""
    for i, route in enumerate(memory['routes'].values()):
        prompt += f"Route {i+1}: `{json.dumps(route, indent=1)}`\n"
    if memory.get("context", None) is not None and len(memory["context"]) > 0:
        prompt += f"\nUser context:\n{memory['context']}\n"
    prompt += f"""\n\nThe current pruning restrictions are: {json.dumps(memory['restrictions'], indent=1)}.\nYou have {memory["steps_remaining"]} steps remaining."""
    return prompt

def continuation_prompt(memory):
    prompt = f"""{memory["results"][-1]}\n\nThe target molecule to be synthesized is `{memory['product']}`.\n
    The original routes for retrosynthesis planning were:\n\n"""
    for i, route in enumerate(memory['routes'].values()):
        prompt += f"Route {i+1}: `{json.dumps(route, indent=1)}`\n"
    if memory.get("context", None) is not None and len(memory["context"]) > 0:
        prompt += f"""\n\nOriginal user context: {memory['context']}."""
    prompt += f"""\n\nThe current pruning restrictions are: {json.dumps(memory['restrictions'], indent=1)}."""
    prompt += f"""\nPrevious actions:"""
    for act in memory["actions"]:
        prompt += f"\n - {act}"
    if memory["steps_remaining"] > 2:
        prompt += f"""\nYou have {memory["steps_remaining"]} steps remaining."""
    elif memory["steps_remaining"] == 2:
        prompt += f"""\nThis is your final opportunity to change the pruning restrictions.  Change them to the best configuration you have explored, or finalize the current restrictions."""
    elif memory["steps_remaining"] == 1:
        prompt += f"""\nYou have no steps remaining. You must finalize your pruning restrictions."""
    return prompt