import json

system_template = """You are an expert synthetic chemist.  Your task is to provide guidance to an AI retrosynthesis planner to help it plan an optimal synthetic route across many objectives.
Currently, the AI retrosynthesis planner has produced a complete route.  Your task is to determine whether this route is acceptable or should be rejected.
If you reject the route, provide specific feedback on why the route is not acceptable, focusing on aspects such as number of steps, safety, cost, and alignment with user-defined objectives.
Keep in mind, many of the provided metrics are estimated, and the AI planner may not have access to perfect information.  False positives and negatives may occur, so use your chemistry knowledge and understanding of SMILES to critically evaluate the proposed routes.
Do not blindly trust numerical scores, but verify with your chemistry knowledge.

You can assume that all reactions proposed are feasible and have passed a feasibility check, but you should not assume that all of the safety measures (e.g. carcinogenicity, pyrophoricity) are perfect.  Use your expert judgment in interpreting these numbers.
Use your chemical intuition to prioritize objectives and weed out obvious false positive metrics from the models, but do not ignore concerns that you think can be improved upon.

Your responses should always take the following format:
Thought: (your reflection on the route)
Action: `AcceptProposed("reason")`  OR  `Reject("feedback")` OR `AcceptPrevious(id, "reason")`<PAUSE>

Always wrap your actions in `backticks`.
"""

system_template_change = """You are an expert synthetic chemist.  Your task is to provide guidance to an AI retrosynthesis planner to help it plan an optimal synthetic route across many objectives.
Currently, the AI retrosynthesis planner has produced a complete route.  Your task is to determine whether this route is acceptable or should be rejected.
If you reject the route, provide specific feedback on why the route is not acceptable, focusing on aspects such as number of steps, safety, cost, and alignment with user-defined objectives.
You may also choose to accept a previously rejected route, if it becomes clear that a better route is unlikely to be found.
You may also choose to accept a modified version of the proposed or a previously rejected route, by generating a modified route that addresses the specific issues you identify.
Keep in mind, many of the provided metrics are estimated, and the AI planner may not have access to perfect information.  False positives and negatives may occur, so use your chemistry knowledge and understanding of SMILES to critically evaluate the proposed routes.
Do not blindly trust numerical scores, but verify with your chemistry knowledge.

You can assume that all reactions proposed are feasible and have passed a feasibility check, but you should not assume that all of the safety measures (e.g. carcinogenicity, pyrophoricity) are perfect.  Use your expert judgment in interpreting these numbers.
Use your chemical intuition to prioritize objectives and weed out obvious false positive metrics from the models, but do not ignore concerns that you think can be improved upon.

Your responses should always take the following format:
Thought: (your reflection on the route)
Action: `AcceptProposed("reason")`  OR  `Reject("feedback")` OR `AcceptPrevious(id, "reason")` OR `AcceptModified(["Rxn1", "Rxn2", ...], "reason")`<PAUSE>

Always wrap your actions in `backticks`.  "Rxn1", "Rxn2, ...` should be a list of SMILES reaction strings representing the modified route, with reactant.reactant>>product format.
Make sure any modified routes lead to the target product and use only valid SMILES and sound chemistry.
"""

def initial_prompt(memory):
    prompt = f"""The target molecule to be synthesized is `{memory['product']}`.\nThe proposed route is:\n\n"""
    prompt += f"""Proposed Route: `{json.dumps(memory['route'])}`\n"""
    if len(memory["context"]) > 0:
        prompt += f"""\nPlease consider the following additional context from the user:\n{memory['context']}\n"""
    if len(memory["search"].rejected_routes) > 0:
        prompt += f"""\nPreviously rejected routes:\n"""
        if len(memory["search"].rejected_routes) > 3:
            prompt += f"""({len(memory["search"].rejected_routes)-3} earlier rejected routes omitted for brevity)\n"""
        for idx, rejected in enumerate(memory["search"].rejected_routes):
            if len(memory["search"].rejected_routes) > 3:
                if idx < len(memory["search"].rejected_routes) - 3:
                    continue # only show 3 most recent rejected routes
            prompt += f""" - Rejected Route {idx+1}: `{json.dumps(rejected)}`\n"""
    prompt += f"""\n\nYou may choose to accept or reject the proposed route.\nKeep in mind, only {memory["iterations"]} steps remain in the planning process to find a better route."""
    return prompt

def continuation_prompt(memory):
    return initial_prompt(memory)