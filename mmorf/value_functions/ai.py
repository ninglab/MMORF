from .base import ValueFn, ReactionLevelWrapper
from ..llms.claude4 import Claude
import hashlib
import os, json, re
profile = lambda x: x  # Default to no profiling if not enabled
if os.environ.get("ENABLE_PROFILING", "0") == "1":
    from line_profiler import profile as p
    profile = p

class AI(ReactionLevelWrapper):
    def child_type(self):
        return _AI

class _AI(ValueFn):
    """
    A value function for synthesis planning that always returns 1.
    This is a placeholder and should be replaced with a more meaningful implementation.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.question = args[0] if len(args) else ""
        self.coeff = 1.0
        self.intercept = 0.0
        self.llm = Claude()
        self.llm.setup_model()
        self.children = []

    @profile
    def __call__(self, reaction=None, v_orig=None, output=None, node=None, force_compute=False, **kwargs):
        if output is None:
            output = "./"
        if reaction is None:
            return self.intercept + self.coeff * 1.0 # default 1.0 good score
        unique_string = self.question + reaction
        file = output + "_AI_"+hashlib.sha256(unique_string.encode()).hexdigest()+".json"
        if os.path.exists(file):
            with open(file, 'r') as f:
                data = json.load(f)
                score = data.get("score", 1.0)
        else:
            prompt = f"Evaluate the reaction: {reaction} based on the question: {self.question} where 1.00 is a good score and 0.00 is a bad score.  Use the format:\nThought: (your reflection)\nScore: `X.XX` (make sure to surround it in backticks)\n<PAUSE>\n"
            response = self.llm.get_response(messages=[{"role": "user", "content": prompt}])
            score = re.findall(r'Score:\s*`([0-9.]+)`', response)
            if not score:
                print(f"Warning: No score found in response for {self.question} and reaction {reaction}. Using default score of 1.0.")
                score = 1.0
            score = float(score[-1])
            with open(file, 'w') as f:
                json.dump({"score": score, "response": response}, f)
        return self.coeff * score + self.intercept
    
    def __repr__(self):
        return f"{self.coeff} * AI() + {self.intercept}"