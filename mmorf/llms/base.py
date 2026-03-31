from typing import List, Dict

class LLM(object):
    def setup_model(self, *args):
        """
        This function should set up the model for the agent.
        """
        return NotImplemented
    
    def get_response(self, messages : List[Dict[str, str]], retry=True, **kwargs) -> str:
        """
        This function should return the response from the model.
        """
        return NotImplemented
    
    def sample_responses(self, messages, num_samples, temperature=0.7):
        for _ in range(num_samples):
            yield self.get_response(messages, seed=_, temperature=temperature)