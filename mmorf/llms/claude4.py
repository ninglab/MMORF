from anthropic import Anthropic
import time
from .base import LLM

class Claude(LLM):
    def setup_model(self):
        self.model = "claude-sonnet-4-5-20250929"
        self.timestamp = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.client = Anthropic()
    
    def get_response(self, messages, retry=True):
        client = self.client
        while self.timestamp > (time.time() - 30):
            time.sleep(10)
        self.timestamp = time.time()
        if messages[0]["role"] == "system":
            return "\n".join([r.text for r in client.messages.create(
                max_tokens=8192,
                stop_sequences=["<PAUSE>"],
                system=messages[0]["content"],
                messages=messages[1:],
                model=self.model
            ).content])
        try:
            result = "\n".join([r.text for r in client.messages.create(
                max_tokens=8192,
                stop_sequences=["<PAUSE>"],
                messages=messages,
                model=self.model
            ).content])
            return result
        except Exception as e:
            if retry:
                if "529" in str(e) or "overloaded" in str(e):
                    time.sleep(0.5)
                    return self.get_response(messages, retry=False)
            else:
                raise e
                