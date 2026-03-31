from openai import OpenAI
import time
from .base import LLM
import os

class Qwen30BInstruct(LLM):
    def setup_model(self):
        self.model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
        self.timestamp = 0
    
    def get_response(self, messages, retry=True, **kwargs):
        client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", None),
            base_url=f"http://{os.environ.get("VLLM_URL","localhost")}:{os.environ.get('VLLM_PORT',7773)}/v1"
        )
        # while self.timestamp > (time.time() - 30):
        #     time.sleep(10)
        self.timestamp = time.time()
        try:
            result = client.chat.completions.create(
                stop=["<PAUSE>"],
                messages=messages,
                model=self.model,
                **kwargs
            ).choices[0].message.content
            return result
        except Exception as e:
            if retry:
                time.sleep(0.5)
                return self.get_response(messages, retry=False)
            else:
                raise e

class Qwen30BThinking(Qwen30BInstruct):
    def setup_model(self):
        self.model = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
        self.timestamp = 0
            
class KimiK2Thinking(LLM):
    def setup_model(self):
        self.model = "nvidia/Kimi-K2-Thinking-NVFP4"
        self.timestamp = 0
    
    def get_response(self, messages, retry=True, **kwargs):
        client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", None),
            base_url=f"http://{os.environ.get("VLLM_URL","localhost")}:{os.environ.get('VLLM_PORT',7773)}/v1"
        )
        # while self.timestamp > (time.time() - 30):
        #     time.sleep(10)
        self.timestamp = time.time()
        try:
            result = client.chat.completions.create(
                stop=["<PAUSE>"],
                messages=messages,
                model=self.model,
                **kwargs
            ).choices[0].message.content
            return result
        except Exception as e:
            if retry:
                time.sleep(0.5)
                return self.get_response(messages, retry=False)
            else:
                raise e

class MistralNemo12B(LLM):
    def setup_model(self):
        self.model = "mistralai/Mistral-Nemo-Instruct-2407"
        self.timestamp = 0
    
    def get_response(self, messages, retry=True, **kwargs):
        client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", None),
            base_url=f"http://{os.environ.get("VLLM_URL","localhost")}:{os.environ.get('VLLM_PORT',7773)}/v1"
        )
        # while self.timestamp > (time.time() - 30):
        #     time.sleep(10)
        self.timestamp = time.time()
        try:
            result = client.chat.completions.create(
                stop=["<PAUSE>"],
                messages=messages,
                model=self.model,
                **kwargs
            ).choices[0].message.content
            return result
        except Exception as e:
            if retry:
                time.sleep(0.5)
                return self.get_response(messages, retry=False)
            else:
                raise e