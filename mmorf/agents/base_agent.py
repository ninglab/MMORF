from ..utils.logger import Logger
import json
import os
import re
import time

class BaseAgent(object):
    include_history = True
    def __init__(self, llm):
        self.llm = llm
        self.llm.setup_model()
        self.memory = {}
        self.message_history = []
    
    def toolset(self):
        raise NotImplementedError("Subclasses must implement toolset method.")
    
    def initial_prompt(self, memory):
        raise NotImplementedError("Subclasses must implement initial_prompt method.")

    def continuation_prompt(self, memory):
        raise NotImplementedError("Subclasses must implement continuation_prompt method.")
    
    def system_prompt(self):
        raise NotImplementedError("Subclasses must implement system_prompt method.")

    def setup_memory(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement setup_memory method.")

    def execute_response(self, response):
        print("Executing: ", response)
        self.memory["thoughts"].append(response.split("Action:")[0])
        self.log({"role": "response", "content": response})
        action = response.split("Action: ")[-1] if "Action: " in response else response
        action = re.findall(r'`([^`]+)`', action+"`")
        if str(action).count(')') < action.count('('):
            action += ")" # bypass errors due to missing closing parenthesis
        if not action:
            self.log({"role": "error", "content": "No action found in response."})
            print(response)
            self.memory["error"] = True
        else:
            print("Action found:", action)
            action = action[-1].strip()
            try:
                output = {"result": None, "error": None}
                self.memory["actions"].append(action)
                self.log({"role": "info", "content": f"Executing action: {action}"})
                action_all = self.toolset() + "\n\noutput['result'], output['error'] = " + action + "\n"
                self.log({"role": "action", "content": "```python\n"+action_all+"\n```"})
                start = time.time()
                try:
                    exec(action_all, {"memory": self.memory, "output": output})
                except Exception as e:
                    if 'never closed' in str(e):
                        self.memory["actions"][-1] += ")"
                        exec(action_all[:-1]+")\n", {"memory": self.memory, "output": output})
                    else:
                        raise e
                end = time.time()
                minutes = int((end - start) // 60)
                if output["error"]:
                    self.memory["actions"][-1] += f"(Latency: {minutes} min, Result: ⚠️ Error)"
                else:
                    self.memory["actions"][-1] += f"(Latency: {minutes} min, Result: ✅ Success)"
                self.memory["results"].append(output["result"])
                self.memory["error"] = output["error"]
            except Exception as e:
                print("Error executing action:", str(e))
                self.memory["results"].append(str(e))
                self.log({"role": "error", "content": f"Error executing action: {str(e)}"})
                if os.environ.get("REPLANNER_DEBUG", "0") == "1":
                    raise e
                self.memory["error"] = True
        return self.memory["error"]
    
    def log(self, message):
        if isinstance(message, dict):
            message = message.copy()
            # Add source information to the message (name of actual agent subclass at runtime)
            message["source"] = self.__class__.__name__
        Logger.log(json.dumps(message, indent=2))

    def loop(self, product, routes, context, step_limit=1, reset=True, stop_on_error=False, count_errors=False):
        if reset or self.memory is None:
            self.message_history = []
            self.setup_memory(product, routes, step_limit, context, True)
            self.message_history = [{"role": "system", "content": self.system_prompt()}]
            self.messages = [{"role": "system", "content": self.system_prompt()},
                             {"role": "user", "content": self.initial_prompt(self.memory)}]
        while not (self.memory["error"] and stop_on_error) and not self.memory["finished"] and self.memory["steps_remaining"] != 0:
            self.memory["error"] = self()
            self.log({"role": "info", "content": f"Finished? {self.memory['finished']}"})
            self.memory["steps_remaining"] -= 1
            if self.memory["error"]:
                if len(self.memory["results"]) > 0:
                    self.log({"role": "error", "content": f"An error occurred during execution."+self.memory["results"][-1]})
                else:
                    self.log({"role": "error", "content": f"An error occurred during execution."})
                if stop_on_error:
                    return False
                else:
                    import sys
                    sys.stderr.write(json.dumps(self.messages, indent=1)+"\n")
                    self.messages.pop(-1) # undo last action
                    self.messages.pop(-1) # undo last user message
                    sys.stderr.write(json.dumps(self.messages, indent=1)+"\n")
                    if not count_errors:
                        self.memory["steps_remaining"] += 1
                    self.memory["error"] = False
            if not self.include_history:
                self.message_history += self.messages[1:]
                self.messages = [{"role": "system", "content": self.system_prompt()},
                                {"role": "user", "content": self.continuation_prompt(self.memory)}]
            if self.memory["finished"]:
                return True
            elif self.include_history:
                self.messages.append({"role": "user", "content": self.continuation_prompt(self.memory)})
        return True
        
    def __call__(self):
        response = None
        print(self.messages)
        response = self.llm.get_response(
            messages=self.messages
        )
        self.log({"role": "system", "content": self.messages[0]["content"]})
        self.log({"role": "user", "content": self.messages[-1]["content"]})
        if response is None or response.strip() == "":
            self.log({"role": "error", "content": "Received empty response from LLM."})
            return True
        self.messages.append({"role": "assistant", "content": response})
        return self.execute_response(response)