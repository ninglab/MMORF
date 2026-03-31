from multiprocessing import Lock
import os, json
import time

class Logger(object):
    """Logger class to handle logging messages to a file."""
    @classmethod
    def log(cls, message, filename=os.environ.get("REPLANNER_LOG_FILE", "replanner.log")):
        """Log a message to the specified log file."""
        if isinstance(message, dict):
            message["timestamp"] = message.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        elif isinstance(message, str):
            try:
                message = json.loads(message)
                message["timestamp"] = message.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            except json.JSONDecodeError:
                message = {"role": "info", "content": message, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        else:
            message = {"role": "info", "content": str(message), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        message = json.dumps(message)
        with cls.lock:
            with open(filename, "a") as log_file:
                log_file.write(f"{message}\n")
        if os.environ.get("REPLANNER_DEBUG", "0") == "1":
            print(f"LOG: {json.dumps(message, indent=2)}")
Logger.lock = Lock()