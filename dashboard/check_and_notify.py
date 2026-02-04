import json
import os
from datetime import datetime

CONV_PATH = "/home/the_host/clawd/dashboard/conversation.jsonl"
STATE_PATH = "/home/the_host/clawd/dashboard/last_seen_msg.json"

def get_last_seen():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f).get("timestamp", "")
        except:
            return ""
    return ""

def save_last_seen(ts):
    with open(STATE_PATH, "w") as f:
        json.dump({"timestamp": ts}, f)

def main():
    last_ts = get_last_seen()
    new_messages = []
    
    if os.path.exists(CONV_PATH):
        with open(CONV_PATH, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        msg = json.loads(line)
                        if msg["sender"] == "User" and msg["timestamp"] > last_ts:
                            new_messages.append(msg)
                    except:
                        continue
    
    if new_messages:
        # Update last seen to the latest user message
        save_last_seen(new_messages[-1]["timestamp"])
        
        for msg in new_messages:
            # Format output for the agent to see
            print(f"--- NEW DASHBOARD MESSAGE ---")
            print(f"Time: {msg['timestamp']}")
            print(f"Message: {msg['text']}")
            print(f"-----------------------------")

if __name__ == "__main__":
    main()
