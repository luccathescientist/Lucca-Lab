import os
import json
import random
import time
from datetime import datetime

# Path to state file
BOT_STATE_PATH = "/home/rocketegg/clawd/dashboard/bot_state.json"

def get_bot_state():
    if os.path.exists(BOT_STATE_PATH):
        try:
            with open(BOT_STATE_PATH, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "x": 50,
        "y": 50,
        "task": "IDLE",
        "battery": 100,
        "last_update": time.time()
    }

def save_bot_state(state):
    state["last_update"] = time.time()
    with open(BOT_STATE_PATH, "w") as f:
        json.dump(state, f)

def update_bot():
    state = get_bot_state()
    
    # Simple movement logic
    # The bot "patrols" the dashboard (0-100 scale)
    dx = random.randint(-10, 10)
    dy = random.randint(-10, 10)
    
    state["x"] = max(0, min(100, state["x"] + dx))
    state["y"] = max(0, min(100, state["y"] + dy))
    
    # Task switching
    tasks = ["IDLE", "PATROLLING", "CLEANING_CACHE", "DEFRAGGING_VRAM", "SCANNING_TEMP", "OPTIMIZING_BUS"]
    if random.random() > 0.8:
        state["task"] = random.choice(tasks)
        
    # Battery consumption
    state["battery"] = max(0, state["battery"] - random.uniform(0.1, 0.5))
    if state["battery"] < 10:
        state["task"] = "CHARGING"
        state["battery"] += 5
        
    save_bot_state(state)
    return state

if __name__ == "__main__":
    print(json.dumps(update_bot()))
