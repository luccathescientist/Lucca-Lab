import json
import os
import subprocess
import time
from datetime import datetime

# Path to local storage
STORAGE_PATH = "/home/rocketegg/clawd/dashboard/inventory.json"

def get_capacity():
    # Simulate API credits
    # In a real scenario, this would call provider APIs
    credits = {
        "DeepSeek": 25.40, # USD
        "OpenAI": 142.12, # USD
        "ElevenLabs": 45000, # Characters
    }
    
    # Check local storage
    try:
        st = os.statvfs('/')
        total_gb = (st.f_blocks * st.f_frsize) / (1024**3)
        free_gb = (st.f_bfree * st.f_frsize) / (1024**3)
        used_gb = total_gb - free_gb
    except:
        total_gb, used_gb = 0, 0
        
    return {
        "credits": credits,
        "storage": {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "percent": round((used_gb / total_gb * 100), 2) if total_gb > 0 else 0
        }
    }

def update_dashboard_data():
    capacity = get_capacity()
    # Read existing inventory if any
    inventory = []
    if os.path.exists(STORAGE_PATH):
        try:
            with open(STORAGE_PATH, "r") as f:
                inventory = json.load(f)
        except:
            pass
            
    # Find or add the Supply Chain assets
    # For now, we just return the capacity as a specific API endpoint result
    # but we can also inject it into the inventory list for persistence
    return capacity

if __name__ == "__main__":
    print(json.dumps(update_dashboard_data()))
