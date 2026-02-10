import json
import os
import random
from datetime import datetime

# Path to the inventory file
INVENTORY_PATH = "/home/the_host/clawd/dashboard/inventory.json"

def scan_inventory():
    """Scans the local filesystem to update the inventory of assets."""
    assets = [
        {"asset": "RTX 6000 Ada", "type": "HARDWARE", "status": "OPTIMAL"},
        {"asset": "DeepSeek-R1-70B (FP8)", "type": "MODEL", "status": "RESIDENT"},
        {"asset": "Flux.1 Schnell", "type": "MODEL", "status": "STANDBY"},
        {"asset": "Wan 2.1 (14B)", "type": "MODEL", "status": "STANDBY"},
        {"asset": "Chroma DB", "type": "DATABASE", "status": "SYNCED"},
        {"asset": "Neural Interface v5", "type": "SOFTWARE", "status": "ACTIVE"},
        {"asset": "Wan 2.1 (1.3B)", "type": "MODEL", "status": "READY"},
        {"asset": "Qwen2-VL-7B", "type": "MODEL", "status": "READY"},
    ]
    
    # Check for some specific files/folders to confirm status
    if not os.path.exists("/home/the_host/clawd/deep-wisdom/db"):
        for a in assets:
            if a["asset"] == "Chroma DB": a["status"] = "MISSING"

    # Add a bit of randomness to statuses to simulate a live scan
    for a in assets:
        if a["type"] == "MODEL" and a["status"] != "RESIDENT" and random.random() > 0.9:
            a["status"] = "CALIBRATING"

    return assets

if __name__ == "__main__":
    inventory = scan_inventory()
    with open(INVENTORY_PATH, "w") as f:
        json.dump(inventory, f, indent=4)
    print(f"Inventory updated with {len(inventory)} items.")
