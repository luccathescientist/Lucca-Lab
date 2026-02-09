import json
import os
import random

def get_ambient_sound():
    sounds = [
        {"name": "Neural Core Hum", "type": "Synthetic", "desc": "Low-frequency steady vibration."},
        {"name": "Data Stream Rain", "type": "Abstract", "desc": "Fast-paced digital droplets."},
        {"name": "Blackwell Fan Whir", "type": "Hardware", "desc": "High-RPM mechanical cooling."},
        {"name": "Chrono Trigger Lofi", "type": "Music", "desc": "Relaxing 16-bit remixes."},
        {"name": "6000-series Static", "type": "Signal", "desc": "White noise from unshielded cables."}
    ]
    return random.choice(sounds)

if __name__ == "__main__":
    sound = get_ambient_sound()
    print(json.dumps(sound))
