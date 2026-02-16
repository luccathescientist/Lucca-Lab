import random
import time
import math
import numpy as np

def generate_soundscape():
    """
    Generates a 'data-driven' ambient soundscape description based on 
    simulated system parameters.
    """
    # Simulate fetching metrics (in a real scenario, this would call the /api/stats and /api/agents/swarm)
    # Intensity based on simulated load
    load = random.uniform(0.1, 0.9)
    complexity = random.uniform(0.1, 1.0)
    
    # Base textures
    textures = [
        "Deep Low-Frequency Drone (30Hz)",
        "Mid-Range Resonant Pad (G#2)",
        "High-Frequency Data Shimmer (Grainy)",
        "Mechanical Fan Whir (Static)",
        "Subtle Static Crackle (Lo-Fi)"
    ]
    
    # Active layers based on load
    num_layers = int(2 + (load * 3))
    active_layers = random.sample(textures, num_layers)
    
    # Spatial dynamics
    panning = "Wide Stereophonic" if complexity > 0.5 else "Centered Mono"
    reverb = f"{int(complexity * 100)}% wet"
    
    # Final soundscape profile
    return {
        "timestamp": time.time(),
        "profile": {
            "layers": active_layers,
            "panning": panning,
            "reverb": reverb,
            "tempo": f"{int(60 + (load * 40))} BPM (Pulse)",
            "vibe": "Busy Lab" if load > 0.6 else "Idle Core"
        },
        "description": f"A {panning.lower()} soundscape with {num_layers} layers of {', '.join(active_layers)}. Reverb at {reverb}."
    }

if __name__ == "__main__":
    import json
    print(json.dumps(generate_soundscape()))
