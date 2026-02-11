import json
import random

def simulate_spatial_reasoning(prompt):
    """Simulates a spatial reasoning task performance."""
    # Logic: Prompts that mention 'coordinates' or 'bounding boxes' perform better.
    score = 0
    if "coordinates" in prompt.lower(): score += 0.4
    if "bounding box" in prompt.lower(): score += 0.3
    if "spatial" in prompt.lower(): score += 0.2
    score += random.uniform(0, 0.1)
    return min(score, 1.0)

def evolve_prompts(initial_prompts, generations=5):
    current_prompts = initial_prompts
    history = []
    
    for gen in range(generations):
        scored = [(p, simulate_spatial_reasoning(p)) for p in current_prompts]
        scored.sort(key=lambda x: x[1], reverse=True)
        history.append(scored[0])
        
        # Keep top 2, mutate
        next_gen = [scored[0][0], scored[1][0]]
        for _ in range(3):
            parent = random.choice(next_gen)
            mutation = parent + " Use coordinates." if "coordinates" not in parent else parent + " Focus on bounding boxes."
            next_gen.append(mutation)
        current_prompts = next_gen
        
    return history

if __name__ == "__main__":
    seeds = [
        "What is in this image?",
        "Describe the objects in the scene.",
        "Analyze the spatial layout of the room.",
        "Where are the chairs located?",
        "Identify and locate every item."
    ]
    
    results = evolve_prompts(seeds)
    with open("evolution_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Evolution complete. Best prompt:", results[-1][0])
