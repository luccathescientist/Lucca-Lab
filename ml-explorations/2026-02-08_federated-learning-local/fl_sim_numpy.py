import numpy as np
from collections import OrderedDict

# Simulation of a simple local model using NumPy (since PyTorch sm_120 kernel is missing)
class LocalModel:
    def __init__(self):
        # Linear layer: weights (1, 10), bias (1)
        self.weights = np.random.randn(1, 10)
        self.bias = np.random.randn(1)

    def get_weights(self):
        return {"weights": self.weights.copy(), "bias": self.bias.copy()}

    def load_weights(self, weights_dict):
        self.weights = weights_dict["weights"].copy()
        self.bias = weights_dict["bias"].copy()

def federated_averaging(model_weights_list):
    """
    Simulates FedAvg: Averages weights from multiple local nodes.
    """
    avg_weights = {}
    num_nodes = len(model_weights_list)
    
    for key in model_weights_list[0].keys():
        avg_weights[key] = sum([weights[key] for weights in model_weights_list]) / num_nodes
        
    return avg_weights

def simulate_fl():
    print("Initializing Federated Learning Simulation (NumPy Fallback)...")
    
    # Node 1: Rig A (Local)
    node_a = LocalModel()
    # Node 2: Rig B (Simulated Remote)
    node_b = LocalModel()
    
    # Initial weights
    weights_a = node_a.get_weights()
    weights_b = node_b.get_weights()
    
    print(f"Node A initial weights sample: {weights_a['weights'][0][0]:.4f}")
    print(f"Node B initial weights sample: {weights_b['weights'][0][0]:.4f}")
    
    # Federated Averaging
    global_weights = federated_averaging([weights_a, weights_b])
    
    # Update Node A with global weights
    node_a.load_weights(global_weights)
    updated_weights_a = node_a.get_weights()
    
    print(f"Global Average sample: {global_weights['weights'][0][0]:.4f}")
    print(f"Node A updated weights sample: {updated_weights_a['weights'][0][0]:.4f}")
    
    if np.allclose(updated_weights_a['weights'], global_weights['weights']):
        print("SUCCESS: Weights synchronized correctly across simulated local nodes.")
    else:
        print("FAILURE: Weight mismatch detected.")

if __name__ == "__main__":
    simulate_fl()
