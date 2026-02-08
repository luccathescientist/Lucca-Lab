import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# Simulation of a simple local model for Federated Learning testing
class LocalModel(nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def get_weights(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def federated_averaging(model_weights_list):
    """
    Simulates FedAvg: Averages weights from multiple local nodes.
    """
    avg_weights = OrderedDict()
    num_nodes = len(model_weights_list)
    
    for key in model_weights_list[0].keys():
        avg_weights[key] = sum([weights[key] for weights in model_weights_list]) / num_nodes
        
    return avg_weights

def simulate_fl():
    print("Initializing Federated Learning Simulation on Blackwell...")
    
    # Node 1: Rig A (Local)
    model_a = LocalModel().cuda()
    # Node 2: Rig B (Simulated Remote)
    model_b = LocalModel().cuda()
    
    # Initial weights
    weights_a = get_weights(model_a)
    weights_b = get_weights(model_b)
    
    print(f"Node A initial weights sample: {weights_a['fc.weight'][0][0]:.4f}")
    print(f"Node B initial weights sample: {weights_b['fc.weight'][0][0]:.4f}")
    
    # Federated Averaging
    global_weights = federated_averaging([weights_a, weights_b])
    
    # Update Node A with global weights
    model_a.load_state_dict(global_weights)
    updated_weights_a = get_weights(model_a)
    
    print(f"Global Average sample: {global_weights['fc.weight'][0][0]:.4f}")
    print(f"Node A updated weights sample: {updated_weights_a['fc.weight'][0][0]:.4f}")
    
    if torch.allclose(updated_weights_a['fc.weight'], global_weights['fc.weight']):
        print("SUCCESS: Weights synchronized correctly.")
    else:
        print("FAILURE: Weight mismatch detected.")

if __name__ == "__main__":
    simulate_fl()
