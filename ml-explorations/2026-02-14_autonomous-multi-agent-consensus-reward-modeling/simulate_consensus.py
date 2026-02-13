import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Simulated data for the "Council of Agents" Consensus
models = ["DeepSeek-R1", "Qwen-2.5-72B", "Llama-3.1-70B"]
iterations = 100

# Reward signals (0.0 to 1.0)
r1_signals = np.random.normal(0.75, 0.05, iterations)
qwen_signals = np.random.normal(0.72, 0.06, iterations)
llama_signals = np.random.normal(0.70, 0.07, iterations)

# Consensus (Weighted Average)
consensus = (r1_signals * 0.5 + qwen_signals * 0.3 + llama_signals * 0.2)
variance = np.var([r1_signals, qwen_signals, llama_signals], axis=0)

# Save simulated raw data
data = {
    "iterations": iterations,
    "r1_mean": np.mean(r1_signals),
    "qwen_mean": np.mean(qwen_signals),
    "llama_mean": np.mean(llama_signals),
    "consensus_mean": np.mean(consensus),
    "avg_variance": np.mean(variance)
}

os.makedirs("ml-explorations/2026-02-14_autonomous-multi-agent-consensus-reward-modeling/data", exist_ok=True)
with open("ml-explorations/2026-02-14_autonomous-multi-agent-consensus-reward-modeling/data/results.json", "w") as f:
    json.dump(data, f, indent=4)

# Generate Chart 1: Signal Distribution
plt.figure(figsize=(10, 6))
plt.hist(r1_signals, alpha=0.5, label="DeepSeek-R1 (Teacher)", bins=20)
plt.hist(qwen_signals, alpha=0.5, label="Qwen-2.5", bins=20)
plt.hist(llama_signals, alpha=0.5, label="Llama-3.1", bins=20)
plt.axvline(np.mean(consensus), color='red', linestyle='dashed', linewidth=2, label="Weighted Consensus")
plt.title("Reward Signal Distribution - Council of Agents")
plt.xlabel("Reward Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("ml-explorations/2026-02-14_autonomous-multi-agent-consensus-reward-modeling/reward_distribution.png")

# Generate Chart 2: Consensus Variance over Iterations
plt.figure(figsize=(10, 6))
plt.plot(variance, color='purple', label="Consensus Variance")
plt.axhline(np.mean(variance), color='black', linestyle='--', label="Avg Variance")
plt.title("Consensus Variance (Logical Agreement)")
plt.xlabel("DPO Pair Iteration")
plt.ylabel("Variance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("ml-explorations/2026-02-14_autonomous-multi-agent-consensus-reward-modeling/consensus_variance.png")

print("Research simulation complete. Files generated in project folder.")
