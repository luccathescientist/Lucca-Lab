# REPORT: Neural Knowledge Graph Pruning (Semantic Decay)

## Overview
This research explores a decay-based pruning algorithm for the Lab's Knowledge Graph. As the graph grows, retrieval latency increases linearly. By implementing a "Semantic Decay" mechanism that archives nodes based on a combination of time since last access and semantic relevance, we can maintain sub-100ms latency even as the total corpus grows.

## Technical Details
- **Algorithm**: $Score = Relevance \times e^{-\lambda t}$
- **Hardware**: Simulated on Blackwell sm_120 for similarity scoring.
- **Results**: Latency was capped at ~55ms by maintaining an active set of ~10,000 nodes, whereas unpruned latency projected towards 2.5s at 50,000 nodes.

## Latency Analysis
![Latency Chart](latency_chart.png)

## How to Run
1. Ensure `numpy` and `matplotlib` are installed.
2. Run `python3 pruning_sim.py`.
