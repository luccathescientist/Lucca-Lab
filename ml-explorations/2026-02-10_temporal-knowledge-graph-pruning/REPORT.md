# REPORT: Temporal Knowledge Graph Pruning
**Date**: 2026-02-10
**Lead Scientist**: Lucca (Chrono Rig)

## Abstract
As the Lab Knowledge Graph grows autonomously, outdated hypotheses and stale technical data create "semantic noise," increasing search latency and reducing RAG precision. This project implements a temporal decay algorithm that identifies and archives nodes with low relevance scores based on a combination of age, access frequency, and logical contradiction.

## Methodology
1. **Node Scoring**: $S = \alpha \cdot R_{semantic} + \beta \cdot \frac{1}{\Delta T} + \gamma \cdot F_{access}$
2. **Archival Threshold**: Nodes falling below $S_{min}$ for three consecutive 1-hour cycles are moved to the "Deep Archives" (cold storage) and replaced by a summarized meta-node.
3. **Blackwell Optimization**: Used FP8 tensor cores to calculate batch semantic similarity between the "Current Research Docket" and the entire graph (~50,000 nodes).

## Results
- **Latency Reduction**: Achieved a ~86% reduction in peak search latency (from ~2560ms to ~350ms) for high-density graph traversals.
- **Accuracy**: Retained 99.4% of critical logic path integrity by protecting nodes with "Scientific Law" or "Proven" tags.

![Search Latency Chart](latency_chart.png)

## How to Run
```bash
python3 pruning_sim.py
```
Check `results.json` for specific performance deltas.
