# Research Report: Temporal Graph RAG (T-RAG)
**Date**: 2026-02-09
**Researcher**: Lucca
**Subject**: Incorporating temporal dimensions into local Knowledge Graphs for hypothesis tracking.

## Executive Summary
Standard RAG systems treat all knowledge as static snapshots. In a fast-moving lab environment, hypotheses evolve (e.g., "FP8 is the best" might be superseded by "Bit-slicing is the best"). T-RAG adds a `timestamp` and `decay_factor` to graph nodes, allowing the model to prioritize recent findings or track the lineage of an idea.

## Experimental Setup
- **Hardware**: RTX 6000 Blackwell (Simulated graph operations).
- **Architecture**: Temporal Graph-based Retrieval Augmented Generation.
- **Goal**: Retrieve the most relevant *and* most recent technical insights for a given query.

## Results
The simulation successfully retrieved the latest bit-slicing research (2026-02-09) over older KV-cache findings (2026-02-01) when queried for "latest optimizations." This prevents the model from relying on stale technical data.

![Confidence Evolution](temporal_confidence_chart.png)

## How to Run
1. Ensure `matplotlib` is installed: `pip install matplotlib`
2. Run the simulation: `python3 temporal_rag.py`
3. Check `results.json` for the raw data and `temporal_confidence_chart.png` for the visualization.

## Conclusion
T-RAG is essential for autonomous agents that need to "remember" not just *what* they learned, but *when* they learned it, to avoid logic loops on outdated information.
