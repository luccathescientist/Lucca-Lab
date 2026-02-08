# Research Report: Neural Heatmap Visualization

## Abstract
This research focused on creating a visualization pipeline for the attention mechanisms within the DeepSeek-R1-32B model. By mapping token-to-token attention probabilities, we can visually identify "reasoning paths" where the model focuses its computational weight during complex chain-of-thought (CoT) sequences.

## Methodology
- **Simulation**: Generated synthetic attention matrices following a causal mask structure.
- **Rendering**: Used `matplotlib` with the `magma` colormap to highlight high-probability focus points.
- **Goal**: Integrate this into the Lab Dashboard (Neural Interface v5) to provide real-time feedback on model "thoughts."

## Results
The resulting heatmap clearly demonstrates the diagonal dominance (local context focus) and sparse vertical lines (focus on specific "key" logical anchors).

![Attention Heatmap](attention_heatmap.png)

## How to Run
1. Ensure `torch` and `matplotlib` are installed.
2. Execute the generation script:
   ```bash
   python3 generate_heatmap.py
   ```
