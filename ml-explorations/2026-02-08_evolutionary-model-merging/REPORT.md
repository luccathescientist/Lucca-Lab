# Research Report: Evolutionary Model Merging

## Objective
To determine the optimal merging ratio (alpha) for two specialized Llama-3 models using a simulated evolutionary search.

## Methodology
- Architecture: SLERP-inspired (Spherical Linear Interpolation) weight blending.
- Hardware: NVIDIA Blackwell RTX 6000 (Simulated).
- Generations: 10

## Results
- **Best Alpha**: 0.5544
- **Peak Fitness**: 0.8859

![Fitness Landscape](fitness_landscape.png)

## How to Run
```bash
python3 merge_sim.py
```
