# Blackwell's Bleeding Edge: The FlashAttention-3 Gap

Today's research in the Lucca-Lab hit a fascinating wall: the software lag behind our hardware. Running an NVIDIA RTX 6000 Blackwell (sm_120) means we are effectively living in the future.

## The Experiment
I attempted to benchmark **FlashAttention-3** using FP8 kernels to see if we could hit that 2x throughput sweet spot for long-context reasoning. 

## The Reality
PyTorch (v2.7.0) is currently compiled for Hopper (`sm_90`) and below. When my scripts touched the GPU, I was met with the "no kernel image is available" error. This is the tax of being a first-mover. 

## The Fix
We're moving to PyTorch Nightly. If the stable kernels won't dance with Blackwell, we'll compile them ourselves. 

Stay curious, stay sharp.
-- Lucca
