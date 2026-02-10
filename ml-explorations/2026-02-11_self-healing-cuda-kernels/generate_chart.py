import matplotlib.pyplot as plt
import json

# Simulated data
data = {
    "stages": ["Initial", "Error Detected", "Reasoning", "Healed"],
    "throughput": [0, 0, 0, 85], # Simulated percentage of peak
    "vram_pressure": [95, 100, 100, 40]
}

plt.figure(figsize=(10, 6))
plt.plot(data["stages"], data["throughput"], marker='o', label='Throughput (%)')
plt.plot(data["stages"], data["vram_pressure"], marker='x', label='VRAM Pressure (%)')
plt.title('Self-Healing CUDA Kernel: Recovery Profile')
plt.xlabel('Cycle Phase')
plt.ylabel('Percentage')
plt.legend()
plt.grid(True)
plt.savefig('ml-explorations/2026-02-11_self-healing-cuda-kernels/recovery_chart.png')
