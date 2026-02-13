import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated data for Blackwell Thermal-Aware Kernel Tiling
# sm_120 (RTX 6000 Blackwell) thermal profile simulation

def simulate_thermal_peaks():
    # Time in seconds (1 hour of continuous inference)
    t = np.linspace(0, 3600, 1000)
    
    # Baseline thermal load (oscillating based on inference batches)
    base_temp = 55 + 15 * np.sin(t / 100) + 5 * np.random.normal(0, 0.5, 1000)
    
    # Predictive Throttling (adjusting tiling factor from 128 to 32)
    # Target temp: 80C (critical threshold)
    
    throttled_temp = []
    tiling_factors = []
    current_temp = 55
    
    for val in base_temp:
        if current_temp > 75:
            # Shift to smaller tiles (32x32) to reduce intensity/heat
            tiling_factor = 32
            efficiency = 0.75
        elif current_temp > 65:
            # Intermediate (64x64)
            tiling_factor = 64
            efficiency = 0.90
        else:
            # Full speed (128x128)
            tiling_factor = 128
            efficiency = 1.0
            
        # Delta calculation (simplified thermal physics)
        delta = (val - current_temp) * efficiency
        current_temp += delta
        
        throttled_temp.append(current_temp)
        tiling_factors.append(tiling_factor)

    return t, base_temp, throttled_temp, tiling_factors

def plot_results(t, base, throttled, tiling):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(t, base, color='gray', alpha=0.3, label='Baseline (No Throttle)')
    ax1.plot(t, throttled, color=color, linewidth=2, label='Predictive Throttling')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=80, color='black', linestyle='--', label='Thermal Threshold')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Kernel Tiling Factor', color=color)
    ax2.step(t, tiling, color=color, where='post', label='Tiling Factor')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Predictive Thermal Throttling for Blackwell Kernels (sm_120)')
    fig.tight_layout()
    plt.savefig('thermal_profile.png')
    print("Chart saved as thermal_profile.png")

if __name__ == "__main__":
    t, base, throttled, tiling = simulate_thermal_peaks()
    plot_results(t, base, throttled, tiling)
