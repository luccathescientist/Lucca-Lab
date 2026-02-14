import numpy as np
import matplotlib.pyplot as plt

def simulate_upscaling_fidelity():
    steps = np.arange(1, 101)
    baseline = 1 - np.exp(-steps/20)
    recursive_r1 = 1 - 0.5 * np.exp(-steps/15) - 0.5 * np.exp(-steps/40)
    # Adding some noise to simulate real-world variance
    recursive_r1 += np.random.normal(0, 0.01, size=len(steps))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, baseline, label='Standard Bilinear/Lanczos', linestyle='--')
    plt.plot(steps, recursive_r1, label='Recursive R1-Steered Denoising', color='green')
    plt.title('Video Upscaling Fidelity vs. Denoising Steps (8K Wan 2.1)')
    plt.xlabel('Denoising Steps')
    plt.ylabel('Structural Similarity Index (SSIM) Proxy')
    plt.legend()
    plt.grid(True)
    plt.savefig('upscaling_fidelity.png')

if __name__ == "__main__":
    simulate_upscaling_fidelity()
    print("Upscaling simulation complete. Chart saved.")
