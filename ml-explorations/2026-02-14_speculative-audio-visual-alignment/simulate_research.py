import numpy as np
import matplotlib.pyplot as plt

def simulate_alignment_latency():
    # Model 1: Traditional (Serial)
    # Whisper -> R1 (Reasoning) -> Wan 2.1 (Generation)
    serial_stages = {
        'Audio Processing (Whisper)': 45,
        'Reasoning/Sync Logic (R1)': 120,
        'Latent Prep': 30,
        'Wan 2.1 Keyframe Gen': 450
    }
    
    # Model 2: Speculative (Parallel/Overlapped)
    # Whisper -> [Speculative Sync Kernel] -> Wan 2.1 Keyframe Gen
    speculative_stages = {
        'Audio Processing (Whisper)': 45,
        'Speculative Sync (Kernel)': 12, # Low-latency projection
        'Wan 2.1 Keyframe Gen': 450
    }
    
    # In speculative, Wan 2.1 starts earlier because the 'Reasoning' step is bypassed/speculated.
    # Total serial = sum(serial_stages)
    # Total speculative = 45 + 12 + 450 (but sync is done during Wan 2.1 startup)
    
    labels = ['Traditional (Serial)', 'Speculative (Lucca-v1)']
    total_latency = [sum(serial_stages.values()), speculative_stages['Audio Processing (Whisper)'] + speculative_stages['Speculative Sync (Kernel)'] + speculative_stages['Wan 2.1 Keyframe Gen']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, total_latency, color=['#4a4a4a', '#00ffcc'])
    plt.ylabel('Latency (ms)')
    plt.title('Audio-Visual Alignment Latency: Serial vs. Speculative (sm_120)')
    
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval}ms', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('ml-explorations/2026-02-14_speculative-audio-visual-alignment/plots/latency_comparison.png')
    
    # Plot 2: Accuracy vs Acceptance Rate
    thresholds = np.linspace(0.5, 0.99, 50)
    acceptance_rate = 0.95 * (1 - (thresholds - 0.5)**2) # Simulated decay
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, acceptance_rate * 100, color='#00ffcc', linewidth=2)
    plt.xlabel('Alignment Confidence Threshold')
    plt.ylabel('Speculative Acceptance Rate (%)')
    plt.title('Speculative Acceptance Efficiency for Lip-Sync Tokens')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('ml-explorations/2026-02-14_speculative-audio-visual-alignment/plots/acceptance_efficiency.png')

if __name__ == "__main__":
    simulate_alignment_latency()
