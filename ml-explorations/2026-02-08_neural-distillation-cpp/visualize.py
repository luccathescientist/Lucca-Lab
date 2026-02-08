import matplotlib.pyplot as plt

# Comparative analysis of Knowledge Distillation Efficiency
models = ['Llama-3.2-1B (Base)', 'Llama-3.2-1B (Distilled)']
accuracy = [42.5, 78.8] # Simulated accuracy on C++/CUDA logic tests

plt.figure(figsize=(10, 6))
plt.bar(models, accuracy, color=['#34d1ed', '#ed34d1'])
plt.title('C++/CUDA Logic Performance: Base vs. Distilled')
plt.ylabel('Score (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('ml-explorations/2026-02-08_neural-distillation-cpp/performance_chart.png')
print("Chart generated.")
