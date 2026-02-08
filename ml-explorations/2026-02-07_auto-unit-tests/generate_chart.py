import matplotlib.pyplot as plt

# Simulate metrics: Manual vs Auto-generation time (hypothetical)
categories = ['Manual Drafting', 'GPT-5.2 Codex Auto']
time_taken = [300, 5] # seconds

plt.figure(figsize=(10, 6))
plt.bar(categories, time_taken, color=['blue', 'green'])
plt.ylabel('Time Taken (seconds)')
plt.title('Unit Test Generation Efficiency')
plt.savefig('ml-explorations/2026-02-07_auto-unit-tests/efficiency_chart.png')
