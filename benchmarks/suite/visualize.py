
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add local site packages
sys.path.append(os.path.abspath('./local_site'))

def generate_benchmark_chart(json_files, output_path):
    data = []
    for f in json_files:
        with open(f, 'r') as file:
            data.append(json.load(file))
    
    df = pd.DataFrame(data)
    
    # Sort by tokens per second
    df = df.sort_values('tokens_per_sec', ascending=False)
    # Ensure model names are strings for plotting
    df['model'] = df['model'].astype(str)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['model'], df['tokens_per_sec'], color=['#7b2cbf', '#5a189a', '#3c096c'][:len(df)])
    
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Tokens Per Second', fontweight='bold')
    plt.title('Blackwell Performance: Tokens/Sec by Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    # Find all report files
    reports = [f for f in os.listdir('.') if f.startswith('report_') and f.endswith('.json')]
    if reports:
        generate_benchmark_chart(reports, 'benchmark_comparison.png')
    else:
        print("No report files found.")
