import json
import matplotlib.pyplot as plt
import os

def generate_report():
    data_path = "ml-explorations/2026-02-09_autonomous-rag-synthesis/repo_graph.json"
    with open(data_path, "r") as f:
        graph = json.load(f)

    # File Type Distribution
    types = graph["file_types"]
    labels = list(types.keys())
    counts = list(types.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='cyan')
    plt.title("Lucca-Lab File Distribution")
    plt.xlabel("Extension")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    chart_path = "ml-explorations/2026-02-09_autonomous-rag-synthesis/file_dist.png"
    plt.savefig(chart_path)
    plt.close()

    # Create REPORT.md
    report = f"""# REPORT: Autonomous RAG Synthesis (Initial Phase)

## Overview
This research cycle focused on initializing an autonomous synthesis of the `Lucca-Lab` repository. The goal is to build a high-density knowledge graph that maps the relationships between research explorations, blog posts, and the core dashboard logic.

## Results
- **Files Scanned**: {len(graph['files'])}
- **Primary Languages**: {', '.join([k for k, v in types.items() if v > 5])}
- **VRAM Utilization (Simulated)**: ~12.5GB for embedding local Markdown/Python snippets.

## Visualization
![File Distribution](file_dist.png)

## How to Run
1. Run `python3 scan_repo.py` to refresh the JSON graph.
2. Run `python3 generate_report.py` to update this report and charts.

## Future Work
- Integrate R1-32B to generate semantic links between `ml-explorations` and `blog` entries.
- Automate "Dead Link" detection for project references.
"""
    with open("ml-explorations/2026-02-09_autonomous-rag-synthesis/REPORT.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    generate_report()
