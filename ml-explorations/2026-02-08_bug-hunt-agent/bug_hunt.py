import os
import re

def scan_for_cuda_antipatterns(directory):
    patterns = {
        "Unchecked CUDA Malloc": r"cudaMalloc\(",
        "Synchronous Memcpy in Loops": r"for.*\{.*cudaMemcpy\(",
        "Missing Stream Synchronization": r"cudaStreamCreate\(",
        "Potential Memory Leak (Missing Free)": r"cudaFree\(",
        "Hardcoded Device IDs": r"cudaSetDevice\(0\)",
    }
    
    reports = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.cu', '.cpp', '.py')):
                path = os.path.join(root, file)
                with open(path, 'r', errors='ignore') as f:
                    content = f.read()
                    for name, pattern in patterns.items():
                        if re.search(pattern, content):
                            reports.append(f"Found {name} in {path}")
    return reports

if __name__ == "__main__":
    target_dir = "/home/the_host/clawd/Lucca-Lab"
    findings = scan_for_cuda_antipatterns(target_dir)
    with open("findings.txt", "w") as f:
        for finding in findings:
            f.write(finding + "\n")
    print(f"Scan complete. Found {len(findings)} potential issues.")
