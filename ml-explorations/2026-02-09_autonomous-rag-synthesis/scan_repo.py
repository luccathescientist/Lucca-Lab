import os
import json
import time

def scan_repo(root_dir):
    knowledge_graph = {
        "files": [],
        "directories": [],
        "file_types": {},
        "summary": ""
    }
    
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden files and directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        knowledge_graph["directories"].append(os.path.relpath(root, root_dir))
        
        for file in files:
            if file.startswith('.'):
                continue
            
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            ext = os.path.splitext(file)[1]
            
            knowledge_graph["files"].append({
                "path": rel_path,
                "extension": ext,
                "size": os.path.getsize(file_path)
            })
            
            knowledge_graph["file_types"][ext] = knowledge_graph["file_types"].get(ext, 0) + 1

    return knowledge_graph

if __name__ == "__main__":
    start_time = time.time()
    repo_path = "/home/user/lab_env/Lucca-Lab"
    graph = scan_repo(repo_path)
    
    output_path = "ml-explorations/2026-02-09_autonomous-rag-synthesis/repo_graph.json"
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=4)
    
    elapsed = time.time() - start_time
    print(f"Scanned {len(graph['files'])} files in {elapsed:.2f}s")
