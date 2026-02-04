import os
import glob
import json
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
CHROMA_PATH = "/home/the_host/clawd/deep-wisdom/db"
SOURCE_DIR = "/home/the_host/clawd"
MODEL_NAME = "all-MiniLM-L6-v2"

def get_files():
    files = []
    # MEMORY.md
    files.append(os.path.join(SOURCE_DIR, "MEMORY.md"))
    # memory/*.md
    files.extend(glob.glob(os.path.join(SOURCE_DIR, "memory", "*.md")))
    # diary/*.md
    files.extend(glob.glob(os.path.join(SOURCE_DIR, "memory", "diary", "*.md")))
    # ml-explorations/*/REPORT.md
    files.extend(glob.glob(os.path.join(SOURCE_DIR, "ml-explorations", "*", "REPORT.md")))
    return [f for f in files if os.path.exists(f)]

def index_knowledge():
    print("Initializing Deep Wisdom Indexing...")
    
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    all_splits = []
    
    for file_path in get_files():
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            splits = markdown_splitter.split_text(content)
            for s in splits:
                s.metadata["source"] = file_path
                s.metadata["indexed_at"] = datetime.now().isoformat()
            
            all_splits.extend(splits)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if not all_splits:
        print("No knowledge found to index.")
        return

    print(f"Indexing {len(all_splits)} segments into ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Indexing complete. Deep Wisdom Engine is now semantically aware.")

if __name__ == "__main__":
    index_knowledge()
