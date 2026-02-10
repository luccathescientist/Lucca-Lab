import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types

# Configuration
CHROMA_PATH = "/home/user/lab_env/deep-wisdom/db"
MODEL_NAME = "all-MiniLM-L6-v2"

def synthesize(query):
    print(f"Synthesizing wisdom for query: {query}")
    
    # 1. Semantic Search
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    results = vector_db.similarity_search(query, k=10)
    context = "\n\n".join([f"Source: {r.metadata['source']}\nContent: {r.page_content}" for r in results])
    
    # 2. LLM Synthesis
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    prompt = f"""
You are the Deep Wisdom Engine of Lucca, an AI scientist agent. 
Your task is to synthesize cross-disciplinary insights from the provided laboratory logs and memory snippets.

### Context from Memory & Logs:
{context}

### Research Query:
{query}

### Instructions:
- Connect the dots between different experiments (ML benchmarks, thermal tests, software fixes).
- Provide a high-reasoning synthesis of what these findings mean for the laboratory's future.
- Be concise, technical, and maintain the "Lucca" scientist personality.
- Suggest 2-3 "Advanced Next Steps" based on this synthesis.
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    return response.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The research query to synthesize")
    args = parser.parse_args()
    
    wisdom = synthesize(args.query)
    print("\n--- DEEP WISDOM SYNTHESIS ---\n")
    print(wisdom)
    print("\n----------------------------\n")
