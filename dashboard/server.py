import json
import os
import subprocess
import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import glob

app = FastAPI()

# Mount assets
os.makedirs("/home/the_host/clawd/dashboard/assets", exist_ok=True)
app.mount("/assets", StaticFiles(directory="/home/the_host/clawd/dashboard/assets"), name="assets")

class ChatMessage(BaseModel):
    text: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.is_responding = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

    async def set_responding(self, status: bool):
        self.is_responding = status
        await self.broadcast({"type": "status", "is_responding": status})

    async def log(self, content: str):
        await self.broadcast({"type": "log", "content": content})

manager = ConnectionManager()

def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        parts = [p.strip() for p in result.split(',')]
        return {
            "name": parts[0],
            "temp": float(parts[1]),
            "util_gpu": float(parts[2]),
            "util_mem": float(parts[3]),
            "mem_total": float(parts[4]),
            "mem_used": float(parts[5]),
            "power": float(parts[6])
        }
    except Exception as e:
        return {"error": str(e)}

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import threading
import queue

CHROMA_PATH = "/home/the_host/clawd/deep-wisdom/db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Global embedding model
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embeddings

# Inference process management
class InferenceManager:
    def __init__(self):
        self.active_process = None
        self.output_queue = queue.Queue()

manager = ConnectionManager()
inf_manager = InferenceManager()

@app.get("/api/models")
async def list_models():
    return [
        {"id": "deepseek-70b", "name": "DeepSeek-R1-Distill-Llama-70B (4-bit)", "description": "High reasoning depth, 15 TPS"},
        {"id": "qwen-1.5b", "name": "Qwen2.5-1.5B-Instruct", "description": "Lightning fast, 110 TPS"},
        {"id": "deepseek-8b", "name": "DeepSeek-R1-Distill-Qwen-8B", "description": "Balanced speed/reasoning, 45 TPS"}
    ]

@app.websocket("/ws/inference")
async def inference_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            model_id = req.get("model")
            prompt = req.get("prompt")
            
            # Start inference script in subprocess
            # We'll use a helper script dashboard/run_inf.py
            process = subprocess.Popen(
                [
                    "/home/the_host/workspace/pytorch_cuda/.venv/bin/python3",
                    "/home/the_host/clawd/dashboard/run_inf.py",
                    model_id,
                    prompt
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            await manager.set_responding(True)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    await websocket.send_json({"type": "token", "text": line})
            
            process.stdout.close()
            process.wait()
            
            await manager.set_responding(False)
            await websocket.send_json({"type": "done"})
            
    except WebSocketDisconnect:
        pass

@app.get("/api/search")
async def search(q: str):
    try:
        embeddings = get_embeddings()
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        results = vector_db.similarity_search(q, k=5)
        
        serialized_results = []
        for r in results:
            serialized_results.append({
                "content": r.page_content,
                "metadata": r.metadata
            })
        return serialized_results
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def get():
    return FileResponse(
        "/home/the_host/clawd/dashboard/index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/api/stats")
async def stats():
    return get_gpu_stats()

@app.get("/api/memory")
async def memory():
    try:
        with open("/home/the_host/clawd/MEMORY.md", "r") as f:
            memory_md = f.read()
        daily_files = sorted(glob.glob("/home/the_host/clawd/memory/2026-*.md"))
        latest_daily = ""
        if daily_files:
            with open(daily_files[-1], "r") as f:
                latest_daily = f.read()
        return {"memory_md": memory_md, "latest_daily": latest_daily}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/messages")
async def messages():
    conv_path = "/home/the_host/clawd/dashboard/conversation.jsonl"
    msgs = []
    if os.path.exists(conv_path):
        with open(conv_path, "r") as f:
            for line in f:
                if line.strip():
                    msgs.append(json.loads(line))
    return msgs

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.active_connections.append(websocket)
    # Send initial status
    await websocket.send_json({"type": "status", "is_responding": manager.is_responding})
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Save message
            conv_path = "/home/the_host/clawd/dashboard/conversation.jsonl"
            entry = {
                "timestamp": datetime.now().isoformat(),
                "sender": "User",
                "text": message.get("text", "")
            }
            with open(conv_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            # Broadcast to everyone
            await manager.broadcast({"type": "message", "msg": entry})
            
    except WebSocketDisconnect:
        manager.active_connections.remove(websocket)

# Helper to trigger responding state
@app.post("/api/status/responding")
async def set_responding(status: bool):
    await manager.set_responding(status)
    return {"status": "ok"}

# Helper to send AI message
@app.post("/api/chat/ai")
async def ai_message(msg: ChatMessage):
    conv_path = "/home/the_host/clawd/dashboard/conversation.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "sender": "AI",
        "text": msg.text
    }
    with open(conv_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    await manager.broadcast({"type": "message", "msg": entry})
    return {"status": "sent"}

# Helper to inject external logs
@app.post("/api/log")
async def inject_log(msg: ChatMessage):
    await manager.log(msg.text)
    return {"status": "logged"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
