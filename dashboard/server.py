import json
import os
import subprocess
import asyncio
import glob
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

# Try to load semantic search dependencies
try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    package_error = None
except ImportError as e:
    package_error = str(e)

app = FastAPI()

# Configuration
PORT = 8889
CHROMA_PATH = "/home/the_host/clawd/deep-wisdom/db"
MODEL_NAME = "all-MiniLM-L6-v2"
VLLM_URL = "http://localhost:8001/v1"

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
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

    async def set_responding(self, status: bool):
        self.is_responding = status
        await self.broadcast({"type": "status", "is_responding": status})

manager = ConnectionManager()
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if package_error:
            raise ImportError(f"Package missing: {package_error}. Ensure server runs in the correct venv.")
        _embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embeddings

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

@app.get("/")
async def get():
    return FileResponse("/home/the_host/clawd/dashboard/index.html", headers={"Cache-Control": "no-cache"})

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

@app.get("/api/search")
async def search(q: str):
    try:
        embeddings = get_embeddings()
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        results = vector_db.similarity_search(q, k=5)
        return [{"content": r.page_content, "metadata": r.metadata} for r in results]
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
                    try: msgs.append(json.loads(line))
                    except: pass
    return msgs

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await websocket.send_json({"type": "status", "is_responding": manager.is_responding})
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            entry = {"timestamp": datetime.now().isoformat(), "sender": "User", "text": message.get("text", "")}
            with open("/home/the_host/clawd/dashboard/conversation.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            await manager.broadcast({"type": "message", "msg": entry})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/inference")
async def inference_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            model_id = req.get("model")
            prompt = req.get("prompt")
            
            await manager.set_responding(True)
            
            # Check if persistent vLLM is up
            try:
                vllm_alive = requests.get(f"{VLLM_URL}/models").status_code == 200
            except:
                vllm_alive = False

            if vllm_alive and (model_id == "deepseek-70b-fp8" or model_id == "deepseek-70b"):
                # Use persistent vLLM
                # R1 models need chat template. For simplicity, we use chat endpoint.
                payload = {
                    "model": "r1-70b-fp8",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "max_tokens": 1024
                }
                response = requests.post(f"{VLLM_URL}/chat/completions", json=payload, stream=True)
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            if "[DONE]" in line_text: break
                            try:
                                chunk = json.loads(line_text[6:])
                                if chunk['choices'][0]['delta'].get('content'):
                                    await websocket.send_json({"type": "token", "text": chunk['choices'][0]['delta']['content']})
                            except: pass
            else:
                # Fallback to subprocess for other models or if vLLM is down
                process = subprocess.Popen(
                    ["/home/the_host/workspace/pytorch_cuda/.venv/bin/python3", "/home/the_host/clawd/dashboard/run_inf.py", model_id, prompt],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                for line in iter(process.stdout.readline, ''):
                    if line: await websocket.send_json({"type": "token", "text": line})
                process.stdout.close()
                process.wait()
            
            await manager.set_responding(False)
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass

@app.post("/api/chat/ai")
async def ai_message(msg: ChatMessage):
    entry = {"timestamp": datetime.now().isoformat(), "sender": "AI", "text": msg.text}
    with open("/home/the_host/clawd/dashboard/conversation.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    await manager.broadcast({"type": "message", "msg": entry})
    return {"status": "sent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
