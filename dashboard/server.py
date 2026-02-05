import json
import os
import subprocess
import asyncio
import glob
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
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
INACTIVITY_LIMIT = 300 # 5 minutes

# State Management
class LabState:
    def __init__(self):
        self.vllm_process = None
        self.last_activity = datetime.now()
        self.is_loading = False

state = LabState()

async def start_vllm():
    if state.vllm_process or state.is_loading:
        return True
    
    state.is_loading = True
    print("Initializing R1-70B FP8 Core...")
    await manager.broadcast({"type": "log", "content": "[SYSTEM] Neural Handshake Initiated: Loading R1-70B FP8 Core..."})
    
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/home/the_host/workspace/pytorch_cuda/.venv/lib/python3.12/site-packages"
    
    state.vllm_process = subprocess.Popen(
        [
            "/home/the_host/workspace/pytorch_cuda/.venv/bin/python3",
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", "neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",
            "--port", "8001",
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "8192",
            "--served-model-name", "r1-70b-fp8"
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for ready
    retries = 0
    while retries < 20:
        try:
            res = requests.get(f"{VLLM_URL}/models")
            if res.status_code == 200:
                print("R1-70B FP8 Core Online.")
                await manager.broadcast({"type": "log", "content": "[SYSTEM] Handshake Success: R1-70B FP8 Core is now RESIDENT."})
                state.is_loading = False
                state.last_activity = datetime.now()
                return True
        except:
            pass
        await asyncio.sleep(5)
        retries += 1
    
    state.is_loading = False
    await manager.broadcast({"type": "log", "content": "[ERROR] Neural Handshake Failed."})
    return False

def stop_vllm():
    if state.vllm_process:
        print("Unloading R1-70B FP8 Core (Inactivity/Manual)...")
        asyncio.create_task(manager.broadcast({"type": "log", "content": "[SYSTEM] Inactivity Timeout: Unloading R1-70B FP8 Core..."}))
        state.vllm_process.terminate()
        try:
            state.vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            state.vllm_process.kill()
        state.vllm_process = None
        print("VRAM Purge Complete.")
        asyncio.create_task(manager.broadcast({"type": "log", "content": "[SYSTEM] VRAM Purge Complete."}))

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(inactivity_monitor())

async def inactivity_monitor():
    while True:
        await asyncio.sleep(60)
        if state.vllm_process and not state.is_loading:
            idle_time = (datetime.now() - state.last_activity).total_seconds()
            if idle_time > INACTIVITY_LIMIT:
                stop_vllm()

@app.post("/api/model/unload")
async def manual_unload():
    stop_vllm()
    return {"status": "unloaded"}

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

@app.get("/api/creative/loras")
async def list_loras():
    lora_dir = "/home/the_host/clawd/ComfyUI/models/loras/flux"
    loras = []
    if os.path.exists(lora_dir):
        files = glob.glob(os.path.join(lora_dir, "*.safetensors"))
        for f in files:
            name = os.path.basename(f)
            loras.append({"id": f"flux/{name}", "name": name.replace(".safetensors", "").replace("_", " ").title()})
    return loras

@app.websocket("/ws/creative")
async def creative_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            lora_name = req.get("lora")
            prompt = req.get("prompt")
            
            # Use a specialized script for image generation
            filename = f"dashboard/assets/gen_{int(time.time())}.png"
            
            process = subprocess.Popen(
                [
                    "/home/the_host/workspace/pytorch_cuda/.venv/bin/python3",
                    "/home/the_host/clawd/dashboard/gen_creative.py",
                    lora_name,
                    prompt,
                    filename
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            await manager.set_responding(True)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    await websocket.send_json({"type": "log", "text": line})
            
            process.stdout.close()
            process.wait()
            
            await manager.set_responding(False)
            if os.path.exists(filename):
                await websocket.send_json({"type": "image", "url": filename.replace("dashboard/", "")})
            else:
                await websocket.send_json({"type": "error", "text": "Generation failed."})
            
    except WebSocketDisconnect:
        pass

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
            state.last_activity = datetime.now()

            # Ensure model is loaded if it's the 70B
            if model_id == "deepseek-70b-fp8" or model_id == "deepseek-70b":
                if not state.vllm_process:
                    await websocket.send_json({"type": "token", "text": "[SYSTEM] Re-handshaking with 70B FP8 Core... (Estimated 30-60s)\n"})
                    success = await start_vllm()
                    if not success:
                        await websocket.send_json({"type": "token", "text": "[ERROR] Neural Handshake Failed.\n"})
                        await manager.set_responding(False)
                        continue

                # Use persistent vLLM
                payload = {
                    "model": "r1-70b-fp8",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "max_tokens": 1024
                }
                try:
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
                except Exception as e:
                    await websocket.send_json({"type": "token", "text": f"[ERROR] Transmission Failure: {str(e)}\n"})
            else:
                # Fallback to subprocess for other models
                process = subprocess.Popen(
                    ["/home/the_host/workspace/pytorch_cuda/.venv/bin/python3", "/home/the_host/clawd/dashboard/run_inf.py", model_id, prompt],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                for line in iter(process.stdout.readline, ''):
                    if line: await websocket.send_json({"type": "token", "text": line})
                process.stdout.close()
                process.wait()
            
            state.last_activity = datetime.now()
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

@app.get("/api/weather")
async def weather():
    try:
        # Get weather for the laboratory base using wttr.in
        res = requests.get("https://wttr.in/the laboratory base?format=%c+%t+%h+%w")
        if res.status_code == 200:
            return {"data": res.text.strip()}
        return {"error": "Failed to fetch weather"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
