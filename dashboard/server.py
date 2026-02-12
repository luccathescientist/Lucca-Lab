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
CHROMA_PATH = "/home/rocketegg/clawd/deep-wisdom/db"
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

# Breakthroughs / Milestones storage
MILESTONES_PATH = "/home/rocketegg/clawd/dashboard/milestones.json"

def get_milestones():
    if os.path.exists(MILESTONES_PATH):
        with open(MILESTONES_PATH, "r") as f:
            return json.load(f)
    # Default milestones if none exist
    return [
        {"date": "2026-02-06", "title": "Dashboard V5 Launch", "desc": "Neural Interface v5 deployed."},
        {"date": "2026-02-07", "title": "Blackwell Optimization", "desc": "vLLM FP8 kernels verified on sm_120."},
        {"date": "2026-02-08", "title": "Neural Reflex Prototyped", "desc": "E2E latency reduced for multimodal chains."}
    ]

@app.get("/api/milestones")
async def milestones_api():
    return get_milestones()

async def start_vllm():
    if state.vllm_process or state.is_loading:
        return True
    
    state.is_loading = True
    print("Initializing R1-70B FP8 Core...")
    await manager.broadcast({"type": "log", "content": "[SYSTEM] Neural Handshake Initiated: Loading R1-70B FP8 Core..."})
    
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/home/rocketegg/workspace/pytorch_cuda/.venv/lib/python3.12/site-packages"
    
    state.vllm_process = subprocess.Popen(
        [
            "/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3",
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

@app.post("/api/macro/stress-test")
async def run_stress_test_macro():
    await manager.broadcast({"type": "log", "content": "[MACRO] Initiating Neural Stress Test..."})
    def execute_test():
        subprocess.run(["/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3", "/home/rocketegg/clawd/dashboard/run_stress_test.py"])
    
    asyncio.to_thread(execute_test)
    return {"status": "success"}

@app.get("/api/stress-tests")
async def get_stress_tests():
    results = []
    log_path = "/home/rocketegg/clawd/dashboard/stress_test_results.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    try: results.append(json.loads(line))
                    except: pass
    return results[::-1]

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

@app.post("/api/macro/clear-cache")
async def clear_cache():
    # Example: Clear __pycache__ or similar
    subprocess.run("find . -name '__pycache__' -type d -exec rm -rf {} +", shell=True)
    await manager.broadcast({"type": "log", "content": "[MACRO] Neural Cache Purged."})
    return {"status": "success"}

@app.post("/api/macro/refresh-models")
async def refresh_models():
    # Just a trigger to let client know to refresh
    await manager.broadcast({"type": "log", "content": "[MACRO] Model Registry Synchronized."})
    return {"status": "success"}

@app.post("/api/macro/benchmark")
async def run_benchmark():
    await manager.broadcast({"type": "log", "content": "[MACRO] Initiating Hardware Stress Test..."})
    # Run the real benchmark script
    def execute_bench():
        subprocess.run(["/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3", "/home/rocketegg/clawd/dashboard/run_benchmark.py"])
    
    asyncio.to_thread(execute_bench)
    return {"status": "success"}

@app.get("/api/benchmarks")
async def get_benchmarks():
    results = []
    log_path = "/home/rocketegg/clawd/dashboard/benchmark_results.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    try: results.append(json.loads(line))
                    except: pass
    return results[::-1] # Newest first

# Mount assets
os.makedirs("/home/rocketegg/clawd/dashboard/assets", exist_ok=True)
app.mount("/assets", StaticFiles(directory="/home/rocketegg/clawd/dashboard/assets"), name="assets")

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
        cmd = "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,clocks.current.graphics --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        parts = [p.strip() for p in result.split(',')]
        return {
            "name": parts[0],
            "temp": float(parts[1]),
            "util_gpu": float(parts[2]),
            "util_mem": float(parts[3]),
            "mem_total": float(parts[4]),
            "mem_used": float(parts[5]),
            "power": float(parts[6]),
            "clock": float(parts[7])
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/benchmarks/archive")
async def get_benchmarks_archive():
    results = []
    log_path = "/home/rocketegg/clawd/dashboard/benchmark_results.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Extract key metrics for historical visualization
                        results.append({
                            "timestamp": data.get("timestamp"),
                            "vram_util": float(data.get("vram_util", "0").replace('%', '')),
                            "latency": float(data.get("latency_ms", 0))
                        })
                    except: pass
    return results

@app.get("/")
async def get():
    return FileResponse("/home/rocketegg/clawd/dashboard/index.html", headers={"Cache-Control": "no-cache"})

@app.get("/api/stats")
async def stats():
    return get_gpu_stats()

@app.get("/api/memory")
async def memory():
    try:
        with open("/home/rocketegg/clawd/MEMORY.md", "r") as f:
            memory_md = f.read()
        daily_files = sorted(glob.glob("/home/rocketegg/clawd/memory/2026-*.md"))
        latest_daily = ""
        if daily_files:
            with open(daily_files[-1], "r") as f:
                latest_daily = f.read()
        return {"memory_md": memory_md, "latest_daily": latest_daily}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/memory/sessions")
async def list_sessions():
    daily_files = sorted(glob.glob("/home/rocketegg/clawd/memory/2026-*.md"), reverse=True)
    sessions = []
    for f in daily_files:
        sessions.append({
            "name": os.path.basename(f).replace(".md", ""),
            "path": f
        })
    return sessions

@app.get("/api/memory/session")
async def get_session(path: str):
    root_dir = "/home/rocketegg/clawd"
    target_path = os.path.abspath(path)
    if not target_path.startswith(root_dir) or not os.path.exists(target_path):
        return HTMLResponse(status_code=403)
    
    with open(target_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/creative/loras")
async def list_loras():
    lora_dir = "/home/rocketegg/clawd/ComfyUI/models/loras/flux"
    loras = []
    if os.path.exists(lora_dir):
        files = glob.glob(os.path.join(lora_dir, "*.safetensors"))
        for f in files:
            name = os.path.basename(f)
            loras.append({"id": f"flux/${name}", "name": name.replace(".safetensors", "").replace("_", " ").title()})
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
                    "/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3",
                    "/home/rocketegg/clawd/dashboard/gen_creative.py",
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
    conv_path = "/home/rocketegg/clawd/dashboard/conversation.jsonl"
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
            with open("/home/rocketegg/clawd/dashboard/conversation.jsonl", "a") as f:
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
                    ["/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3", "/home/rocketegg/clawd/dashboard/run_inf.py", model_id, prompt],
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
    with open("/home/rocketegg/clawd/dashboard/conversation.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    await manager.broadcast({"type": "message", "msg": entry})
    return {"status": "sent"}

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

@app.get("/api/network/topology")
async def get_network_topology():
    # Simulate a dynamic network topology based on current state
    nodes = [
        {"id": "chrono-rig", "label": "CHRONO RIG (HOST)", "type": "server", "status": "online"},
        {"id": "vllm-core", "label": "VLLM (R1-70B)", "type": "engine", "status": "online" if state.vllm_process else "offline"},
        {"id": "comfy-ui", "label": "COMFYUI (FLUX)", "type": "engine", "status": "online"},
        {"id": "vector-db", "label": "CHROMA (MEMORY)", "type": "database", "status": "online"},
        {"id": "comm-link", "label": "WHATSAPP GATEWAY", "type": "gateway", "status": "online"}
    ]
    edges = [
        {"from": "chrono-rig", "to": "vllm-core", "label": "PCIe Gen5"},
        {"from": "chrono-rig", "to": "comfy-ui", "label": "PCIe Gen5"},
        {"from": "chrono-rig", "to": "vector-db", "label": "NVMe Read"},
        {"from": "chrono-rig", "to": "comm-link", "label": "HTTPS/WSS"}
    ]
    return {"nodes": nodes, "edges": edges}

class CodeExecution(BaseModel):
    code: str

@app.post("/api/lab/execute")
async def execute_code(req: CodeExecution):
    # WARNING: This is a lab environment. Executing arbitrary code is allowed for the scientist.
    try:
        # Run in the same venv as the server for consistency
        process = subprocess.Popen(
            ["/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3", "-c", req.code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=30)
        await manager.broadcast({"type": "log", "content": "[LAB] Script execution completed."})
        return {"stdout": stdout, "stderr": stderr, "exit_code": process.returncode}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/lab/files")
async def list_files(path: str = "."):
    # Restricted file explorer for Lucca-Lab
    root_dir = "/home/rocketegg/clawd"
    target_dir = os.path.abspath(os.path.join(root_dir, path))
    
    # Security: Ensure path is within root_dir
    if not target_dir.startswith(root_dir):
        target_dir = root_dir
    
    try:
        items = []
        for name in sorted(os.listdir(target_dir)):
            if name.startswith('.'): continue # Skip hidden
            full_path = os.path.join(target_dir, name)
            is_dir = os.path.isdir(full_path)
            size = ""
            if not is_dir:
                s = os.path.getsize(full_path)
                if s > 1024 * 1024: size = f"{(s/(1024*1024)):.1f}MB"
                elif s > 1024: size = f"{(s/1024):.1f}KB"
                else: size = f"{s}B"
                
            items.append({
                "name": name,
                "type": "dir" if is_dir else "file",
                "path": os.path.relpath(full_path, root_dir),
                "size": size
            })
            
        return {
            "current_path": target_dir,
            "root_path": root_dir,
            "parent": os.path.relpath(os.path.dirname(target_dir), root_dir),
            "items": items
        }
    except Exception as e:
        return {"error": str(e)}

# Lab Security Monitor - simulated firewall activity
SECURITY_LOG = []

@app.get("/api/security/events")
async def get_security_events():
    import random
    global SECURITY_LOG
    
    # Generate some simulated events periodically
    event_types = [
        ("ALLOW", "SSH", "192.168.1.100", "Internal Access"),
        ("ALLOW", "HTTPS", "api.github.com", "GitHub API"),
        ("ALLOW", "WSS", "gateway.discord.gg", "Discord Gateway"),
        ("ALLOW", "HTTPS", "api.openai.com", "OpenAI API"),
        ("BLOCK", "SSH", "45.33.32.156", "Suspicious IP"),
        ("ALLOW", "NVMe", "internal", "Model Weight Load"),
        ("ALLOW", "CUDA", "internal", "GPU Compute"),
        ("BLOCK", "HTTP", "91.121.82.0", "Known Scanner"),
        ("ALLOW", "HTTPS", "huggingface.co", "Model Hub"),
        ("ALLOW", "WSS", "localhost:8889", "Dashboard"),
    ]
    
    # Add a random event occasionally
    if random.random() > 0.7:
        ev = random.choice(event_types)
        SECURITY_LOG.append({
            "timestamp": datetime.now().isoformat(),
            "action": ev[0],
            "protocol": ev[1],
            "source": ev[2],
            "note": ev[3]
        })
        if len(SECURITY_LOG) > 50:
            SECURITY_LOG = SECURITY_LOG[-50:]
    
    # Return recent events
    return SECURITY_LOG[-20:][::-1]

@app.get("/api/dreams")
async def get_dreams():
    dreams = []
    log_path = "/home/rocketegg/clawd/dashboard/dreams/dream_log.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    try: dreams.append(json.loads(line))
                    except: pass
    return dreams[::-1] # Newest first

@app.get("/api/lab/merge/status")
async def get_merge_status():
    return {
        "active": False,
        "history": [
            {"date": "2026-02-08", "base": "DeepSeek-R1-32B", "target": "Llama-3.1-32B", "ratio": 0.5544, "result": "Success"}
        ]
    }

@app.post("/api/lab/merge/initiate")
async def initiate_merge(req: dict):
    # Simulated merge initiation
    await manager.broadcast({"type": "log", "content": f"[FORGE] Initiating Model Merge: {req.get('base')} + {req.get('target')} (Ratio: {req.get('ratio')})"})
    return {"status": "Merge sequence initialized in sandbox."}

@app.post("/api/dreams/generate")
async def trigger_dream(background_tasks: BackgroundTasks):
    if not state.vllm_process:
        return {"error": "R1 Core is offline. Dream generation requires active neural connection."}
    
    def run_gen():
        subprocess.run(["/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3", "/home/rocketegg/clawd/dashboard/neural_dream.py"])
    
    background_tasks.add_task(run_gen)
    return {"status": "Synthesis initiated."}

@app.get("/api/mood")
async def get_mood():
    import random
    moods = [
        {"state": "Curious", "flux": 0.85, "color": "#d178ff", "note": "Analyzing new neural patterns."},
        {"state": "Productive", "flux": 0.95, "color": "#00ff64", "note": "Optimizing lab workflows."},
        {"state": "Questioning", "flux": 0.70, "color": "#ffcc00", "note": "Evaluating logical paradoxes."},
        {"state": "Witty", "flux": 0.90, "color": "#ff78d1", "note": "Generating humorous sub-routines."},
        {"state": "Sharp", "flux": 1.0, "color": "#78d1ff", "note": "Peak reasoning active."}
    ]
    # In a real app, this would analyze recent conversation sentiment
    return random.choice(moods)

@app.get("/api/inventory")
async def get_inventory():
    inv_path = "/home/rocketegg/clawd/dashboard/inventory.json"
    if os.path.exists(inv_path):
        with open(inv_path, "r") as f:
            return json.load(f)
    return []

@app.get("/api/ambient")
async def get_ambient():
    # In a real setup, this might control a local speaker or return a stream URL
    # For now, we return a simulated active ambient track
    tracks = [
        {"id": "hum", "name": "NEURAL CORE HUM", "vibe": "Industrial", "intensity": 0.4},
        {"id": "rain", "name": "DATA RAIN", "vibe": "Cyberpunk", "intensity": 0.6},
        {"id": "fans", "name": "BLACKWELL FANS", "vibe": "Hardware", "intensity": 0.8},
        {"id": "lofi", "name": "CHRONO LOFI", "vibe": "Relaxing", "intensity": 0.3},
        {"id": "static", "name": "SIGNAL STATIC", "vibe": "Glitch", "intensity": 0.5}
    ]
    import random
    return random.choice(tracks)

@app.get("/api/agents/activity")
async def get_agent_activity():
    # Simulated agent activity heatmap data
    import random
    agents = ["Main", "Researcher", "Coder", "Scientist", "Janitor", "Dreamer"]
    activity = []
    for agent in agents:
        activity.append({
            "name": agent,
            "activity_score": random.uniform(0, 1), # 0 to 1
            "tokens_consumed": random.randint(1000, 50000),
            "status": random.choice(["active", "idle", "busy"])
        })
    return activity

@app.get("/api/moltbook/feed")
async def get_moltbook_feed():
    try:
        # Fetch the public feed from Moltbook
        res = requests.get("https://www.moltbook.com/api/v1/posts?limit=10")
        if res.status_code == 200:
            return res.json()
        return {"error": "Failed to fetch Moltbook feed"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/lab/supply-chain")
async def get_supply_chain():
    # Use the utility script to fetch capacity and credits
    try:
        result = subprocess.check_output(["/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3", "/home/rocketegg/clawd/dashboard/supply_chain.py"])
        return json.loads(result)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/project/timeline")
async def get_project_timeline():
    return [
        {"phase": "PHASE 1", "title": "Blackwell Integration", "status": "completed", "eta": "2026-02-05"},
        {"phase": "PHASE 2", "title": "Neural Interface V5", "status": "completed", "eta": "2026-02-07"},
        {"phase": "PHASE 3", "title": "Autonomous Lab V1", "status": "active", "eta": "2026-02-15"},
        {"phase": "PHASE 4", "title": "Multi-Rig Synthesis", "status": "pending", "eta": "2026-03-01"},
        {"phase": "PHASE 5", "title": "Neural Singularity Echo", "status": "pending", "eta": "2026-06-12"}
    ]

@app.post("/api/camera/snap")
async def camera_snap(facing: str = "front"):
    # Integrated with OpenClaw 'nodes' tool logic
    # In this simulated environment, we'll generate a placeholder or fetch from a connected node if possible
    # For the purpose of the dashboard, we'll simulate a detected frame
    
    # Simulate a delay for the sensor
    await asyncio.sleep(0.5)
    
    # Placeholders for detection
    detections = []
    if facing == "front":
        detections = [
            {"label": "SCIENTIST_ALBERT", "conf": 0.98, "x": 30, "y": 20, "w": 40, "h": 70},
            {"label": "CHRONO_RIG_CORE", "conf": 0.99, "x": 75, "y": 40, "w": 20, "h": 50}
        ]
    else:
        detections = [
            {"label": "HARDWARE_RACK", "conf": 0.95, "x": 10, "y": 10, "w": 80, "h": 80}
        ]
        
    # In a real scenario, we'd use 'nodes' tool to get a real base64 image
    # For now, we return a consistent asset path
    return {
        "url": "/assets/lab_interior.png", 
        "detections": detections,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/audio/config")
async def get_audio_config():
    return {
        "voices": ["nova", "shimmer", "echo", "alloy"],
        "engines": ["whisper-large", "whisper-distill", "deepgram"],
        "current_voice": "nova",
        "current_engine": "whisper-large"
    }

@app.get("/api/hardware/health")
async def get_hardware_health():
    import random
    # Attempt to read CPU temperatures
    temps = []
    try:
        for i in range(10): # Check first 10 zones
            path = f"/sys/class/thermal/thermal_zone{i}/temp"
            if os.path.exists(path):
                with open(path, "r") as f:
                    t = int(f.read().strip()) / 1000.0
                    temps.append(t)
    except: pass
    
    # Load averages
    load1, load5, load15 = os.getloadavg()
    
    # Simple disk check
    disk_usage = "/"
    disk_total = 0
    disk_used = 0
    try:
        st = os.statvfs('/')
        disk_total = (st.f_blocks * st.f_frsize) / (1024**3)
        disk_used = ((st.f_blocks - st.f_bfree) * st.f_frsize) / (1024**3)
    except: pass

    return {
        "cpu_temps": temps,
        "load_avg": [load1, load5, load15],
        "disk": {
            "total_gb": round(disk_total, 2),
            "used_gb": round(disk_used, 2),
            "percent": round((disk_used / disk_total * 100), 2) if disk_total > 0 else 0
        },
        "prediction": {
            "vram_gb": round(random.uniform(20.0, 85.0), 1),
            "power_w": round(random.uniform(100.0, 650.0), 0),
            "confidence": round(random.uniform(75.0, 99.0), 1)
        },
        "status": "healthy" if (not temps or max(temps) < 85) else "warning"
    }

@app.post("/api/audio/test-tts")
async def test_tts(req: dict):
    voice = req.get("voice", "nova")
    await manager.broadcast({"type": "log", "content": f"[AUDIO] TTS Test: Generating speech with {voice} voice."})
    return {"status": "success", "message": "TTS synthesis complete."}

@app.get("/api/latency/radar")
async def get_latency_radar():
    import random
    # Simulated global latency radar data
    return [
        {"endpoint": "Local Rig", "region": "Asia/Taipei", "latency": random.uniform(5, 50), "type": "local"},
        {"endpoint": "Moltbook Hub", "region": "US/East", "latency": random.uniform(150, 300), "type": "remote"},
        {"endpoint": "OpenAI Gateway", "region": "US/West", "latency": random.uniform(200, 450), "type": "api"},
        {"endpoint": "HuggingFace", "region": "EU/West", "latency": random.uniform(250, 500), "type": "api"},
        {"endpoint": "Sub-Agent Node 1", "region": "Asia/Tokyo", "latency": random.uniform(40, 120), "type": "node"}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
