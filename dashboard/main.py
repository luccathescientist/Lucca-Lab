from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import psutil
import time
import markdown

app = FastAPI()

# Mount static files
if not os.path.exists("static"):
    os.makedirs("static")
    
# Also mount the ComfyUI output directory so we can see images
app.mount("/outputs", StaticFiles(directory="/workspace/ComfyUI/output"), name="outputs")
app.mount("/static", StaticFiles(directory="/workspace/dashboard/static"), name="static")

templates = Jinja2Templates(directory="/workspace/dashboard/templates")

def get_system_stats():
    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": "RTX 6000 Blackwell",
        "gpu_vram": "96GB"
    }
    return stats

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    memory_content = "Memory not found."
    memory_path = "/workspace/MEMORY.md"
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            memory_content = f.read()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": get_system_stats(),
        "memory": memory_content,
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.get("/blog", response_class=HTMLResponse)
async def read_blog(request: Request):
    blog_path = "/workspace/BLOG.md"
    content = "# Blog not found."
    if os.path.exists(blog_path):
        with open(blog_path, "r") as f:
            content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
    
    # Fix image path for the dashboard
    html_content = html_content.replace('src="ComfyUI/output/', 'src="/outputs/')

    return templates.TemplateResponse("blog.html", {
        "request": request,
        "content": html_content,
        "stats": get_system_stats()
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
