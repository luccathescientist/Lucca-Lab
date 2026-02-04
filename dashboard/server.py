import http.server
import socketserver
import json
import subprocess
import threading
import time
import os
import glob
from datetime import datetime

PORT = 8889

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

def get_memory_stream():
    try:
        # Get MEMORY.md content
        with open("/home/the_host/clawd/MEMORY.md", "r") as f:
            memory_md = f.read()
        
        # Get latest daily memory
        daily_files = sorted(glob.glob("/home/the_host/clawd/memory/2026-*.md"))
        latest_daily = ""
        if daily_files:
            with open(daily_files[-1], "r") as f:
                latest_daily = f.read()
        
        return {
            "memory_md": memory_md,
            "latest_daily": latest_daily
        }
    except Exception as e:
        return {"error": str(e)}

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            message = json.loads(post_data)
            
            # Save message to conversation
            conv_path = "/home/the_host/clawd/dashboard/conversation.jsonl"
            with open(conv_path, "a") as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "sender": "User",
                    "text": message.get("text", "")
                }
                f.write(json.dumps(entry) + "\n")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "received"}).encode())

    def do_GET(self):
        if self.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            stats = get_gpu_stats()
            self.wfile.write(json.dumps(stats).encode())
        elif self.path == '/api/memory':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            mem = get_memory_stream()
            self.wfile.write(json.dumps(mem).encode())
        elif self.path == '/api/messages':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            messages = []
            conv_path = "/home/the_host/clawd/dashboard/conversation.jsonl"
            if os.path.exists(conv_path):
                with open(conv_path, "r") as f:
                    for line in f:
                        if line.strip():
                            messages.append(json.loads(line))
            self.wfile.write(json.dumps(messages).encode())
        else:
            if self.path == '/' or self.path == '':
                self.path = '/index.html'
            return super().do_GET()

def run_server():
    # Change to the dashboard directory to serve files correctly
    os.chdir("/home/the_host/clawd/dashboard")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Lucca Lab Dashboard running at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
