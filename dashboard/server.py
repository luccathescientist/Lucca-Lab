import http.server
import socketserver
import json
import subprocess
import threading
import time
import os

PORT = 8888

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

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            stats = get_gpu_stats()
            self.wfile.write(json.dumps(stats).encode())
        else:
            # Serve index.html for root, or static files
            if self.path == '/' or self.path == '':
                self.path = '/index.html'
            return super().do_GET()

def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Lucca Lab Dashboard running at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
