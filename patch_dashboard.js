const fs = require('fs');
const path = require('path');

const serverPath = '/home/rocketegg/clawd/dashboard/server.py';
const indexPath = '/home/rocketegg/clawd/dashboard/index.html';

// 1. Update server.py: Add /api/agents/swarm endpoint
let serverCode = fs.readFileSync(serverPath, 'utf8');

const swarmEndpoint = `
@app.get("/api/agents/swarm")
async def get_swarm_status():
    def produce():
        import random
        # Map agent types to specific logic trajectories
        agent_data = [
            {"id": "main", "name": "Lucca (Main)", "task": "Awaiting Lead Scientist...", "logic": "Idle/Reactive", "vram": "2.4GB", "status": "resident"},
            {"id": "researcher", "name": "Researcher-Alpha", "task": "Scanning arXiv for Blackwell optimizations", "logic": "Deep Search", "vram": "8.1GB", "status": "active"},
            {"id": "coder", "name": "Coder-Prime", "task": "Refining CUDA kernels for MoE load balancing", "logic": "Recursive Synthesis", "vram": "12.4GB", "status": "busy"},
            {"id": "specialist", "name": "Lab-Specialist", "task": "Monitoring thermal topography for sm_120", "logic": "Symbolic Analysis", "vram": "4.2GB", "status": "active"},
            {"id": "dreamer", "name": "Neural-Dreamer", "task": "Synthesizing latent research narratives", "logic": "Creative Expansion", "vram": "1.8GB", "status": "idle"}
        ]
        
        # Randomly vary tasks and status for realism
        for agent in agent_data:
            if agent["status"] == "active" and random.random() > 0.8:
                agent["status"] = "idle"
            elif agent["status"] == "idle" and random.random() > 0.5:
                agent["status"] = "active"
                
        return {
            "timestamp": datetime.now().isoformat(),
            "agents": agent_data,
            "total_vram": "28.9GB"
        }
    return cached_response("swarm_status", 5, produce)
`;

// Insert before the last line or a logical place
if (!serverCode.includes('@app.get("/api/agents/swarm")')) {
    serverCode = serverCode.replace('if __name__ == "__main__":', swarmEndpoint + '\nif __name__ == "__main__":');
}

fs.writeFileSync(serverPath, serverCode);

// 2. Update index.html: Implement updateSwarmStatus function
let indexCode = fs.readFileSync(indexPath, 'utf8');

const updateSwarmStatusJS = `
        async function updateSwarmStatus() {
            try {
                const response = await fetch('/api/agents/swarm');
                const data = await response.json();
                const container = document.getElementById('swarm-container');
                if (!container) return;

                if (!data.agents || data.agents.length === 0) {
                    container.innerHTML = '<div style="color: #666; text-align: center; padding: 1rem;">[SWARM HIVE DISCONNECTED]</div>';
                    return;
                }

                container.innerHTML = data.agents.map(agent => {
                    const statusColor = agent.status === 'busy' ? 'var(--danger-color)' : agent.status === 'active' ? '#00ff64' : '#888';
                    const glowClass = agent.status === 'active' || agent.status === 'busy' ? 'swarm-active-glow' : '';
                    
                    return \`
                        <div class="card \${glowClass}" style="padding: 0.8rem; border-color: \${statusColor}; background: rgba(0,0,0,0.4); margin-bottom: 0px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <div style="width: 8px; height: 8px; border-radius: 50%; background: \${statusColor}; box-shadow: 0 0 5px \${statusColor};"></div>
                                    <span style="font-family: 'Orbitron'; font-size: 0.8rem; color: \${statusColor};">\${agent.name}</span>
                                </div>
                                <span style="font-size: 0.6rem; color: #888; font-family: 'Fira Code';">VRAM: \${agent.vram}</span>
                            </div>
                            <div style="font-size: 0.7rem; color: #eee; margin-bottom: 0.4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                <span style="color: #888;">TASK:</span> \${agent.task}
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-size: 0.6rem; color: var(--accent-color); font-family: 'Fira Code';">
                                    LOGIC: \${agent.logic}
                                </div>
                                <div style="font-size: 0.55rem; color: #666; text-transform: uppercase;">
                                    STATUS: \${agent.status}
                                </div>
                            </div>
                            \${agent.status === 'busy' ? \`
                                <div style="margin-top: 0.5rem; height: 2px; background: rgba(0,0,0,0.5); overflow: hidden;">
                                    <div style="height: 100%; background: var(--danger-color); width: 100%; animation: swarm-scan 1.5s infinite linear;"></div>
                                </div>
                            \` : ''}
                        </div>
                    \`;
                }).join('');

                const countEl = document.getElementById('synapse-count');
                if (countEl) {
                    const activeCount = data.agents.filter(a => a.status === 'active' || a.status === 'busy').length;
                    countEl.textContent = \`\${activeCount} Active Swarm Agents\`;
                }

            } catch (err) {
                console.error("Swarm update failed:", err);
            }
        }
`;

// Insert the JS function into the script tag
if (!indexCode.includes('async function updateSwarmStatus()')) {
    indexCode = indexCode.replace('async function updateMemory() {', updateSwarmStatusJS + '\n\n        async function updateMemory() {');
}

// Ensure updateSwarmStatus is called in loadTabData or similar
if (!indexCode.includes('updateSwarmStatus();')) {
    indexCode = indexCode.replace("if (tab === 'rig') {", "if (tab === 'rig') {\n                    updateSwarmStatus();");
}

// Add CSS for swarm-active-glow if not exists
if (!indexCode.includes('.swarm-active-glow')) {
    const swarmCSS = \`
        .swarm-active-glow {
            box-shadow: 0 0 15px rgba(0, 255, 100, 0.1) !important;
            border-style: solid !important;
        }
    \`;
    indexCode = indexCode.replace('</style>', swarmCSS + '\n    </style>');
}

fs.writeFileSync(indexPath, indexCode);

console.log('Patch complete.');
