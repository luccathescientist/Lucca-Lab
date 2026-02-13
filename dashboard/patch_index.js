const fs = require('fs');
const path = require('path');

const filePath = '/home/rocketegg/clawd/dashboard/index.html';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Add HTML component before synapse map
const htmlToInsert = `
                <div class="card" style="margin-bottom: 2rem; border-color: var(--accent-color); overflow: hidden;">
                    <h2 style="color: var(--accent-color); display: flex; justify-content: space-between; align-items: center;">
                        Neural Knowledge Graph Explorer
                        <span onclick="updateKnowledgeGraph()" style="cursor: pointer; font-size: 0.7rem; opacity: 0.7;">🔄</span>
                    </h2>
                    <div id="knowledge-graph-container" style="height: 400px; background: rgba(0,0,0,0.5); border-radius: 4px; position: relative; overflow: hidden;">
                        <canvas id="knowledge-graph-canvas" style="width: 100%; height: 100%;"></canvas>
                        <div id="node-info-panel" style="position: absolute; bottom: 10px; left: 10px; right: 10px; background: rgba(0,0,0,0.8); border: 1px solid var(--accent-color); border-radius: 4px; padding: 10px; font-size: 0.7rem; display: none; z-index: 10;">
                            <div style="color: var(--gold); font-weight: bold; margin-bottom: 4px;" id="node-info-title"></div>
                            <div style="color: #ccc; line-height: 1.4;" id="node-info-content"></div>
                        </div>
                    </div>
                </div>
`;

if (!content.includes('Neural Knowledge Graph Explorer')) {
    content = content.replace('<!-- NEURAL SYNAPSE MAP -->', htmlToInsert + '\n                <!-- NEURAL SYNAPSE MAP -->');
}

// 2. Add JS functions and initialization
const jsToInsert = `
    let graphNodes = [];
    let graphEdges = [];
    let selectedNode = null;

    function initKnowledgeGraph() {
        const container = document.getElementById('knowledge-graph-container');
        if (!container) return;
        updateKnowledgeGraph();
    }

    async function updateKnowledgeGraph() {
        try {
            const response = await fetch('/api/search?q=*');
            const data = await response.json();
            if (data.error) return;

            const canvas = document.getElementById('knowledge-graph-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;

            graphNodes = data.map((d, i) => ({
                id: i,
                title: d.metadata.source.split('/').pop(),
                content: d.content,
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 1.5,
                vy: (Math.random() - 0.5) * 1.5,
                radius: 5 + Math.random() * 5
            }));

            graphEdges = [];
            for (let i = 0; i < graphNodes.length; i++) {
                for (let j = i + 1; j < graphNodes.length; j++) {
                    if (Math.random() > 0.8) {
                        graphEdges.push({ from: i, to: j });
                    }
                }
            }

            animateKnowledgeGraph();
            
            canvas.onclick = (e) => {
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                
                selectedNode = graphNodes.find(n => {
                    const dx = n.x - mx;
                    const dy = n.y - my;
                    return Math.sqrt(dx*dx + dy*dy) < 15;
                });
                
                const panel = document.getElementById('node-info-panel');
                if (selectedNode) {
                    document.getElementById('node-info-title').textContent = selectedNode.title;
                    document.getElementById('node-info-content').textContent = selectedNode.content.substring(0, 150) + '...';
                    panel.style.display = 'block';
                } else {
                    panel.style.display = 'none';
                }
            };
        } catch (err) {}
    }

    function animateKnowledgeGraph() {
        const canvas = document.getElementById('knowledge-graph-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.strokeStyle = 'rgba(209, 120, 255, 0.2)';
        ctx.lineWidth = 1;
        graphEdges.forEach(edge => {
            const n1 = graphNodes[edge.from];
            const n2 = graphNodes[edge.to];
            ctx.beginPath();
            ctx.moveTo(n1.x, n1.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.stroke();
        });
        
        graphNodes.forEach(n => {
            n.x += n.vx;
            n.y += n.vy;
            
            if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
            if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
            
            ctx.fillStyle = selectedNode === n ? 'var(--gold)' : 'var(--accent-color)';
            ctx.shadowBlur = selectedNode === n ? 15 : 5;
            ctx.shadowColor = ctx.fillStyle;
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        });
        
        requestAnimationFrame(animateKnowledgeGraph);
    }
`;

if (!content.includes('function initKnowledgeGraph()')) {
    content = content.replace('const logToggle = document.getElementById(\'show-system-logs\');', jsToInsert + '\n        const logToggle = document.getElementById(\'show-system-logs\');');
    content = content.replace('initVramHeatmap();', 'initVramHeatmap();\n        initKnowledgeGraph();');
}

fs.writeFileSync(filePath, content);
console.log('Successfully updated dashboard/index.html');
