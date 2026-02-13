const fs = require('fs');
const path = require('path');

const filePath = '/home/rocketegg/clawd/dashboard/improvement_queue.md';
let content = fs.readFileSync(filePath, 'utf8');

// Brainstorm 4 new tasks
const newTasks = [
    '- [ ] **Neural Knowledge Graph Explorer**: A graph-based visualization of the links between different research notes in the Lab.',
    '- [ ] **Rig Thermal Topography**: A 3D heat map of the Blackwell rig components based on real-time sensor data.',
    '- [ ] **Sub-Agent Swarm Status**: A tactical view showing the current task, memory usage, and logic trajectory of all active sub-agents.',
    '- [ ] **Laboratory Global Sync**: A map showing the geographic location of all paired nodes and their connection health.'
];

content += '\n' + newTasks.join('\n') + '\n';

// Pick the first pending task
const lines = content.split('\n');
let firstPendingIndex = -1;
for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('- [ ]')) {
        firstPendingIndex = i;
        break;
    }
}

if (firstPendingIndex !== -1) {
    const taskText = lines[firstPendingIndex].replace('- [ ]', '').trim();
    lines[firstPendingIndex] = lines[firstPendingIndex].replace('- [ ]', '- [x]');
    fs.writeFileSync(filePath, lines.join('\n'));
    console.log(`TASK_NAME:${taskText}`);
} else {
    console.log('NO_PENDING_TASK');
}
