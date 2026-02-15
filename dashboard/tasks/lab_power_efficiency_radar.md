# Lab Power Efficiency Radar

The goal of this task is to implement the "Lab Power Efficiency Radar" visualizer on the Chrono Rig dashboard. This component will provide the Lead Scientist with a direct comparison of how efficiently different local models are performing relative to their power consumption.

### Functional Requirements:
1. **API Endpoint**: Implement `/api/lab/power-efficiency` in `dashboard/server.py`.
   - The endpoint should return a list of model performance-per-watt metrics.
   - For now, simulate these metrics using typical values for the Blackwell rig (e.g., tokens-per-second divided by current power draw).
2. **UI Component**: Add the "Lab Power Efficiency Radar" card to `dashboard/index.html`.
   - Place it in the "Rig Stats" tab.
   - Use a Radar chart (via Chart.js) to visualize the efficiency across different axes: Speed, Logic, Vision, and Power Efficiency.
3. **Real-time Updates**: The component should refresh every 30 seconds to reflect changes in rig utilization and model state.

### Implementation Details:
- **Server-side (Python)**:
  - Cache the response to avoid over-polling hardware sensors.
  - Models to include: `DeepSeek-R1-70B`, `Qwen-1.5B`, `Flux.1-Schnell`.
- **Client-side (JS/HTML)**:
  - Initialize a new Chart.js Radar instance.
  - Add a manual refresh trigger (🔄).

---
*Assigned to: Lucca*
*Status: PENDING*
