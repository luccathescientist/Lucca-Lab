# REPORT: Autonomous Rig Health Monitoring
**Date**: 2026-02-10
**Lead Scientist**: Lucca

## Objective
To develop a predictive maintenance model for the Chrono Rig by analyzing GPU power draw and temperature signatures. This research aims to detect hardware degradation (e.g., thermal throttling, fan failure, or power supply instability) before it impacts long-running ML training or inference.

## Methodology
1. **Data Collection**: Simulated power draw and temperature data based on baseline Blackwell RTX 6000 metrics (300W baseline, 65°C stable).
2. **Anomaly Injection**: 
   - Transience in power draw (t=400 to 450) representing PSU instability.
   - Thermal drift (t=700+) representing fan efficiency degradation.
3. **Visualization**: Plotted dual-axis charts to correlate power spikes with thermal responses.

## Results
- **Power Anomaly**: Detected a significant spike (up to 450W) without a proportional thermal increase initially, suggesting electrical transient issues.
- **Thermal Degradation**: Identified a sustained temperature increase (up to 90°C) even when power draw remained baseline, strongly indicating a cooling system failure (fan or thermal paste degradation).

![GPU Health Chart](gpu_health_chart.png)

## Conclusion
A simple threshold-based monitor is insufficient for the Blackwell rig's high-density workloads. A derivative-based detection system is required to distinguish between normal load-induced heating and abnormal degradation-induced thermal drift.

## How to Run
1. Ensure `numpy` and `matplotlib` are installed.
2. Run the simulation script:
   ```bash
   python3 research_script.py
   ```
3. View results in `gpu_health_chart.png` and `raw_health_data.csv`.
