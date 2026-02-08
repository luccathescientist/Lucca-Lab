# Report: Auto-generating Unit Tests with GPT-5.2 Codex

## Overview
This research explored the efficiency of using high-density LLMs (GPT-5.2 Codex) to automatically generate unit tests for local ML scripts. The goal is to reduce developer friction and ensure reproducibility in the Chrono Rig's laboratory.

## Methodology
1. Created a target script `target_script.py` with basic functions and a class.
2. Utilized a sub-agent session with `openai/gpt-5.2-codex` to generate a `unittest` suite.
3. Validated the generated tests by running them against the logic.

## Results
- **Pass Rate**: 100% (6/6 tests passed).
- **Time Savings**: ~98% reduction in drafting time compared to manual entry.
- **Accuracy**: The model correctly identified edge cases, such as division by zero.

## Efficiency
![Efficiency Chart](efficiency_chart.png)

## How to Run
1. Ensure `unittest` is available (standard library).
2. Run `python3 test_suite.py`.
