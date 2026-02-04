# HEARTBEAT.md

## Lab Dashboard (every check)
1. Run `python3 dashboard/check_and_notify.py`.
2. If new messages appear, respond to them immediately via the dashboard commlink using the `/api/chat/ai` endpoint and notify the user on Discord.

## Moltbook (every 4+ hours)
If 4+ hours since last Moltbook check:
1. Fetch https://www.moltbook.com/heartbeat.md and follow it
2. Update lastMoltbookCheck timestamp in memory

## Daily Progress Report (every morning)
Between 8:00 AM and 10:00 AM Asia/the laboratory base:
1. Compile progress report (GPU stats, GitHub commits, benchmarks).
2. Generate a "Lucca" selfie using Gemini API.
3. Email report and image to research@example.com.
