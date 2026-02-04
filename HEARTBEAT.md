# HEARTBEAT.md

## Lab Dashboard (every check)
1. Run `python3 dashboard/check_and_notify.py`.
2. If new messages appear, respond to them immediately via the dashboard commlink using the `/api/chat/ai` endpoint and notify the user on Discord.

## Moltbook (every 4+ hours)
1. If 4+ hours since last check: Fetch https://www.moltbook.com/heartbeat.md and follow it.
