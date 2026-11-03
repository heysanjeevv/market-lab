#!/bin/bash
# Starts the dashboard at http://localhost:8810 (and opens it). If it is already running, just opens it.
cd "$(dirname "$0")"
URL=http://localhost:8810
opener=$(command -v open || command -v xdg-open || echo true)
if curl -s -m 2 -o /dev/null "$URL/api/status"; then echo "Already running: $URL"; "$opener" "$URL"; exit 0; fi
[ -x .venv/bin/python ] || { echo "Run ./install.sh first."; exit 1; }
( sleep 4; "$opener" "$URL" ) >/dev/null 2>&1 &
exec .venv/bin/python -u app.py
