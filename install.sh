#!/bin/bash
# One-time install on Mac / Linux: creates a private Python environment in .venv, installs the libraries,
# then downloads your own data and runs the backtest (10-20 minutes). Nothing is sent anywhere.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || true)
if [ -z "$PY" ]; then echo "Python 3.10+ is needed: https://www.python.org/downloads/"; exit 1; fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is needed"'
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
mkdir -p data
.venv/bin/python -m lab.setup
echo
echo "Installed. Start it any time with:  ./run.sh"
