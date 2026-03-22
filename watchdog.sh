#!/bin/bash
# genesis watchdog — monitors evolve.py, auto-restarts on crash
# Usage: nohup bash watchdog.sh > watchdog.log 2>&1 &

SCRIPT="evolve.py"
LOGFILE="run.log"
MAX_RESTARTS=20
RESTART_DELAY=10  # seconds between restarts

restarts=0

while [ $restarts -lt $MAX_RESTARTS ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $SCRIPT (restart #$restarts)"
    
    # Source uv environment and run
    source ~/.local/bin/env
    uv run python3 "$SCRIPT" >> "$LOGFILE" 2>&1
    exit_code=$?
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SCRIPT exited with code $exit_code"
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Clean exit — evolution complete."
        break
    fi
    
    # Log the crash
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CRASH detected. Last 5 lines:" 
    tail -5 "$LOGFILE"
    
    restarts=$((restarts + 1))
    
    if [ $restarts -lt $MAX_RESTARTS ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in ${RESTART_DELAY}s (will resume from checkpoint)..."
        sleep $RESTART_DELAY
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max restarts ($MAX_RESTARTS) reached. Giving up."
    fi
done
