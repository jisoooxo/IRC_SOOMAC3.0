#!/bin/sh
# Protects the running training job from host OOM while Claude's packaging checks run.
# Logs MemAvailable every 5 s. Below the floor it writes STOP and kills ONLY:
#   - containers labelled soomac.claude-check=1
#   - docker image save/load CLI processes for the golden candidate tag
# It never signals any other process (the training PID is never a target).
set -u
LOG="$1"
FLOOR_KB="${2:-20971520}"   # 20 GiB
STOP="$LOG.STOP"
while :; do
    avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    echo "$(date +%T) MemAvailable_kB=$avail" >> "$LOG"
    if [ "$avail" -lt "$FLOOR_KB" ]; then
        echo "$(date +%T) BELOW FLOOR $FLOOR_KB -> stopping Claude checks only" >> "$LOG"
        touch "$STOP"
        for c in $(docker ps -q --filter label=soomac.claude-check=1); do docker kill "$c" >> "$LOG" 2>&1; done
        pkill -f "docker image (save|load).*golden-candidate-20261102" >> "$LOG" 2>&1
    fi
    sleep 5
done
