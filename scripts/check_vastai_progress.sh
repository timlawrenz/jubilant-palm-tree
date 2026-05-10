#!/bin/bash
# Quick progress check for vast.ai training
echo "=== vast.ai Training Progress ($(date)) ==="
echo ""

LOG="/home/tim/source/activity/jubilant-palm-tree/runs/vastai/vastai_run.log"
REMOTE_LOG="root@76.66.207.49:/root/jubilant-palm-tree/runs/vastai_run.log"

# Try remote first, fall back to local rsync'd copy
DATA=$(ssh -p 44675 -o ConnectTimeout=5 root@76.66.207.49 'cat /root/jubilant-palm-tree/runs/vastai_run.log' 2>/dev/null)
if [ -z "$DATA" ]; then
  DATA=$(cat "$LOG" 2>/dev/null)
  echo "(Using local rsync'd copy)"
fi

# Latest progress line
echo "$DATA" | grep -oP "Epoch \d+:.*?\d+/494.*?svr=[^\]]*" | tail -1
echo ""

# SVR distribution
echo "SVR readings:"
echo "$DATA" | grep -oP "svr=\d+\.\d+%" | sort | uniq -c | sort -rn
echo ""

# Completed epochs
echo "Completed epochs:"
echo "$DATA" | grep -i "epoch.*summary\|epoch.*complete\|Epoch.*time" | tail -10
