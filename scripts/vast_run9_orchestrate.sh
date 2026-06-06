#!/bin/bash
# Run 9 Vast.ai Orchestration Script
# - Waits for checkpoints on remote
# - Starts training
# - Monitors that training is running
# - Evacuates checkpoints+logs on completion
# - Destroys instance

set -e

INSTANCE_ID=36914029
REMOTE="root@81.166.173.12"
PORT=10405
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30"
SCP="scp -P $PORT -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh -p $PORT -o StrictHostKeyChecking=no"
REMOTE_DIR="/root/jubilant-palm-tree"
LOCAL_DIR="/home/tim/source/activity/jubilant-palm-tree"
LOCAL_CKPT="$LOCAL_DIR/checkpoints/rlaif/vastai_run9"
LOG="$LOCAL_DIR/runs/vastai/run9_orchestration.log"

mkdir -p "$LOCAL_CKPT"
mkdir -p "$LOCAL_DIR/runs/vastai"
exec > >(tee -a "$LOG") 2>&1

echo "=== Run 9 Orchestration Started: $(date) ==="
echo "Instance: $INSTANCE_ID | Remote: $REMOTE:$PORT"

# ─── PHASE 1: Wait for checkpoint transfer ───
echo ""
echo "=== [1/5] Waiting for checkpoint transfer (external rsync proc) ==="
MAX_WAIT=7200  # 2 hours
WAITED=0
while true; do
  BASE_OK=$($SSH $REMOTE "[ -f $REMOTE_DIR/checkpoints/num_dit_epoch_340.pt ] && echo 1 || echo 0" 2>/dev/null)
  RUN8_OK=$($SSH $REMOTE "[ -f $REMOTE_DIR/checkpoints/rlaif/vastai_run8/rlaif_struct_epoch_1.pt ] && echo 1 || echo 0" 2>/dev/null)
  if [ "$BASE_OK" = "1" ] && [ "$RUN8_OK" = "1" ]; then
    echo "  Both checkpoints present on remote. Proceeding."
    break
  fi
  echo "  [$(date +%H:%M:%S)] Base=$BASE_OK Run8=$RUN8_OK — waiting 60s..."
  sleep 60
  WAITED=$((WAITED + 60))
  if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Checkpoints not transferred after $MAX_WAIT seconds. Aborting."
    exit 1
  fi
done

# Verify checkpoint integrity
echo "  Checking checkpoint sizes..."
$SSH $REMOTE "ls -lh $REMOTE_DIR/checkpoints/num_dit_epoch_340.pt $REMOTE_DIR/checkpoints/rlaif/vastai_run8/rlaif_struct_epoch_1.pt"

# ─── PHASE 2: Start Training ───
echo ""
echo "=== [2/5] Starting Run 9 Training ==="
echo "  Config: --resume run8_epoch1 --beta-struct 0.2 --beta-recon 0.8 --lr 5e-6 --max-epochs 10 --batch-size 2 --grad-steps 6"

$SSH $REMOTE "
  cd $REMOTE_DIR
  mkdir -p checkpoints/rlaif/vastai_run9
  nohup bash -c '
    export PYTHONPATH=/root/jubilant-palm-tree
    python3 -m src.rlaif.train_rlaif \
      --checkpoint checkpoints/num_dit_epoch_340.pt \
      --resume checkpoints/rlaif/vastai_run8/rlaif_struct_epoch_1.pt \
      --beta-struct 0.2 \
      --beta-recon 0.8 \
      --lr 5e-6 \
      --max-epochs 10 \
      --batch-size 2 \
      --grad-steps 6 \
      --eval-every 10 \
      --save-every 1 \
      > /root/run9_train.log 2>&1
    echo \"EXIT_CODE: \$?\" >> /root/run9_train.log
  ' &
  echo \"Training PID: \$!\"
  echo \$! > /root/run9_train.pid
  sleep 5
  echo 'Background training started. PID:' \$(cat /root/run9_train.pid)
"

echo "  Training launched."

# ─── PHASE 3: Verify training running (5 min check) ───
echo ""
echo "=== [3/5] Verifying training still running (waiting 5 min) ==="
sleep 300

$SSH $REMOTE "
  PID=\$(cat /root/run9_train.pid 2>/dev/null)
  if [ -z \"\$PID\" ]; then
    echo 'ERROR: PID file missing'
    exit 1
  fi
  if kill -0 \$PID 2>/dev/null; then
    echo \"Training RUNNING (PID \$PID). Last log lines:\"
    tail -15 /root/run9_train.log
  else
    echo 'ERROR: Training process has died!'
    cat /root/run9_train.log
    exit 1
  fi
"

echo "  Training confirmed running."

# ─── PHASE 4: Wait for training completion ───
echo ""
echo "=== [4/5] Monitoring training to completion ==="
MAX_TRAIN_WAIT=43200  # 12 hours
WAITED=0
while true; do
  STATUS=$($SSH $REMOTE "
    PID=\$(cat /root/run9_train.pid 2>/dev/null)
    if kill -0 \$PID 2>/dev/null; then
      echo 'running'
    else
      echo 'done'
    fi
  " 2>/dev/null)
  
  LAST_LOG=$($SSH $REMOTE "tail -3 /root/run9_train.log 2>/dev/null" 2>/dev/null)
  echo "  [$(date +%H:%M:%S)] Status=$STATUS | $LAST_LOG"
  
  if [ "$STATUS" = "done" ]; then
    # Check if it exited cleanly
    EXIT_CODE=$($SSH $REMOTE "grep 'EXIT_CODE:' /root/run9_train.log | tail -1" 2>/dev/null)
    echo "  Training process ended. $EXIT_CODE"
    break
  fi
  
  sleep 600  # Check every 10 min
  WAITED=$((WAITED + 600))
  if [ $WAITED -ge $MAX_TRAIN_WAIT ]; then
    echo "WARNING: Training hasn't completed after 12h. Proceeding to evacuation."
    break
  fi
done

# ─── PHASE 5: Evacuate all training data ───
echo ""
echo "=== [5/5] Evacuating training data ==="
SYNC_TAG="vastai_run9_$(date +%Y-%m-%d_%H%M)"
mkdir -p "$LOCAL_CKPT"

# Training log
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  "$REMOTE:/root/run9_train.log" \
  "$LOCAL_DIR/runs/vastai/run9_train.log"

# Checkpoints
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  "$REMOTE:$REMOTE_DIR/checkpoints/rlaif/vastai_run9/" \
  "$LOCAL_CKPT/"

# TensorBoard runs
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  --include="rlaif_run9*/" \
  --include="rlaif_run9*/**" \
  --exclude="*" \
  "$REMOTE:$REMOTE_DIR/runs/" \
  "$LOCAL_DIR/runs/vastai_run9_sync/"

echo "  Evacuation complete. Local checkpoints: $LOCAL_CKPT"
ls -lh "$LOCAL_CKPT/"

# Also back up to NAS
NAS_DIR="/mnt/nas-ai-models/training-data/jubilant-palm-tree/checkpoints/rlaif/vastai_run9"
if [ -d "/mnt/nas-ai-models" ]; then
  mkdir -p "$NAS_DIR"
  rsync -av "$LOCAL_CKPT/" "$NAS_DIR/"
  echo "  NAS backup complete: $NAS_DIR"
fi

# ─── DESTROY INSTANCE ───
echo ""
echo "=== Destroying instance $INSTANCE_ID ==="
vastai destroy instance $INSTANCE_ID
echo "  Instance destroyed."

echo ""
echo "=== Run 9 Orchestration Complete: $(date) ==="
echo "  Checkpoints at: $LOCAL_CKPT"
