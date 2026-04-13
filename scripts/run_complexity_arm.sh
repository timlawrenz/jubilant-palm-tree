#!/usr/bin/env bash
# Runner script for GNN complexity prediction experiments.
# Outputs METRICS:{json} for Ratiocinator fleet parsing.
#
# Environment variables (set by Ratiocinator fleet):
#   CONV_TYPE     - GNN convolution type: GCN, SAGE, GAT, GIN, GraphConv (default: SAGE)
#   HIDDEN_DIM    - Hidden dimension (default: 64)
#   NUM_LAYERS    - Number of GNN layers (default: 3)
#   DROPOUT       - Dropout rate (default: 0.1)
#   LEARNING_RATE - Learning rate (default: 0.001)
#   EPOCHS        - Training epochs (default: 50)
#   BATCH_SIZE    - Batch size (default: 32)
#   DATASET_PATH  - Path to dataset dir (default: dataset/)

set -uo pipefail

CONV_TYPE="${CONV_TYPE:-SAGE}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATASET_PATH="${DATASET_PATH:-dataset/}"
OUTPUT_PATH="models/experiment_model.pt"

echo "=== GNN Complexity Arm ==="
echo "CONV_TYPE=$CONV_TYPE HIDDEN_DIM=$HIDDEN_DIM NUM_LAYERS=$NUM_LAYERS"
echo "DROPOUT=$DROPOUT LR=$LEARNING_RATE EPOCHS=$EPOCHS BATCH=$BATCH_SIZE"

# Pull LFS files if they are pointers (e.g., after shallow clone)
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    echo "Pulling LFS files..."
    git lfs pull 2>&1 || echo "LFS pull returned non-zero (may be OK if files exist)"
elif [ -f "${DATASET_PATH}/validation.jsonl" ] && head -1 "${DATASET_PATH}/validation.jsonl" | grep -q "^version https://git-lfs"; then
    echo "ERROR: LFS pointer files detected but git-lfs not installed"
    echo "Install with: apt-get install -y git-lfs && git lfs pull"
    exit 1
fi

# Ensure train/val split exists
if [ ! -f "${DATASET_PATH}/train.jsonl" ]; then
    echo "Creating train/val split..."
    python scripts/split_complexity_data.py \
        --input "${DATASET_PATH}/validation.jsonl" \
        --output-dir "${DATASET_PATH}"
fi

# Symlink val.jsonl as validation.jsonl if train.py expects it
if [ -f "${DATASET_PATH}/val.jsonl" ] && [ ! -f "${DATASET_PATH}/validation_split.jsonl" ]; then
    cp "${DATASET_PATH}/val.jsonl" "${DATASET_PATH}/validation_split.jsonl"
fi

mkdir -p models

# Symlink validation.jsonl → val.jsonl for compatibility
if [ -f "${DATASET_PATH}/val.jsonl" ]; then
    ORIG_VAL="${DATASET_PATH}/validation.jsonl"
    if [ -f "$ORIG_VAL" ] && ! [ -L "$ORIG_VAL" ]; then
        mv "$ORIG_VAL" "${DATASET_PATH}/validation_full.jsonl"
    fi
    ln -sf val.jsonl "${DATASET_PATH}/validation.jsonl"
fi

# Run training — stream output directly (no capturing)
TRAIN_LOG="/tmp/train_output_$$.log"
python train.py \
    --dataset_path "$DATASET_PATH" \
    --epochs "$EPOCHS" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --conv_type "$CONV_TYPE" \
    --dropout "$DROPOUT" \
    --num_workers 0 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "ERROR: train.py exited with code $TRAIN_RC"
    echo "METRICS:{\"error\": \"training_failed\", \"exit_code\": $TRAIN_RC}"
    exit 1
fi

# Extract best validation loss from training output
BEST_VAL_LOSS=$(grep "Best validation loss" "$TRAIN_LOG" | grep -oP '[\d.]+' | tail -1)

# Run evaluation to get MAE on the validation set
python -c "
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname('.'), 'src'))
from data_processing import create_data_loaders
from models import RubyComplexityGNN
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('$OUTPUT_PATH', map_location=device, weights_only=False)
config = checkpoint['model_config']
model = RubyComplexityGNN(
    input_dim=config.get('input_dim', 74),
    hidden_dim=config.get('hidden_dim', 64),
    num_layers=config.get('num_layers', 3),
    conv_type=config.get('conv_type', 'SAGE'),
    dropout=config.get('dropout', 0.1)
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

val_path = os.path.join('${DATASET_PATH}', 'val.jsonl')
if not os.path.exists(val_path):
    val_path = os.path.join('${DATASET_PATH}', 'validation.jsonl')
_, val_loader = create_data_loaders(val_path, val_path, batch_size=64, shuffle=False, num_workers=0)

all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        preds = model(batch).squeeze()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch.y.cpu().numpy().tolist())

preds = np.array(all_preds)
targets = np.array(all_targets)
mae = float(np.mean(np.abs(preds - targets)))
mse = float(np.mean((preds - targets) ** 2))
r2 = float(1 - np.sum((targets - preds)**2) / np.sum((targets - np.mean(targets))**2))

print('METRICS:' + json.dumps({
    'val_mae': round(mae, 4),
    'val_mse': round(mse, 4),
    'val_r2': round(r2, 4),
    'best_val_loss': round(float('${BEST_VAL_LOSS:-0}'), 4),
    'conv_type': '$CONV_TYPE',
    'hidden_dim': $HIDDEN_DIM,
    'num_layers': $NUM_LAYERS,
    'dropout': $DROPOUT,
    'learning_rate': $LEARNING_RATE,
    'epochs': $EPOCHS
}))
" 2>&1

rm -f "$TRAIN_LOG"
