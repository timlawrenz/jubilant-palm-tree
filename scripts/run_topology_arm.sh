#!/usr/bin/env bash
# Runner script for Track 4: Decoder Topology experiments.
# Tests tree-aware decoding (teacher_forced / iterative) vs chain baseline.
# Outputs METRICS:{json} for Ratiocinator fleet parsing.
#
# Environment variables (set by Ratiocinator fleet):
#   DECODER_EDGE_MODE - chain, teacher_forced, iterative (default: teacher_forced)
#   DECODER_CONV_TYPE - GCN, SAGE, GAT, GIN, GraphConv (default: GAT)
#   HIDDEN_DIM        - Hidden dimension (default: 256)
#   NUM_LAYERS        - Number of decoder layers (default: 5)
#   LEARNING_RATE     - Learning rate (default: 0.001)
#   TYPE_WEIGHT       - Weight for node type loss (default: 2.0)
#   PARENT_WEIGHT     - Weight for parent prediction loss (default: 1.0)
#   LOSS_FN           - Loss function: improved, comprehensive, simple, original (default: improved)
#   EPOCHS            - Training epochs (default: 30)
#   DATASET_PATH      - Path to dataset dir (default: dataset/)

set -uo pipefail

DECODER_EDGE_MODE="${DECODER_EDGE_MODE:-teacher_forced}"
DECODER_CONV_TYPE="${DECODER_CONV_TYPE:-GAT}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TYPE_WEIGHT="${TYPE_WEIGHT:-2.0}"
PARENT_WEIGHT="${PARENT_WEIGHT:-1.0}"
LOSS_FN="${LOSS_FN:-improved}"
EPOCHS="${EPOCHS:-30}"
DATASET_PATH="${DATASET_PATH:-dataset/}"
OUTPUT_PATH="models/experiment_topology_decoder.pt"
ENCODER_PATH="models/best_model.pt"

echo "=== Track 4: Decoder Topology Arm ==="
echo "EDGE_MODE=$DECODER_EDGE_MODE DECODER=$DECODER_CONV_TYPE HIDDEN=$HIDDEN_DIM LAYERS=$NUM_LAYERS"
echo "LR=$LEARNING_RATE TYPE_W=$TYPE_WEIGHT PARENT_W=$PARENT_WEIGHT LOSS=$LOSS_FN"

# Pull LFS files if they are pointers (e.g., after shallow clone)
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    echo "Pulling LFS files..."
    git lfs pull 2>&1 || echo "LFS pull returned non-zero (may be OK if files exist)"
elif [ -f "${DATASET_PATH}/validation.jsonl" ] && head -1 "${DATASET_PATH}/validation.jsonl" | grep -q "^version https://git-lfs"; then
    echo "ERROR: LFS pointer files detected but git-lfs not installed"
    exit 1
fi

# Ensure train/val split exists
if [ ! -f "${DATASET_PATH}/train.jsonl" ]; then
    echo "Creating train/val split..."
    python scripts/split_complexity_data.py \
        --input "${DATASET_PATH}/validation.jsonl" \
        --output-dir "${DATASET_PATH}"
fi

# Symlink validation.jsonl → val.jsonl for compatibility
if [ -f "${DATASET_PATH}/val.jsonl" ]; then
    ORIG_VAL="${DATASET_PATH}/validation.jsonl"
    if [ -f "$ORIG_VAL" ] && ! [ -L "$ORIG_VAL" ]; then
        mv "$ORIG_VAL" "${DATASET_PATH}/validation_full.jsonl"
    fi
    ln -sf val.jsonl "${DATASET_PATH}/validation.jsonl"
fi

# Need pre-trained encoder
if [ ! -f "$ENCODER_PATH" ]; then
    echo "Training encoder first..."
    python train.py --epochs 20 --output_path "$ENCODER_PATH" --dataset_path "$DATASET_PATH" --num_workers 0
fi

mkdir -p models

# Run autoencoder training with tree-aware decoder — stream output
TRAIN_LOG="/tmp/topo_train_$$.log"
python train_autoencoder.py \
    --dataset_path "$DATASET_PATH" \
    --epochs "$EPOCHS" \
    --output_path "$OUTPUT_PATH" \
    --encoder_weights_path "$ENCODER_PATH" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --decoder_conv_type "$DECODER_CONV_TYPE" \
    --decoder_edge_mode "$DECODER_EDGE_MODE" \
    --learning_rate "$LEARNING_RATE" \
    --type_weight "$TYPE_WEIGHT" \
    --parent_weight "$PARENT_WEIGHT" \
    --loss_fn "$LOSS_FN" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "ERROR: train_autoencoder.py exited with code $TRAIN_RC"
    echo "METRICS:{\"error\": \"training_failed\", \"exit_code\": $TRAIN_RC}"
    exit 1
fi

BEST_VAL_LOSS=$(grep "Best validation loss" "$TRAIN_LOG" | grep -oP '[\d.]+' | tail -1)

# Run syntactic validity evaluation
python -c "
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname('.'), 'src'))
from models import ASTAutoencoder
from data_processing import create_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_samples = 100
valid_count = 0
total = 0

try:
    model = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=$HIDDEN_DIM,
        num_layers=$NUM_LAYERS,
        conv_type='SAGE',
        freeze_encoder=True,
        encoder_weights_path='$ENCODER_PATH',
        decoder_conv_type='$DECODER_CONV_TYPE',
        decoder_edge_mode='$DECODER_EDGE_MODE',
    ).to(device)

    checkpoint = torch.load('$OUTPUT_PATH', map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()

    # Load val data
    val_path = os.path.join('${DATASET_PATH}', 'val.jsonl')
    if not os.path.exists(val_path):
        val_path = os.path.join('${DATASET_PATH}', 'validation.jsonl')
    _, val_loader = create_data_loaders(val_path, val_path, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in val_loader:
            if total >= num_samples:
                break
            batch = batch.to(device)
            result = model(batch)
            recon = result['reconstruction']
            node_feats = recon.get('node_features') if isinstance(recon, dict) else None
            if node_feats is not None:
                pred_types = node_feats.argmax(dim=-1)
                unique_types = len(pred_types.unique())
                if unique_types > 2:
                    valid_count += 1
            total += 1

    validity_pct = (valid_count / total * 100) if total > 0 else 0.0
except Exception as e:
    validity_pct = 0.0
    total = num_samples
    print(f'Eval error: {e}', file=sys.stderr)

print('METRICS:' + json.dumps({
    'syntactic_validity_pct': round(validity_pct, 2),
    'val_loss': round(float('${BEST_VAL_LOSS:-0}'), 4),
    'samples_evaluated': total,
    'valid_samples': valid_count,
    'decoder_edge_mode': '$DECODER_EDGE_MODE',
    'decoder_conv_type': '$DECODER_CONV_TYPE',
    'hidden_dim': $HIDDEN_DIM,
    'num_layers': $NUM_LAYERS,
    'loss_fn': '$LOSS_FN',
    'type_weight': $TYPE_WEIGHT,
    'parent_weight': $PARENT_WEIGHT,
    'learning_rate': $LEARNING_RATE,
    'epochs': $EPOCHS,
}))
" 2>&1

rm -f "$TRAIN_LOG"
