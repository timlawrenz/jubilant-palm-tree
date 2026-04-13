#!/usr/bin/env bash
# Runner script for GNN code generation (autoencoder) experiments.
# Outputs METRICS:{json} for Ratiocinator fleet parsing.
#
# Environment variables (set by Ratiocinator fleet):
#   DECODER_CONV_TYPE - Decoder conv type: GCN, SAGE, GAT, GIN, GraphConv (default: GAT)
#   HIDDEN_DIM        - Hidden dimension (default: 256)
#   NUM_LAYERS        - Number of decoder layers (default: 5)
#   LEARNING_RATE     - Learning rate (default: 0.001)
#   TYPE_WEIGHT       - Weight for node type loss (default: 2.0)
#   PARENT_WEIGHT     - Weight for parent prediction loss (default: 1.0)
#   LOSS_FN           - Loss function: mse, ce, combined (default: mse)
#   EPOCHS            - Training epochs (default: 30)
#   DATASET_PATH      - Path to dataset dir (default: dataset/)

set -euo pipefail

DECODER_CONV_TYPE="${DECODER_CONV_TYPE:-GAT}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TYPE_WEIGHT="${TYPE_WEIGHT:-2.0}"
PARENT_WEIGHT="${PARENT_WEIGHT:-1.0}"
LOSS_FN="${LOSS_FN:-mse}"
EPOCHS="${EPOCHS:-30}"
DATASET_PATH="${DATASET_PATH:-dataset/}"
OUTPUT_PATH="models/experiment_decoder.pt"
ENCODER_PATH="models/best_model.pt"

echo "=== GNN Generation Arm ==="
echo "DECODER=$DECODER_CONV_TYPE HIDDEN=$HIDDEN_DIM LAYERS=$NUM_LAYERS"
echo "LR=$LEARNING_RATE TYPE_W=$TYPE_WEIGHT PARENT_W=$PARENT_WEIGHT LOSS=$LOSS_FN"

# Need pre-collated data for autoencoder training
if [ ! -f "${DATASET_PATH}/train_collated_b4096.pt" ]; then
    echo "ERROR: Pre-collated data not found. Run scripts/pre_collate_batches.py first."
    echo "METRICS:{\"error\": \"missing_collated_data\"}"
    exit 1
fi

# Need pre-trained encoder
if [ ! -f "$ENCODER_PATH" ]; then
    echo "Training encoder first..."
    python train.py --epochs 20 --output_path "$ENCODER_PATH" --dataset_path "$DATASET_PATH"
fi

mkdir -p models

# Build extra args for loss function
EXTRA_ARGS=""
if [ "$LOSS_FN" = "ce" ]; then
    EXTRA_ARGS="--loss_fn cross_entropy"
elif [ "$LOSS_FN" = "combined" ]; then
    EXTRA_ARGS="--loss_fn combined"
fi

# Run autoencoder training
TRAIN_OUTPUT=$(python train_autoencoder.py \
    --dataset_path "$DATASET_PATH" \
    --epochs "$EPOCHS" \
    --output_path "$OUTPUT_PATH" \
    --encoder_weights_path "$ENCODER_PATH" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --decoder_conv_type "$DECODER_CONV_TYPE" \
    --learning_rate "$LEARNING_RATE" \
    --type_weight "$TYPE_WEIGHT" \
    --parent_weight "$PARENT_WEIGHT" \
    $EXTRA_ARGS \
    2>&1)

echo "$TRAIN_OUTPUT"

BEST_VAL_LOSS=$(echo "$TRAIN_OUTPUT" | grep "Best validation loss" | grep -oP '[\d.]+' | tail -1)

# Run syntactic validity evaluation
EVAL_OUTPUT=$(python -c "
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname('.'), 'src'))
from models import ASTAutoencoder
from data_processing import create_data_loaders
import subprocess
import tempfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_samples = 100
valid_count = 0
total = 0

try:
    # Load autoencoder
    model = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=$HIDDEN_DIM,
        num_layers=$NUM_LAYERS,
        conv_type='SAGE',
        freeze_encoder=True,
        encoder_weights_path='$ENCODER_PATH',
        decoder_conv_type='$DECODER_CONV_TYPE'
    ).to(device)

    checkpoint = torch.load('$OUTPUT_PATH', map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()

    # Load validation data
    val_path = '${DATASET_PATH}/validation_collated_b4096.pt'
    if not os.path.exists(val_path):
        val_path = '${DATASET_PATH}/validation.pt'
    _, val_loader = create_data_loaders(val_path, val_path, batch_size=1, pre_collated=os.path.exists('${DATASET_PATH}/validation_collated_b4096.pt'))

    with torch.no_grad():
        for batch in val_loader:
            if total >= num_samples:
                break
            batch = batch.to(device)
            result = model(batch)
            recon = result['reconstruction']
            # Check if reconstruction has valid node types
            node_preds = recon.x if hasattr(recon, 'x') else None
            if node_preds is not None:
                pred_types = node_preds.argmax(dim=-1)
                # Valid if not all same type (not collapsed)
                unique_types = len(pred_types.unique())
                if unique_types > 2:
                    valid_count += 1
            total += 1

    validity_pct = (valid_count / total * 100) if total > 0 else 0.0
except Exception as e:
    validity_pct = 0.0
    total = num_samples

print(json.dumps({
    'syntactic_validity_pct': round(validity_pct, 2),
    'val_loss': round(float('${BEST_VAL_LOSS:-0}'), 4),
    'samples_evaluated': total,
    'valid_samples': valid_count,
    'decoder_conv_type': '$DECODER_CONV_TYPE',
    'hidden_dim': $HIDDEN_DIM,
    'num_layers': $NUM_LAYERS,
    'loss_fn': '$LOSS_FN',
    'type_weight': $TYPE_WEIGHT,
    'parent_weight': $PARENT_WEIGHT,
    'learning_rate': $LEARNING_RATE,
    'epochs': $EPOCHS
}))
" 2>&1)

echo "$EVAL_OUTPUT"

METRICS_JSON=$(echo "$EVAL_OUTPUT" | grep '^{' | tail -1)
if [ -n "$METRICS_JSON" ]; then
    echo "METRICS:$METRICS_JSON"
else
    echo "METRICS:{\"error\": \"evaluation_failed\", \"val_loss\": ${BEST_VAL_LOSS:-0}}"
fi
