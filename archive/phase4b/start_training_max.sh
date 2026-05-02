#!/bin/bash
# MAXIMUM PERFORMANCE training script - uses as much GPU/CPU as possible

cd /home/tim/source/activity/jubilant-palm-tree

echo "🔥 Starting MAXIMUM PERFORMANCE training..."
echo "Settings:"
echo "  - Batch size: 256 (very large batches)"
echo "  - Data workers: 12 (maximum parallelism)"
echo "  - Epochs per level: 2 (minimal but should still converge)"
echo "  - Log file: training_log.txt"
echo ""
echo "Expected speedup: ~20x faster"
echo "Estimated time: 15-30 minutes"
echo "VRAM usage: 40-60%"
echo ""

# Run with maximum performance settings
python3 train_hierarchical.py \
  --batch-size 256 \
  --num-workers 12 \
  --epochs-per-level 2 \
  --learning-rate 1e-4 \
  2>&1 | tee training_log.txt
