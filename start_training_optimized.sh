#!/bin/bash
# Optimized training script with higher batch size and parallel data loading

cd /home/tim/source/activity/jubilant-palm-tree

echo "🚀 Starting OPTIMIZED hierarchical AST decoder training..."
echo "Settings:"
echo "  - Batch size: 128 (increased from 16 = 8x larger batches)"
echo "  - Data workers: 8 (parallel data loading)"
echo "  - Epochs per level: 3 (reduced from 5 for faster iteration)"
echo "  - Log file: training_log.txt"
echo ""
echo "Expected speedup: ~10-15x faster overall"
echo "Estimated time: 30-60 minutes (down from 5-6 hours)"
echo ""

# Run with optimized settings
python3 train_hierarchical.py \
  --batch-size 128 \
  --num-workers 8 \
  --epochs-per-level 3 \
  --learning-rate 1e-4 \
  2>&1 | tee training_log.txt
