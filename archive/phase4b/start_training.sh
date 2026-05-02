#!/bin/bash
# Script to start hierarchical model training with progress monitoring

cd /home/tim/source/activity/jubilant-palm-tree

echo "Starting hierarchical AST decoder training..."
echo "Log file: training_log.txt"
echo ""

python3 train_hierarchical.py 2>&1 | tee training_log.txt
