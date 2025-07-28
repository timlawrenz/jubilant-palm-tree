#!/bin/bash
# Master script for the end-to-end data preparation and training pipeline.

set -e # Exit immediately if a command exits with a non-zero status.

echo "🚀 STARTING: Full Data Preparation and Model Training Pipeline"
echo "=============================================================="

# --- Stage 1: Data Preparation ---
echo "\n[STAGE 1/3] Preparing Datasets..."
echo "---------------------------------"
echo "Step 1.1: Cloning source code repositories..."
./scripts/01_clone_repos.sh

echo "\nStep 1.2: Extracting Ruby methods..."
bundle exec ruby scripts/02_extract_methods.rb

echo "\nStep 1.3: Processing methods and calculating complexity..."
bundle exec ruby scripts/03_process_methods.rb

echo "\nStep 1.4: Assembling the core dataset..."
bundle exec ruby scripts/04_assemble_dataset.rb

echo "\nStep 1.5: Creating the paired (text-code) dataset..."
bundle exec ruby scripts/05_create_paired_dataset.rb

echo "\nStep 1.6: Splitting the paired dataset into train/validation sets..."
python scripts/split_paired_dataset.py

echo "\nStep 1.7: Pre-computing text embeddings for performance..."
python scripts/precompute_embeddings.py

echo "✅ STAGE 1 COMPLETE: All datasets have been prepared."

# --- Stage 2: Production Model Training ---
echo "\n[STAGE 2/3] Training Production Models..."
echo "---------------------------------------"
echo "Step 2.1: Training GNN complexity model (best_model.pt)..."
python train.py

echo "\nStep 2.2: Training AST Autoencoder (best_decoder.pt)..."
python train_autoencoder.py

echo "\nStep 2.3: Training Text-Code Alignment model (best_alignment_model.pt)..."
python train_alignment.py

echo "\nStep 2.4: Training Autoregressive Decoder (best_autoregressive_decoder.pt)..."
python train_autoregressive.py

echo "✅ STAGE 2 COMPLETE: All production models have been trained."

# --- Stage 3: Sample Data and Model Generation ---
echo "\n[STAGE 3/3] Generating Sample Assets for Testing..."
echo "---------------------------------------------------"
echo "Step 3.1: Creating sample datasets..."
./scripts/create_sample_datasets.sh

echo "\nStep 3.2: Training sample models..."
./scripts/train_sample_models.sh

echo "✅ STAGE 3 COMPLETE: All sample assets have been generated."

echo "\n=============================================================="
echo "🎉 SUCCESS: Full pipeline completed successfully!"
echo "=============================================================="