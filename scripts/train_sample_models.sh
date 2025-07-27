#!/bin/bash

# Script to train sample models for testing
# This script creates small, fast-training models using sample datasets
# for use in CI/CD and local development environments.

set -e  # Exit on any error

echo "🚀 Training Sample Models for Testing"
echo "===================================="
echo ""

# Create models/samples directory if it doesn't exist
SAMPLE_MODELS_DIR="models/samples"
echo "📁 Creating sample models directory: $SAMPLE_MODELS_DIR"
mkdir -p "$SAMPLE_MODELS_DIR"

# Configuration for sample training
DATASET_PATH="dataset/samples/"
EPOCHS=1
SAMPLE_LEARNING_RATE=0.01
SAMPLE_BATCH_SIZE=8

echo "📋 Sample Training Configuration:"
echo "   Dataset path: $DATASET_PATH"
echo "   Epochs: $EPOCHS"
echo "   Learning rate: $SAMPLE_LEARNING_RATE"
echo "   Batch size: $SAMPLE_BATCH_SIZE"
echo ""

# Check if sample dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Sample dataset directory '$DATASET_PATH' does not exist!"
    echo "   Please ensure the sample datasets have been created first."
    exit 1
fi

# Function to check if required files exist
check_sample_files() {
    local required_files=("train_sample.jsonl" "validation_sample.jsonl" "train_paired_data_sample.jsonl" "validation_paired_data_sample.jsonl")
    for file in "${required_files[@]}"; do
        if [ ! -f "$DATASET_PATH/$file" ]; then
            echo "❌ Error: Required sample file '$DATASET_PATH/$file' not found!"
            exit 1
        fi
    done
    echo "✅ All required sample dataset files found"
}

check_sample_files

echo ""
echo "🧠 Training sample models..."
echo "=============================="

# 1. Train main complexity prediction model
echo ""
echo "1️⃣  Training main complexity prediction model..."
echo "   Input:  $DATASET_PATH"
echo "   Output: $SAMPLE_MODELS_DIR/best_model.pt"
python train.py \
    --dataset_path "$DATASET_PATH" \
    --epochs $EPOCHS \
    --output_path "$SAMPLE_MODELS_DIR/best_model.pt" \
    --learning_rate $SAMPLE_LEARNING_RATE \
    --batch_size $SAMPLE_BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✅ Main model training completed successfully"
else
    echo "❌ Main model training failed"
    exit 1
fi

# 2. Train autoencoder model
echo ""
echo "2️⃣  Training autoencoder model..."
echo "   Input:  $DATASET_PATH"
echo "   Output: $SAMPLE_MODELS_DIR/best_decoder.pt"
python train_autoencoder.py \
    --dataset_path "$DATASET_PATH" \
    --epochs $EPOCHS \
    --output_path "$SAMPLE_MODELS_DIR/best_decoder.pt" \
    --encoder_weights_path "$SAMPLE_MODELS_DIR/best_model.pt" \
    --learning_rate $SAMPLE_LEARNING_RATE \
    --batch_size $SAMPLE_BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✅ Autoencoder training completed successfully"
else
    echo "❌ Autoencoder training failed"
    exit 1
fi

# 3. Train alignment model
echo ""
echo "3️⃣  Training alignment model..."
echo "   Input:  $DATASET_PATH"
echo "   Output: $SAMPLE_MODELS_DIR/best_alignment_model.pt"
python train_alignment.py \
    --dataset_path "$DATASET_PATH" \
    --epochs $EPOCHS \
    --output_path "$SAMPLE_MODELS_DIR/best_alignment_model.pt" \
    --code_encoder_weights_path "$SAMPLE_MODELS_DIR/best_model.pt" \
    --learning_rate $SAMPLE_LEARNING_RATE \
    --batch_size $SAMPLE_BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✅ Alignment model training completed successfully"
else
    echo "❌ Alignment model training failed"
    exit 1
fi

# 4. Train autoregressive model
echo ""
echo "4️⃣  Training autoregressive model..."
echo "   Input:  $DATASET_PATH"
echo "   Output: $SAMPLE_MODELS_DIR/best_autoregressive_decoder.pt"
python train_autoregressive.py \
    --dataset_path "$DATASET_PATH" \
    --epochs $EPOCHS \
    --output_path "$SAMPLE_MODELS_DIR/best_autoregressive_decoder.pt" \
    --alignment_model_path "$SAMPLE_MODELS_DIR/best_alignment_model.pt" \
    --code_encoder_path "$SAMPLE_MODELS_DIR/best_model.pt" \
    --learning_rate $SAMPLE_LEARNING_RATE \
    --batch_size 2 \
    --patience 3

if [ $? -eq 0 ]; then
    echo "✅ Autoregressive model training completed successfully"
else
    echo "❌ Autoregressive model training failed"
    exit 1
fi

echo ""
echo "🎉 Sample Model Training Complete!"
echo "=================================="
echo ""
echo "✅ All sample models have been successfully trained and saved to:"
echo "   📁 $SAMPLE_MODELS_DIR/"
echo ""
echo "📋 Generated sample models:"
echo "   🧠 best_model.pt                    - Main complexity prediction model"
echo "   🔄 best_decoder.pt                  - AST autoencoder decoder"
echo "   🔗 best_alignment_model.pt          - Text-code alignment model"
echo "   📝 best_autoregressive_decoder.pt   - Autoregressive AST decoder"
echo ""
echo "These lightweight models can be used for:"
echo "   • Fast unit testing"
echo "   • CI/CD pipeline validation"
echo "   • Local development and debugging"
echo "   • Integration testing"
echo ""
echo "✨ Sample model training completed successfully! ✨"