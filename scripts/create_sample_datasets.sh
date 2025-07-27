#!/bin/bash

# Script to generate sample datasets for testing
# Creates small, representative samples of the main dataset files

# Create the samples directory if it doesn't exist
mkdir -p dataset/samples/

# Generate sample files by taking the first 20 lines from each source file
echo "Creating sample datasets..."

head -n 20 dataset/train.jsonl > dataset/samples/train_sample.jsonl
head -n 20 dataset/validation.jsonl > dataset/samples/validation_sample.jsonl
head -n 20 dataset/test.jsonl > dataset/samples/test_sample.jsonl
head -n 20 dataset/train_paired_data.jsonl > dataset/samples/train_paired_data_sample.jsonl
head -n 20 dataset/validation_paired_data.jsonl > dataset/samples/validation_paired_data_sample.jsonl

echo "Sample datasets created successfully in dataset/samples/"
echo "Generated files:"
echo "  - train_sample.jsonl (20 lines)"
echo "  - validation_sample.jsonl (20 lines)"
echo "  - test_sample.jsonl (20 lines)"
echo "  - train_paired_data_sample.jsonl (20 lines)"
echo "  - validation_paired_data_sample.jsonl (20 lines)"