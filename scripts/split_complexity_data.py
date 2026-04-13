#!/usr/bin/env python3
"""Split the complexity dataset (validation.jsonl) into train/val JSONL files.

The original pipeline produced a single validation.jsonl with all 22K+ methods.
This script creates proper train/val splits for the GNN architecture search.
"""

import json
import os
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description="Split complexity JSONL into train/val")
    parser.add_argument("--input", default="dataset/validation.jsonl",
                        help="Path to full complexity JSONL (default: dataset/validation.jsonl)")
    parser.add_argument("--output-dir", default="dataset/",
                        help="Output directory (default: dataset/)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} methods from {args.input}")

    random.seed(args.seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - args.val_fraction))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")

    with open(train_path, "w") as f:
        f.writelines(train_lines)
    with open(val_path, "w") as f:
        f.writelines(val_lines)

    print(f"Train: {len(train_lines)} methods → {train_path}")
    print(f"Val:   {len(val_lines)} methods → {val_path}")

    # Verify data integrity
    for path, count in [(train_path, len(train_lines)), (val_path, len(val_lines))]:
        with open(path) as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                assert "complexity_score" in d, f"Missing complexity_score in {path} line {i}"
                assert "ast_json" in d, f"Missing ast_json in {path} line {i}"
        print(f"  ✓ {path} verified ({count} records)")


if __name__ == "__main__":
    main()
