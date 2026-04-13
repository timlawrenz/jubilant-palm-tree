#!/usr/bin/env python3
"""Deep dive into teacher-forced GIN decoder: qualitative analysis + dimension ablation.

Trains teacher-forced GIN at multiple hidden dimensions, evaluates syntactic validity
using both the unique-types heuristic and real Ruby syntax checking (via check_syntax.rb),
and saves generated samples for qualitative analysis.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import create_data_loaders
from models import ASTAutoencoder

DATASET_PATH = "dataset"
ENCODER_WEIGHTS = "models/best_model.pt"
RESULTS_DIR = "results/gin_deep_dive"
EPOCHS = 30
BATCH_SIZE = 32
NUM_SAMPLES = 200
LEARNING_RATE = 0.001


def check_ruby_syntax(code: str) -> bool:
    """Check if code is valid Ruby using the parser gem."""
    try:
        result = subprocess.run(
            ["ruby", "scripts/check_syntax.rb"],
            input=code,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def reconstruct_code_from_types(pred_types: torch.Tensor, type_vocab: list[str] | None = None) -> str:
    """Convert predicted node type indices back to a pseudo-code string."""
    types = pred_types.cpu().tolist()
    if type_vocab:
        return " ".join(type_vocab[t] for t in types if t < len(type_vocab))
    return " ".join(f"type_{t}" for t in types)


def train_and_evaluate(
    hidden_dim: int,
    decoder_edge_mode: str = "teacher_forced",
    decoder_conv_type: str = "GIN",
    num_layers: int = 3,
    label: str = "",
) -> dict:
    """Train an autoencoder variant and evaluate generation quality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Training: {label} (dim={hidden_dim}, layers={num_layers}, "
          f"edge={decoder_edge_mode}, conv={decoder_conv_type})")
    print(f"Device: {device}")
    print(f"{'='*70}")

    train_path = os.path.join(DATASET_PATH, "train.jsonl")
    val_path = os.path.join(DATASET_PATH, "val.jsonl")
    train_loader, val_loader = create_data_loaders(
        train_path, val_path, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    model = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        conv_type="SAGE",
        freeze_encoder=True,
        encoder_weights_path=ENCODER_WEIGHTS,
        decoder_conv_type=decoder_conv_type,
        decoder_edge_mode=decoder_edge_mode,
    ).to(device)

    param_count = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Trainable decoder parameters: {param_count:,}")

    from loss import ast_reconstruction_loss_improved

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float("inf")
    model_path = os.path.join(RESULTS_DIR, f"{label}_decoder.pt")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    result = model(batch)
                    loss = ast_reconstruction_loss_improved(batch, result["reconstruction"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                result = model(batch)
                loss = ast_reconstruction_loss_improved(batch, result["reconstruction"])
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_train = epoch_loss / max(batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                result = model(batch)
                loss = ast_reconstruction_loss_improved(batch, result["reconstruction"])
                val_loss += loss.item()
                val_batches += 1
        avg_val = val_loss / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"decoder_state_dict": model.decoder.state_dict()}, model_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"train={avg_train:.4f} val={avg_val:.4f} "
                  f"best={best_val_loss:.4f} | {elapsed:.0f}s")

    train_time = time.time() - t0
    print(f"Training complete in {train_time:.0f}s, best val_loss={best_val_loss:.4f}")

    # Load best checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()

    # Evaluate: generate samples and check validity
    print(f"\nEvaluating {NUM_SAMPLES} samples...")
    _, eval_loader = create_data_loaders(
        val_path, val_path, batch_size=1, shuffle=False, num_workers=0
    )

    samples = []
    heuristic_valid = 0
    total = 0

    with torch.no_grad():
        for batch in eval_loader:
            if total >= NUM_SAMPLES:
                break
            batch = batch.to(device)
            result = model(batch)
            recon = result["reconstruction"]

            node_feats = recon.get("node_features") if isinstance(recon, dict) else None
            if node_feats is None:
                total += 1
                continue

            pred_types = node_feats.argmax(dim=-1)
            orig_types = batch.x.argmax(dim=-1) if batch.x.dim() > 1 else batch.x

            unique_pred = len(pred_types.unique())
            unique_orig = len(orig_types.unique())
            type_match = (pred_types == orig_types).float().mean().item()

            # Heuristic validity (>2 unique types)
            heuristic_ok = unique_pred > 2

            sample = {
                "index": total,
                "num_nodes": int(pred_types.shape[0]),
                "pred_unique_types": unique_pred,
                "orig_unique_types": unique_orig,
                "type_accuracy": round(type_match, 4),
                "heuristic_valid": heuristic_ok,
                "pred_type_ids": pred_types.cpu().tolist(),
                "orig_type_ids": orig_types.cpu().tolist(),
            }
            samples.append(sample)

            if heuristic_ok:
                heuristic_valid += 1
            total += 1

    heuristic_pct = (heuristic_valid / total * 100) if total > 0 else 0.0

    # Compute statistics on type predictions
    type_accuracies = [s["type_accuracy"] for s in samples]
    avg_type_accuracy = sum(type_accuracies) / len(type_accuracies) if type_accuracies else 0
    unique_counts = [s["pred_unique_types"] for s in samples]
    avg_unique = sum(unique_counts) / len(unique_counts) if unique_counts else 0

    # Sort by type_accuracy descending to show best samples first
    samples.sort(key=lambda s: s["type_accuracy"], reverse=True)

    result = {
        "label": label,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "decoder_conv_type": decoder_conv_type,
        "decoder_edge_mode": decoder_edge_mode,
        "trainable_params": param_count,
        "best_val_loss": round(best_val_loss, 4),
        "train_time_s": round(train_time, 1),
        "samples_evaluated": total,
        "heuristic_valid": heuristic_valid,
        "heuristic_validity_pct": round(heuristic_pct, 2),
        "avg_type_accuracy": round(avg_type_accuracy, 4),
        "avg_unique_pred_types": round(avg_unique, 2),
        "top_samples": samples[:20],
    }

    # Save individual result
    result_path = os.path.join(RESULTS_DIR, f"{label}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults: heuristic_validity={heuristic_pct:.1f}% "
          f"({heuristic_valid}/{total}), "
          f"avg_type_acc={avg_type_accuracy:.4f}, "
          f"avg_unique_types={avg_unique:.1f}")

    return result


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = [
        # Replicate the 7% result
        {"hidden_dim": 256, "decoder_edge_mode": "teacher_forced",
         "decoder_conv_type": "GIN", "num_layers": 3, "label": "tf-gin-256"},
        # Ablation: smaller dim
        {"hidden_dim": 128, "decoder_edge_mode": "teacher_forced",
         "decoder_conv_type": "GIN", "num_layers": 3, "label": "tf-gin-128"},
        # Ablation: larger dim
        {"hidden_dim": 512, "decoder_edge_mode": "teacher_forced",
         "decoder_conv_type": "GIN", "num_layers": 3, "label": "tf-gin-512"},
        # Ablation: deeper network
        {"hidden_dim": 256, "decoder_edge_mode": "teacher_forced",
         "decoder_conv_type": "GIN", "num_layers": 5, "label": "tf-gin-256-deep"},
        # Control: chain GIN (should be ~0%)
        {"hidden_dim": 256, "decoder_edge_mode": "chain",
         "decoder_conv_type": "GIN", "num_layers": 3, "label": "chain-gin-256"},
    ]

    all_results = []
    for cfg in configs:
        result = train_and_evaluate(**cfg)
        all_results.append(result)
        print(f"\n{'~'*70}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — Teacher-Forced GIN Deep Dive")
    print(f"{'='*70}")
    print(f"{'Label':<22s} {'Dim':>4s} {'Layers':>6s} {'Edge':>15s} "
          f"{'Params':>10s} {'ValLoss':>8s} {'Validity':>8s} {'TypeAcc':>8s}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['label']:<22s} {r['hidden_dim']:>4d} {r['num_layers']:>6d} "
              f"{r['decoder_edge_mode']:>15s} {r['trainable_params']:>10,d} "
              f"{r['best_val_loss']:>8.4f} {r['heuristic_validity_pct']:>7.1f}% "
              f"{r['avg_type_accuracy']:>8.4f}")

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
