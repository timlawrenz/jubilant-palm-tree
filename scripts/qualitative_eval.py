#!/usr/bin/env python3
"""Qualitative analysis: reconstruct Ruby code from teacher-forced GIN and check syntax.

Uses the full pipeline: model → predict types + parents → build AST tree → Ruby pretty-print → syntax check.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import ASTGraphConverter, ASTNodeEncoder, create_data_loaders
from models import ASTAutoencoder
from torch_geometric.data import Data

RESULTS_DIR = "results/gin_deep_dive"
DATASET_PATH = "dataset"
ENCODER_WEIGHTS = "models/best_model.pt"
NUM_SAMPLES = 200


def run_ruby_script(script_path: str, stdin_data: str) -> str:
    """Run a Ruby script with stdin and return stdout."""
    try:
        result = subprocess.run(
            ["ruby", script_path],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def check_ruby_syntax(code: str) -> bool:
    """Check if code is valid Ruby."""
    if not code or not code.strip():
        return False
    try:
        result = subprocess.run(
            ["ruby", "scripts/check_syntax.rb"],
            input=code,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def evaluate_model_full(model_path: str, label: str, hidden_dim: int,
                        num_layers: int, decoder_conv_type: str,
                        decoder_edge_mode: str) -> dict:
    """Full evaluation: model → AST → Ruby code → syntax check."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    converter = ASTGraphConverter()

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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()

    val_path = os.path.join(DATASET_PATH, "val.jsonl")
    _, val_loader = create_data_loaders(
        val_path, val_path, batch_size=1, shuffle=False, num_workers=0
    )

    # Also load raw source for comparison
    raw_sources = []
    with open(val_path) as f:
        for line in f:
            d = json.loads(line)
            raw_sources.append(d["raw_source"])

    results = []
    syntax_valid = 0
    heuristic_valid = 0
    total = 0

    print(f"\nEvaluating {label} on {NUM_SAMPLES} samples...")

    with torch.no_grad():
        for batch in val_loader:
            if total >= NUM_SAMPLES:
                break

            batch = batch.to(device)
            result = model(batch)
            recon = result["reconstruction"]

            node_feats = recon.get("node_features") if isinstance(recon, dict) else None
            parent_logits = recon.get("parent_logits") if isinstance(recon, dict) else None

            if node_feats is None:
                total += 1
                continue

            pred_types = node_feats.argmax(dim=-1)
            orig_types = batch.x.argmax(dim=-1) if batch.x.dim() > 1 else batch.x

            unique_pred = len(pred_types.unique())
            type_match = (pred_types == orig_types).float().mean().item()
            heuristic_ok = unique_pred > 2

            # Build reconstructed AST tree from types + parents
            reconstructed_code = ""
            if parent_logits is not None:
                parent_preds = parent_logits.squeeze(0).argmax(dim=-1)
                node_map = {}
                for i, type_idx in enumerate(pred_types.cpu().numpy()):
                    nt = converter.node_encoder.node_types[type_idx] \
                        if type_idx < len(converter.node_encoder.node_types) else "unknown"
                    node_map[i] = {"type": nt, "children": []}

                root_nodes = []
                for i, parent_idx in enumerate(parent_preds.cpu().numpy()):
                    pi = int(parent_idx)
                    if i == pi or pi >= len(node_map):
                        root_nodes.append(node_map[i])
                    elif pi in node_map:
                        node_map[pi]["children"].append(node_map[i])

                if root_nodes:
                    ast_json = json.dumps(root_nodes)
                    reconstructed_code = run_ruby_script(
                        "scripts/pretty_print_ast.rb", ast_json
                    )

            is_valid = check_ruby_syntax(reconstructed_code)

            sample = {
                "index": total,
                "num_nodes": int(pred_types.shape[0]),
                "type_accuracy": round(type_match, 4),
                "heuristic_valid": heuristic_ok,
                "syntax_valid": is_valid,
                "original_code": raw_sources[total] if total < len(raw_sources) else "",
                "reconstructed_code": reconstructed_code,
                "pred_types": [converter.node_encoder.node_types[t]
                               if t < len(converter.node_encoder.node_types) else "unknown"
                               for t in pred_types.cpu().tolist()],
            }
            results.append(sample)

            if heuristic_ok:
                heuristic_valid += 1
            if is_valid:
                syntax_valid += 1
            total += 1

            if total % 50 == 0:
                print(f"  {total}/{NUM_SAMPLES} done, "
                      f"syntax_valid={syntax_valid}, heuristic_valid={heuristic_valid}")

    heuristic_pct = (heuristic_valid / total * 100) if total > 0 else 0
    syntax_pct = (syntax_valid / total * 100) if total > 0 else 0

    print(f"\n{label}: heuristic={heuristic_pct:.1f}%, syntax={syntax_pct:.1f}% "
          f"({syntax_valid}/{total})")

    # Show examples of valid reconstructions
    valid_examples = [r for r in results if r["syntax_valid"]]
    print(f"\n--- Valid reconstructions ({len(valid_examples)} total) ---")
    for r in valid_examples[:5]:
        print(f"\nSample #{r['index']} ({r['num_nodes']} nodes, type_acc={r['type_accuracy']}):")
        print(f"  ORIGINAL:      {r['original_code'][:80].strip()}")
        print(f"  RECONSTRUCTED: {r['reconstructed_code'][:80].strip()}")

    # Show best type-accuracy samples that failed syntax
    failed_high_acc = sorted(
        [r for r in results if not r["syntax_valid"] and r["type_accuracy"] > 0.9],
        key=lambda x: -x["type_accuracy"],
    )
    if failed_high_acc:
        print(f"\n--- High accuracy but invalid syntax ({len(failed_high_acc)} total) ---")
        for r in failed_high_acc[:3]:
            print(f"\nSample #{r['index']} (type_acc={r['type_accuracy']}):")
            print(f"  ORIGINAL:      {r['original_code'][:80].strip()}")
            print(f"  RECONSTRUCTED: {r['reconstructed_code'][:80].strip()}")
            print(f"  PRED TYPES: {r['pred_types'][:12]}")

    output = {
        "label": label,
        "total": total,
        "heuristic_valid": heuristic_valid,
        "heuristic_pct": round(heuristic_pct, 2),
        "syntax_valid": syntax_valid,
        "syntax_pct": round(syntax_pct, 2),
        "valid_examples": valid_examples[:10],
        "failed_high_acc": [
            {k: v for k, v in r.items() if k != "pred_types"}
            for r in failed_high_acc[:10]
        ],
    }

    out_path = os.path.join(RESULTS_DIR, f"{label}_qualitative.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    return output


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = [
        {
            "model_path": os.path.join(RESULTS_DIR, "tf-gin-256-deep_decoder.pt"),
            "label": "tf-gin-256-deep",
            "hidden_dim": 256, "num_layers": 5,
            "decoder_conv_type": "GIN", "decoder_edge_mode": "teacher_forced",
        },
        {
            "model_path": os.path.join(RESULTS_DIR, "chain-gin-256_decoder.pt"),
            "label": "chain-gin-256",
            "hidden_dim": 256, "num_layers": 3,
            "decoder_conv_type": "GIN", "decoder_edge_mode": "chain",
        },
    ]

    all_results = []
    for cfg in configs:
        if not os.path.exists(cfg["model_path"]):
            print(f"Skipping {cfg['label']}: no model at {cfg['model_path']}")
            continue
        result = evaluate_model_full(**cfg)
        all_results.append(result)

    print("\n" + "=" * 60)
    print("FULL RECONSTRUCTION SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['label']:25s}: heuristic={r['heuristic_pct']:5.1f}%  "
              f"syntax={r['syntax_pct']:5.1f}% ({r['syntax_valid']}/{r['total']})")


if __name__ == "__main__":
    main()
