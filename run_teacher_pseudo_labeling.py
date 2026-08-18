#!/usr/bin/env python3
"""
Task 4 Runner: Teacher Pseudo-Labeling Engine.

Processes candidate persona portfolios from crawlr:
1. Runs BT-Net binary model (Jul 22 checkpoint, val_acc = 0.9477) + Sapiens Torso gating on all photos.
2. Identifies bare photos (where breast_pixels > 1000 and conf > 0.80).
3. Computes persona anchor score bar_V_k = mean(BT-Net_Conf_bare) for personas with >= 1 bare photo.
4. Assigns target label bar_V_k to all clothed photos in the persona portfolio.
5. Saves pseudo-labels to SQLite database data/teacher_pseudo_labels.db.
"""
import os
import sys
import json
import time
import cv2
import numpy as np
import torch

os.environ["MIOPEN_USER_DB_PATH"] = os.path.expanduser("~/.config/miopen")
os.environ["MIOPEN_FIND_MODE"] = "7"
os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HSA_ENABLE_SDMA"] = "0"

sys.path.insert(0, "/home/tim/source/activity/sapiens2_full")
sys.path.insert(0, "/home/tim/source/activity/volume-estimator")

from volume_estimator import load_segmentation_model
from src.breast_tissue_gate import BreastTissueGate
from src.crawlr_db import fetch_canonical_auraface_embeddings
from src.persona_clustering import cluster_auraface_embeddings
from src.teacher_labeler import (
    compute_persona_anchor_score,
    init_labels_db,
    save_persona_pseudo_label,
)

BTNET = "/home/tim/source/activity/BreastTissue-Net"
DB_LABELS = "/home/tim/source/activity/volume-estimator/data/teacher_pseudo_labels.db"
TIMESTAMP = time.strftime("%Y-%m-%d_%H%M")


def main():
    print("=" * 72)
    print("Teacher Pseudo-Labeling & Anchor Aggregation Engine")
    print("=" * 72)

    init_labels_db(DB_LABELS)

    # Step 1: Fetch photos & cluster personas
    print("\n[1/3] Fetching canonical photos & clustering personas...")
    records = fetch_canonical_auraface_embeddings(limit=3000)
    clusters = cluster_auraface_embeddings(records, distance_threshold=0.50)
    multi_clusters = [c for c in clusters if c["size"] >= 3]
    print(f"Loaded {len(records)} photos across {len(multi_clusters)} multi-photo personas.")

    # Step 2: Load models
    print("\n[2/3] Loading BreastTissue-Net (Jul 22 checkpoint) & Sapiens Seg...")
    gate = BreastTissueGate(
        checkpoint=f"{BTNET}/checkpoints/binary/best.pt",
        pretrained=f"{BTNET}/models/hf_pretrain_0.4b/model.safetensors",
        arch="sapiens2_0.4b",
        device="cuda:0",
    )
    seg_model = load_segmentation_model()

    # Step 3: Run Pseudo-Labeling Loop
    print("\n[3/3] Generating Teacher Pseudo-Labels for Personas with Bare Photos...")
    total_clothed_labeled = 0
    total_personas_labeled = 0

    for i, cluster in enumerate(multi_clusters):
        cid = cluster["cluster_id"]
        members = cluster["members"]

        photo_labels = []
        bare_confs = []

        for m in members:
            pid = m["photo_id"]
            img_path = m["image_path"]

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # BT-Net check
            bt = gate.check(img_path)
            raw_mask = bt["mask"]
            mean_conf = bt["mean_confidence"]

            # Sapiens Torso Seg
            with torch.no_grad():
                sd = seg_model.pipeline(dict(img=img))
                sd = seg_model.data_preprocessor(sd)
                inputs = sd["inputs"]
                seg_out = seg_model(inputs).argmax(dim=1).squeeze(0).cpu().numpy()

            seg_resized = cv2.resize(seg_out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            torso_mask = (seg_resized == 22)

            cleaned_mask = raw_mask & torso_mask
            breast_px_cleaned = int(np.sum(cleaned_mask))

            is_bare = bool(breast_px_cleaned > 1000 and mean_conf >= 0.80)

            if is_bare:
                bare_confs.append(mean_conf)

            photo_labels.append({
                "photo_id": pid,
                "is_bare": is_bare,
                "confidence": mean_conf,
                "breast_pixels": breast_px_cleaned,
            })

        # Only assign teacher pseudo-labels if persona contains >= 1 bare photo
        if len(bare_confs) >= 1:
            anchor_stats = compute_persona_anchor_score(bare_confs)
            anchor_val = anchor_stats["anchor_score"]
            cv_val = anchor_stats["cv"]
            std_val = anchor_stats["std"]

            save_persona_pseudo_label(
                db_path=DB_LABELS,
                persona_id=cid,
                anchor_score=anchor_val,
                cv=cv_val,
                photo_labels=photo_labels,
                std=std_val,
            )

            num_clothed = sum(1 for p in photo_labels if not p["is_bare"])
            total_clothed_labeled += num_clothed
            total_personas_labeled += 1

            print(
                f"  ✓ Persona #{cid:3d} (Size {len(members):2d}): "
                f"Bare={len(bare_confs):2d} (Anchor V_k = {anchor_val:.4f}) | "
                f"Clothed Targets Labeled = {num_clothed:2d}"
            )

        # Cap at 15 labeled personas for this initial training run
        if total_personas_labeled >= 15:
            break

    del seg_model
    del gate
    torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("TEACHER PSEUDO-LABELING COMPLETE")
    print("=" * 72)
    print(f"Generated pseudo-labels for {total_clothed_labeled} clothed photos across {total_personas_labeled} personas in SQLite DB: {DB_LABELS}")


if __name__ == "__main__":
    main()
