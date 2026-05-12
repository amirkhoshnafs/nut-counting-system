"""Evaluate trained YOLO-NAS on the combined test split using pycocotools.

Nut-only system — class 0 is nut. Usage:
    conda run -n ml2 python scripts/evaluate.py --model yolo_nas_s --conf 0.50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from super_gradients.training import models
from tqdm import tqdm

import super_gradients

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed/combined"


def find_best_checkpoint() -> Path:
    ckpt_dir = PROJECT_ROOT / "checkpoints/nut_counting_phase1"
    for p in ckpt_dir.rglob("ckpt_best.pth"):
        return p
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default="yolo_nas_m", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.01)
    args = parser.parse_args()

    super_gradients.setup_device(device=args.device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint()
    print(f"Checkpoint: {ckpt_path}")
    print(f"Model:     {args.model}")

    # Load model
    model = models.get(
        args.model, pretrained_weights=None,
        num_classes=2, checkpoint_path=str(ckpt_path),
    )
    model.eval()

    # Load COCO ground truth
    gt_path = DATA_DIR / "annotations/test.json"
    coco_gt = COCO(str(gt_path))
    cat_ids = coco_gt.getCatIds()

    # Load images from test split
    test_img_dir = DATA_DIR / "images/test"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_predictions = []
    img_id_to_coco = {img["file_name"]: img["id"] for img in gt_data["images"]}

    image_files = sorted(test_img_dir.glob("*.jpg"))
    print(f"\nRunning inference on {len(image_files)} test images ...")

    for img_path in tqdm(image_files, desc="Inference"):
        fname = img_path.name
        coco_img_id = img_id_to_coco.get(fname)
        if coco_img_id is None:
            continue

        result = model.predict(
            str(img_path),
            conf=args.conf,
            iou=0.65,
            fuse_model=False,
            fp16=True,
        )

        # result is ImageDetectionPrediction
        pred = result.prediction
        boxes = pred.bboxes_xyxy if hasattr(pred, 'bboxes_xyxy') else pred.bboxes
        scores = pred.confidence if hasattr(pred, 'confidence') else np.array([])
        labels = pred.labels if hasattr(pred, 'labels') else np.array([])

        if boxes is None or len(boxes) == 0:
            continue

        for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                all_predictions.append({
                    "image_id": int(coco_img_id),
                    "category_id": int(label),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(w), 2), round(float(h), 2)],
                    "score": round(float(score), 4),
                })

    print(f"  Generated {len(all_predictions)} predictions")

    if len(all_predictions) == 0:
        print("ERROR: No predictions generated. Check model and data.")
        return

    # Save predictions
    pred_path = PROJECT_ROOT / "reports/test_predictions.json"
    pred_path.parent.mkdir(exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)

    # Evaluate with pycocotools
    coco_dt = coco_gt.loadRes(str(pred_path))

    print("\n" + "=" * 50)
    print("Test Set Results (pycocotools)")
    print("=" * 50)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP@0.50:0.95": round(float(coco_eval.stats[0]), 4),
        "mAP@0.50":      round(float(coco_eval.stats[1]), 4),
        "mAP@0.75":      round(float(coco_eval.stats[2]), 4),
        "AR@1":          round(float(coco_eval.stats[6]), 4),
        "AR@10":         round(float(coco_eval.stats[7]), 4),
        "AR@100":        round(float(coco_eval.stats[8]), 4),
    }

    print(f"\n  mAP@0.50:      {metrics['mAP@0.50']}")
    print(f"  mAP@0.50:0.95: {metrics['mAP@0.50:0.95']}")
    print(f"  mAP@0.75:      {metrics['mAP@0.75']}")

    passed = metrics["mAP@0.50"] >= 0.85
    print(f"\n  Milestone gate (mAP@50 >= 0.85): {'PASS' if passed else 'NOT YET'}")

    report_path = PROJECT_ROOT / "reports/phase1_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to: {report_path}")


if __name__ == "__main__":
    main()
