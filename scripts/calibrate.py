"""Confidence calibration for YOLO-NAS nut detector.

1. Run inference on calibration split, collect (confidence, is_correct) pairs
2. Fit isotonic regression calibrator
3. Plot reliability diagram (before/after)
4. Compute ECE and Brier score on test split

Usage:
    conda run -n ml2 python scripts/calibrate.py
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from super_gradients.training import models
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from torch.utils.data import DataLoader
from super_gradients.training.utils.collate_fn import CrowdDetectionCollateFN
from tqdm import tqdm

import super_gradients

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed/combined"


def find_best_checkpoint() -> Path:
    ckpt_dir = PROJECT_ROOT / "checkpoints/nut_counting_phase1"
    for p in ckpt_dir.rglob("ckpt_best.pth"):
        return p
    raise FileNotFoundError("No checkpoint found")


def collect_predictions(model, loader, device) -> tuple[list, list]:
    """Run model on a split, return (confidences, is_correct) arrays."""
    confidences: list[float] = []
    labels: list[int] = []  # 1 = correct (> 0.5 IoU), 0 = incorrect

    model.eval()
    with tqdm(loader, desc="Collecting") as pbar:
        for images, targets, crowd_targets in loader:
            images = images.to(device)
            with __import__("torch").no_grad():
                preds = model(images)

            # preds is raw output tuple — decode with post-processing
            if isinstance(preds, tuple):
                # Process each image
                for i, img_targets in enumerate(targets):
                    if isinstance(img_targets, __import__("torch").Tensor) and img_targets.shape[0] > 0:
                        gt_boxes = img_targets.cpu().numpy()[:, :4]  # ignoring batch_idx and class_id
                    else:
                        gt_boxes = __import__("numpy").array([])

                    # Get pred boxes from model output via predict on a dummy
                    # Actually, just use model.predict on the image
                    # We'll collect at the image level below
                    pass

            pbar.update(1)

    return confidences, labels


def collect_via_predict(model, split: str, device: str) -> tuple[list, list]:
    """Collect (confidence, is_correct) pairs using model.predict() on each image."""
    import json as _json
    import cv2

    # Load COCO ground truth
    gt_path = DATA_DIR / "annotations" / f"{split}.json"
    with open(gt_path) as f:
        gt_data = _json.load(f)

    img_dir = DATA_DIR / "images" / split
    confidences: list[float] = []
    labels: list[int] = []

    image_files = sorted(img_dir.glob("*.jpg"))
    gt_by_file = {img["file_name"]: img for img in gt_data["images"]}
    ann_by_image = {img["id"]: [] for img in gt_data["images"]}
    for ann in gt_data["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    for img_path in tqdm(image_files, desc=f"Calibrating on {split}"):
        fname = img_path.name
        gt_img = gt_by_file.get(fname)
        if gt_img is None:
            continue
        gt_boxes = [
            ann["bbox"] for ann in ann_by_image[gt_img["id"]]
        ]  # COCO bbox: [x, y, w, h]

        # Run prediction
        result = model.predict(str(img_path), conf=0.01, iou=0.65,
                               fuse_model=False, fp16=(device == "cuda"))
        pred = result.prediction
        pred_boxes = pred.bboxes_xyxy
        pred_scores = pred.confidence

        if pred_boxes is None or len(pred_boxes) == 0:
            continue

        # Match predictions to ground truth
        for pred_box, score in zip(pred_boxes, pred_scores):
            # Convert pred [x1,y1,x2,y2] to COCO [x,y,w,h]
            px, py = pred_box[0], pred_box[1]
            pw, ph = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]

            best_iou = 0.0
            for gt_bbox in gt_boxes:
                gx, gy, gw, gh = gt_bbox
                iou = box_iou([px, py, pw, ph], [gx, gy, gw, gh])
                best_iou = max(best_iou, iou)

            confidences.append(float(score))
            labels.append(1 if best_iou >= 0.5 else 0)

    return confidences, labels


def box_iou(a, b):
    """COCO format IoU: [x, y, w, h]."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


def compute_ece(confidences, labels, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(confidences, labels):
    """Brier score — mean squared error between confidence and correctness."""
    return float(np.mean((np.array(confidences) - np.array(labels)) ** 2))


def plot_reliability(confidences, labels, cal_confidences, cal_labels, save_path):
    """Plot reliability diagram: before vs after calibration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, title, c, l in [
        (ax1, "Raw Model", confidences, labels),
        (ax2, "After Isotonic Calibration", cal_confidences, cal_labels),
    ]:
        bins = np.linspace(0, 1, 11)
        bin_acc = []
        bin_conf = []
        for i in range(10):
            mask = (np.array(c) >= bins[i]) & (np.array(c) < bins[i + 1])
            if mask.sum() > 0:
                bin_conf.append(np.array(c)[mask].mean())
                bin_acc.append(np.array(l)[mask].mean())
            else:
                bin_conf.append((bins[i] + bins[i + 1]) / 2)
                bin_acc.append(0)

        ax.bar(bins[:-1] + 0.05, bin_acc, width=0.08, alpha=0.6, label="Actual Accuracy")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Reliability diagram saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="yolo_nas_s")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    super_gradients.setup_device(device=args.device)

    ckpt_path = find_best_checkpoint()
    print(f"Checkpoint: {ckpt_path}")

    model = models.get(
        args.model_name, pretrained_weights=None, num_classes=2,
        checkpoint_path=str(ckpt_path),
    )

    # ── 1. Collect on calibration split ──────────────
    print("\n[1] Collecting predictions on calibration split ...")
    cal_conf, cal_labels = collect_via_predict(model, "calibration", args.device)
    cal_conf = np.array(cal_conf)
    cal_labels = np.array(cal_labels)
    print(f"  {len(cal_conf)} predictions, accuracy={cal_labels.mean():.4f}")

    # Save calibration data
    cal_data_path = PROJECT_ROOT / "data" / "calibration" / "calibration_data.json"
    cal_data_path.parent.mkdir(exist_ok=True)
    with open(cal_data_path, "w") as f:
        json.dump({"confidences": cal_conf.tolist(), "labels": cal_labels.tolist()}, f)

    # ── 2. Fit isotonic regression ───────────────────
    print("\n[2] Fitting isotonic regression ...")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(cal_conf, cal_labels)

    calibrator_path = PROJECT_ROOT / "checkpoints" / "calibrator.pkl"
    with open(calibrator_path, "wb") as f:
        pickle.dump(iso, f)
    print(f"  Calibrator saved to: {calibrator_path}")

    # ── 3. Evaluate on test split ────────────────────
    print("\n[3] Evaluating on test split ...")
    test_conf, test_labels_raw = collect_via_predict(model, "test", args.device)
    test_conf = np.array(test_conf)
    test_labels = np.array(test_labels_raw)
    print(f"  {len(test_conf)} predictions, accuracy={test_labels.mean():.4f}")

    # Apply calibration
    test_conf_cal = iso.predict(test_conf)

    # ── 4. Compute metrics ───────────────────────────
    print("\n[4] Computing calibration metrics ...")
    ece_raw = compute_ece(test_conf, test_labels)
    ece_cal = compute_ece(test_conf_cal, test_labels)
    brier_raw = compute_brier(test_conf, test_labels)
    brier_cal = compute_brier(test_conf_cal, test_labels)

    print(f"  ECE (raw):        {ece_raw:.4f}")
    print(f"  ECE (calibrated): {ece_cal:.4f}")
    print(f"  Brier (raw):      {brier_raw:.4f}")
    print(f"  Brier (cal):      {brier_cal:.4f}")

    # ── 5. Plot ──────────────────────────────────────
    print("\n[5] Plotting reliability diagram ...")
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    plot_reliability(test_conf, test_labels, test_conf_cal, test_labels,
                     report_dir / "reliability_diagram.png")

    # ── Save metrics ─────────────────────────────────
    metrics = {
        "ece_raw": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "brier_raw": round(brier_raw, 4),
        "brier_calibrated": round(brier_cal, 4),
        "ece_gate_passed": ece_cal < 0.05,
        "num_predictions_test": len(test_conf),
    }
    with open(report_dir / "calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'PASS' if metrics['ece_gate_passed'] else 'NOT YET'}: ECE < 0.05")
    print(f"Metrics saved to: {report_dir / 'calibration_metrics.json'}")


if __name__ == "__main__":
    main()
