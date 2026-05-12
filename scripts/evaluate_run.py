"""Compute counting metrics from a pipeline CSV log.

Compares counted events against ground truth annotations or manual counts.

Usage:
    conda run -n ml2 python scripts/evaluate_run.py --csv logs/counts.csv
    conda run -n ml2 python scripts/evaluate_run.py --csv logs/counts.csv --gt ground_truth.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


def load_csv(csv_path: str) -> list[dict]:
    """Load counting events from CSV."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["counted"] = row["counted"].lower() == "true"
            row["confidence"] = float(row["confidence"])
            row["track_id"] = int(row["track_id"])
            row["age"] = int(row["age"])
            rows.append(row)
    return rows


def compute_metrics(rows: list[dict], gt_counts: dict[str, int] | None = None) -> dict:
    """Compute counting metrics from logged events.

    Metrics:
        - total_counted: number of objects counted
        - rejected: number of crossing events rejected by gates
        - gate_reasons: breakdown of rejection reasons
        - id_switches: N/A without ground truth IDs
        - ghost_count: N/A without ground truth

    If gt_counts is provided ({"class_name": count}), also computes count error.
    """
    counted = [r for r in rows if r["counted"]]
    rejected = [r for r in rows if not r["counted"]]

    # Rejection reason breakdown
    gate_reasons: dict[str, int] = defaultdict(int)
    for r in rejected:
        gate_reasons[r.get("reason", "unknown")] += 1

    metrics: dict = {
        "total_counted": len(counted),
        "total_events": len(rows),
        "rejected": len(rejected),
        "rejection_rate": round(len(rejected) / len(rows), 4) if rows else 0.0,
        "gate_reasons": dict(gate_reasons),
        "mean_confidence_counted": round(
            sum(r["confidence"] for r in counted) / len(counted), 4
        ) if counted else 0.0,
        "per_class_counts": {},
    }

    # Per-class counts
    for r in counted:
        cls = r.get("class", "0")
        metrics["per_class_counts"][cls] = metrics["per_class_counts"].get(cls, 0) + 1

    # Compare with ground truth if provided
    if gt_counts:
        for cls_name, gt_count in gt_counts.items():
            pred_count = metrics["per_class_counts"].get(cls_name, 0)
            error = abs(pred_count - gt_count)
            error_rate = error / gt_count if gt_count > 0 else 0.0
            metrics[f"{cls_name}_gt"] = gt_count
            metrics[f"{cls_name}_pred"] = pred_count
            metrics[f"{cls_name}_error"] = error
            metrics[f"{cls_name}_error_rate"] = round(error_rate, 4)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to pipeline CSV output")
    parser.add_argument("--gt", default=None, help="Optional ground truth JSON: {'nut': 100, ...}")
    args = parser.parse_args()

    rows = load_csv(args.csv)

    gt_counts = None
    if args.gt:
        with open(args.gt) as f:
            gt_counts = json.load(f)

    metrics = compute_metrics(rows, gt_counts)

    print("=" * 50)
    print("Counting Metrics")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
