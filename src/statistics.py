"""Batch statistics — Wilson confidence intervals and batch summary reports."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

import numpy as np


def wilson_score_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes.
        trials: Total number of trials.
        z: Z-score (1.96 for 95% CI).

    Returns:
        (lower_bound, upper_bound) as (0-1 range, 0-1 range).
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_count_uncertainty(
    raw_count: int,
    test_recall: float = 0.865,
    test_fpr: float = 0.05,
    test_n: int = 29487,
) -> tuple[int, int, int]:
    """Estimate count uncertainty from detection performance on test set.

    Args:
        raw_count: Raw count from the pipeline.
        test_recall: Measured recall on test set (e.g., 0.865).
        test_fpr: Measured false positive rate on test set.
        test_n: Number of predictions in test evaluation.

    Returns:
        (expected_count, lower_ci, upper_ci).
    """
    # Adjust for missed detections and false positives
    expected = raw_count / test_recall - test_fpr * raw_count
    expected = max(0, int(round(expected)))

    # Simple error propagation
    sigma = math.sqrt(raw_count * (1 - test_recall) + test_fpr * raw_count)
    lower = max(0, int(expected - 1.96 * sigma))
    upper = int(expected + 1.96 * sigma)

    return expected, lower, upper


class BatchReporter:
    """Generates batch summary reports with calibrated confidence and uncertainty."""

    def __init__(self, test_recall: float = 0.865, test_fpr: float = 0.05):
        self._start_time = datetime.now()
        self._counts: list[dict] = []
        self._recall = test_recall
        self._fpr = test_fpr

    def add_count(self, count_event: dict) -> None:
        self._counts.append(count_event)

    def summary(self) -> str:
        """Generate a formatted batch summary report."""
        total = len(self._counts)
        if total == 0:
            return "BATCH: No counts recorded."

        per_class = defaultdict(int)
        high = medium = low = 0
        confidences = []

        for c in self._counts:
            cls_name = str(c.get("class", 0))
            per_class[cls_name] += 1
            confidences.append(c.get("confidence", 0.0))
            label = c.get("reliability", "LOW")
            if label == "HIGH":
                high += 1
            elif label == "MEDIUM":
                medium += 1
            else:
                low += 1

        mean_conf = np.mean(confidences) if confidences else 0.0

        # Count uncertainty
        expected, lower, upper = compute_count_uncertainty(
            total, self._recall, self._fpr,
        )

        lines = [
            "=" * 42,
            f" BATCH REPORT  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 42,
        ]
        for cls_name, count in sorted(per_class.items()):
            name = "nut" if cls_name == "0" else f"class_{cls_name}"
            e, l, u = compute_count_uncertainty(count, self._recall, self._fpr)
            ci = f"[{l}, {u}]"
            lines.append(f"  {name:10s}  {count:5d}  ±{e-count:3d}   (95% CI: {ci})")
        lines.append("-" * 42)
        lines.append(f"  {'TOTAL':10s}  {total:5d}  ±{expected-total:3d}   (95% CI: [{lower}, {upper}])")
        lines.append("-" * 42)

        reliability = "HIGH" if low == 0 and medium / max(total, 1) < 0.05 else (
            "MEDIUM" if low / max(total, 1) < 0.05 else "LOW"
        )
        lines.append(f"  RELIABILITY: {reliability}  ({high} high, {medium} med, {low} low)")
        lines.append(f"  MEAN CONF:   {mean_conf:.4f}")
        lines.append("=" * 42)

        return "\n".join(lines)
