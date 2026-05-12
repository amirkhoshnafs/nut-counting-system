"""Confidence calibrator — maps raw YOLO-NAS confidences to calibrated probabilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class Calibrator:
    """Wraps an isotonic regression model for confidence calibration."""

    def __init__(self, calibrator_path: str | Path):
        with open(calibrator_path, "rb") as f:
            self._model = pickle.load(f)

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Map raw confidences to calibrated probabilities."""
        return self._model.predict(confidences)

    def calibrate_single(self, confidence: float) -> float:
        """Calibrate a single confidence value."""
        return float(self._model.predict(np.array([confidence]))[0])

    def reliability_label(self, calibrated_prob: float) -> str:
        """Classify a calibrated probability into HIGH/MEDIUM/LOW."""
        if calibrated_prob >= 0.90:
            return "HIGH"
        elif calibrated_prob >= 0.70:
            return "MEDIUM"
        else:
            return "LOW"

    def needs_review(self, calibrated_prob: float) -> bool:
        """Whether this count should be flagged for manual review."""
        return calibrated_prob < 0.70
