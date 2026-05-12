"""CSV logger for counting events and batch reports."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class CSVLogger:
    """Logs per-count events to a CSV file."""

    COLUMNS = [
        "timestamp", "track_id", "class", "counted",
        "confidence", "calibrated_confidence", "reliability",
        "age", "reason", "centroid_x", "centroid_y",
    ]

    def __init__(self, csv_path: str):
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()

    def log(self, event: dict) -> None:
        """Write a counting event to CSV."""
        centroid = event.get("centroid", (0, 0))
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "track_id": event.get("track_id", -1),
            "class": event.get("class", 0),
            "counted": event.get("counted", False),
            "confidence": round(event.get("confidence", 0.0), 4),
            "calibrated_confidence": round(event.get("calibrated_confidence", event.get("confidence", 0.0)), 4),
            "reliability": event.get("reliability", "UNKNOWN"),
            "age": event.get("age", 0),
            "reason": event.get("reason", ""),
            "centroid_x": round(centroid[0], 1),
            "centroid_y": round(centroid[1], 1),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
