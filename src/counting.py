"""Counting zone: ROI polygon + counting line crossing.

Only objects inside the ROI polygon are tracked.  Counting happens
when a tracked object crosses the counting line from one side to the other.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import cv2

from src.config import CountingConfig, CONVEYOR_UNIT_VECTORS
from src.tracking import Track


class CountingZone:
    """ROI-gated counting zone with a single counting line."""

    def __init__(self, cfg: CountingConfig):
        self.cfg = cfg
        self._roi = np.array(cfg.roi_points, dtype=np.int32)
        self._count_line = tuple(
            (int(p[0]), int(p[1])) for p in cfg.count_line
        )  # ((x1,y1), (x2,y2))
        self._counted_ids: Set[int] = set()
        self._total_count: int = 0
        self._class_counts: Dict[int, int] = {0: 0}
        self._stale_track_ids: Set[int] = set()
        self._events: List[dict] = []

    # ── Public ───────────────────────────────────────

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def class_counts(self) -> Dict[int, int]:
        return dict(self._class_counts)

    @property
    def recent_events(self) -> List[dict]:
        return self._events[-20:]

    # ── Main update ──────────────────────────────────

    def update(self, tracks: List[Track], frame_shape: Tuple[int, int]) -> List[dict]:
        """Process tracks, check line crossing + gates, return newly counted."""
        newly_counted: List[dict] = []

        for track in tracks:
            if track.track_id in self._counted_ids:
                continue

            # Must be CONFIRMED
            if track.state != "CONFIRMED":
                continue

            # Must be inside ROI
            if not self._inside_roi(track.latest_centroid):
                continue

            # Need at least 2 positions for crossing check
            if len(track.centroids) < 2:
                continue

            prev = track.centroids[-2]
            curr = track.centroids[-1]

            # Check line crossing
            if not self._crossed_line(prev, curr):
                continue

            # Apply statistical gates
            passed, reason = self._apply_gates(track)

            if not passed:
                self._events.append({
                    "track_id": track.track_id, "class": track.class_id,
                    "counted": False, "reason": reason,
                    "confidence": track.mean_confidence,
                })
                continue

            # Count it — assign sequential count ID
            self._counted_ids.add(track.track_id)
            self._total_count += 1
            self._class_counts[track.class_id] = self._class_counts.get(track.class_id, 0) + 1
            track.state = "COUNTED"
            track.is_counted = True
            track.count_id = self._total_count

            event = {
                "track_id": track.track_id, "count_id": track.count_id,
                "class": track.class_id,
                "counted": True, "reason": "ok",
                "confidence": track.mean_confidence, "age": track.age,
                "centroid": curr,
            }
            self._events.append(event)
            newly_counted.append(event)

        return newly_counted

    # ── ROI check ────────────────────────────────────

    def _inside_roi(self, centroid: Optional[Tuple[float, float]]) -> bool:
        if centroid is None:
            return False
        if len(self._roi) == 0:  # no ROI — accept everything
            return True
        return cv2.pointPolygonTest(self._roi, (float(centroid[0]), float(centroid[1])), False) >= 0

    # ── Line crossing ────────────────────────────────

    def _crossed_line(self, prev: Tuple[float, float], curr: Tuple[float, float]) -> bool:
        """Check if prev→curr crosses the counting line in the configured direction."""
        (x1, y1), (x2, y2) = self._count_line
        side_prev = _line_side(prev, x1, y1, x2, y2)
        side_curr = _line_side(curr, x1, y1, x2, y2)

        # Basic crossing: sign changed
        if side_prev * side_curr >= 0:
            return False

        # Direction check for vertical line (x1 ≈ x2)
        # left_to_right: prev_x < line_x <= curr_x
        gate_x = (x1 + x2) / 2
        direction = getattr(self.cfg, 'count_direction', 'any')
        if direction == 'left_to_right':
            return prev[0] < gate_x <= curr[0]
        elif direction == 'right_to_left':
            return prev[0] > gate_x >= curr[0]
        return True  # 'any' direction

    # ── Statistical gates ────────────────────────────

    def _apply_gates(self, track: Track) -> Tuple[bool, str]:
        if track.age < self.cfg.min_track_age:
            return False, "too_short"
        if track.mean_confidence < self.cfg.min_mean_confidence:
            return False, "low_confidence"

        # Direction check
        if len(track.centroids) >= 4:
            dir_vec = self._estimate_direction(track.centroids[-10:])
            # Compare to expected direction (down-right for this conveyor)
            ref = np.array([1.0, -0.3])  # nuts move right and slightly up in frame
            cos_sim = _cosine_similarity(dir_vec, ref)
            if cos_sim < self.cfg.direction_cos_threshold:
                return False, "wrong_direction"

        # Speed
        if len(track.centroids) >= 4:
            speed = self._estimate_speed(track.centroids[-10:])
            if not (self.cfg.min_speed <= speed <= self.cfg.max_speed):
                return False, "abnormal_speed"

        # Box size
        mean_area = track.mean_box_area
        if not (self.cfg.min_area <= mean_area <= self.cfg.max_area):
            return False, "abnormal_size"

        return True, "ok"

    # ── Helpers ──────────────────────────────────────

    @staticmethod
    def _estimate_direction(centroids: List[Tuple]) -> np.ndarray:
        if len(centroids) < 2:
            return np.array([0.0, 1.0])
        pts = np.array(centroids[-10:])
        x = np.arange(len(pts))
        dx = np.polyfit(x, pts[:, 0], 1)[0]
        dy = np.polyfit(x, pts[:, 1], 1)[0]
        vec = np.array([dx, dy])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.array([0.0, 1.0])

    @staticmethod
    def _estimate_speed(centroids: List[Tuple]) -> float:
        if len(centroids) < 2:
            return 0.0
        pts = np.array(centroids[-10:])
        speeds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(np.mean(speeds))


# ── Geometry helpers ──────────────────────────────────

def _line_side(p: Tuple[float, float], x1: int, y1: int, x2: int, y2: int) -> float:
    """Cross product: >0 on one side, <0 on the other, 0 on the line."""
    x, y = p
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0
