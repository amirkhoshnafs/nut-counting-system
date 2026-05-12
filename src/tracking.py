"""ByteTrack wrapper for nut tracking.

Uses the supervision library's built-in ByteTrack implementation.
Returns tracks with persistent IDs and accumulated statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from src.config import TrackingConfig


@dataclass
class Track:
    """A tracked object with accumulated history."""

    track_id: int
    class_id: int = 0  # always 0 (nut) in current system
    age: int = 0
    state: str = "PENDING"  # PENDING | CONFIRMED | COUNTED | STALE | LOST

    # Accumulated history
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    box_areas: List[float] = field(default_factory=list)
    frames_since_movement: int = 0
    is_counted: bool = False
    count_id: int | None = None  # sequential 1, 2, 3... when counted

    @property
    def latest_centroid(self) -> Optional[Tuple[float, float]]:
        return self.centroids[-1] if self.centroids else None

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    @property
    def mean_box_area(self) -> float:
        return float(np.mean(self.box_areas)) if self.box_areas else 0.0


class ByteTrackWrapper:
    """Wraps supervision's ByteTrack for our tracking + counting pipeline."""

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._tracks: dict[int, Track] = {}  # track_id → Track
        self._next_id = 0
        self._lost_ids: set[int] = set()

        # Lazily import supervision to avoid import errors if not installed
        import supervision as sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=cfg.track_buffer,
            minimum_matching_threshold=cfg.iou_threshold,
            minimum_consecutive_frames=cfg.min_hits,
            frame_rate=30,
        )

    def update(self, detections: List[Tuple[np.ndarray, float]]) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of (bbox_xyxy, confidence) tuples.

        Returns:
            List of active Track objects with updated state.
        """
        if not detections:
            # No detections — age all existing tracks
            for track in list(self._tracks.values()):
                track.age += 1
                track.frames_since_movement += 1
            self._prune_lost_tracks()
            return list(self._tracks.values())

        # Convert to supervision Detections format
        boxes = np.array([d[0] for d in detections])
        scores = np.array([d[1] for d in detections])

        sv_detections = self._to_sv_detections(boxes, scores)

        # Run ByteTrack — returns sv.Detections with tracker_id
        tracked = self._tracker.update_with_detections(sv_detections)

        # Extract arrays from Detections object
        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            # No tracks — age all existing and return
            for track in list(self._tracks.values()):
                track.age += 1
                track.frames_since_movement += 1
            self._prune_lost_tracks()
            return list(self._tracks.values())

        tracker_ids = tracked.tracker_id  # numpy array of ints
        bboxes = tracked.xyxy               # numpy array [N, 4]
        scores = tracked.confidence if hasattr(tracked, 'confidence') else np.zeros(len(tracker_ids))

        # Update our Track objects
        active_ids = set()
        for i in range(len(tracker_ids)):
            track_id = int(tracker_ids[i])
            if track_id is None:
                continue
            active_ids.add(track_id)

            bbox = bboxes[i]
            score = float(scores[i]) if i < len(scores) else 0.0
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if track_id not in self._tracks:
                self._tracks[track_id] = Track(track_id=track_id)
            track = self._tracks[track_id]

            track.centroids.append(centroid)
            track.confidences.append(score)
            track.box_areas.append(area)
            track.age += 1
            track.frames_since_movement = 0

            if track.state == "PENDING" and track.age >= self.cfg.min_hits:
                track.state = "CONFIRMED"

        # Mark tracks that weren't seen this frame as potentially lost
        for track_id, track in list(self._tracks.items()):
            if track_id not in active_ids:
                track.frames_since_movement += 1

        self._prune_lost_tracks()
        return list(self._tracks.values())

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def mark_counted(self, track_id: int) -> None:
        track = self._tracks.get(track_id)
        if track:
            track.state = "COUNTED"
            track.is_counted = True

    def mark_stale(self, track_id: int) -> None:
        track = self._tracks.get(track_id)
        if track:
            track.state = "STALE"

    def _prune_lost_tracks(self) -> None:
        """Remove tracks that have been unseen for > track_buffer frames."""
        to_remove = []
        for track_id, track in self._tracks.items():
            if track.frames_since_movement > self.cfg.track_buffer:
                if track.state not in ("COUNTED", "STALE"):
                    track.state = "LOST"
                if not track.is_counted:
                    to_remove.append(track_id)

        for track_id in to_remove:
            self._lost_ids.add(track_id)
            del self._tracks[track_id]

    @staticmethod
    def _to_sv_detections(boxes: np.ndarray, scores: np.ndarray):
        import supervision as sv
        n = len(boxes)
        return sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=np.zeros(n, dtype=int),
            tracker_id=None,
        )
