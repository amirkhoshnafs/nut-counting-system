"""Production alerts: jam detection and gap monitoring."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Dict, Optional


class AlertManager:
    """Monitors counting pipeline for anomalies and raises alerts."""

    def __init__(self, jam_threshold: int = 3, gap_timeout: float = 10.0,
                 stale_frames: int = 30, movement_thresh: float = 5.0):
        self.jam_threshold = jam_threshold
        self.gap_timeout = gap_timeout
        self.stale_frames = stale_frames
        self.movement_thresh = movement_thresh

        self._last_count_time: Optional[float] = None
        self._stale_tracks: Dict[int, int] = defaultdict(int)  # track_id → stale frame count
        self._active_alerts: List[dict] = []

    def update(self, tracks, counted: list, frame_idx: int) -> List[dict]:
        """Process current frame state and return any new alerts."""
        new_alerts = []

        now = time.time()

        # ── Gap detection ────────────────────────────
        if counted:
            self._last_count_time = now
        elif self._last_count_time is not None:
            gap_duration = now - self._last_count_time
            if gap_duration > self.gap_timeout:
                alert = {
                    "type": "gap",
                    "severity": "WARN",
                    "message": f"No nuts counted for {gap_duration:.0f}s",
                    "timestamp": now,
                    "frame": frame_idx,
                }
                new_alerts.append(alert)
                self._last_count_time = now  # reset to avoid repeated alerts

        # ── Jam detection ────────────────────────────
        # Check for stalled tracks inside the counting zone
        stale_ids = set()
        for track in tracks:
            if hasattr(track, 'frames_since_movement'):
                if track.frames_since_movement >= self.stale_frames:
                    stale_ids.add(track.track_id)

        if len(stale_ids) >= self.jam_threshold:
            alert = {
                "type": "jam",
                "severity": "ERROR",
                "message": f"Jam detected: {len(stale_ids)} stalled objects",
                "stalled_track_ids": list(stale_ids),
                "timestamp": now,
                "frame": frame_idx,
            }
            new_alerts.append(alert)

        # ── Track alert history ──────────────────────
        self._active_alerts.extend(new_alerts)
        if len(self._active_alerts) > 50:
            self._active_alerts = self._active_alerts[-50:]

        return new_alerts

    @property
    def recent_alerts(self) -> List[dict]:
        return self._active_alerts[-10:]

    @property
    def active(self) -> bool:
        """Whether any recent alert is still active."""
        if not self._active_alerts:
            return False
        return (time.time() - self._active_alerts[-1]["timestamp"]) < 5.0
