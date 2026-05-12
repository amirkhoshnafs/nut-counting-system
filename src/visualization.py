"""OpenCV visualization overlay for the counting pipeline."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from src.config import Config


# Per-state colors (BGR)
STATE_COLORS = {
    "PENDING": (0, 255, 255),     # Yellow
    "CONFIRMED": (0, 255, 0),     # Green
    "COUNTED": (255, 0, 0),       # Blue
    "STALE": (0, 0, 255),         # Red
    "LOST": (128, 128, 128),      # Gray
}


class Visualizer:
    """Draws bounding boxes, track IDs, counting lines, and live counts."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def draw(self, result: dict) -> np.ndarray:
        """Draw overlay on the frame."""
        frame = result["frame"].copy()
        h, w = frame.shape[:2]

        # Draw ROI polygon (if configured)
        if self.cfg.counting.roi_points:
            roi = np.array(self.cfg.counting.roi_points, dtype=np.int32)
            cv2.polylines(frame, [roi], True, (255, 0, 0), 2)

        # Draw counting line
        cl = self.cfg.counting.count_line
        pt1 = (int(cl[0][0]), int(cl[0][1]))
        pt2 = (int(cl[1][0]), int(cl[1][1]))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 3)

        # Draw tracks — skip stale ones that left the frame
        for track in result["tracks"]:
            if track.frames_since_movement >= 3:  # no recent detection = left frame
                continue
            color = STATE_COLORS.get(track.state, (255, 255, 255))
            centroid = track.latest_centroid

            if centroid is not None:
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (cx, cy), 6, color, -1)
                cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 1)
                # Show count ID for counted nuts, tracker ID for active tracks
                label = f"#{track.count_id}" if track.count_id else f"t{track.track_id}"
                cv2.putText(frame, label,
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Trajectory tails disabled for clean display

        # Draw count panel
        overlay = frame.copy()
        panel_h = 80
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        total = result.get("total_count", 0)
        cv2.putText(frame, f"TOTAL: {total}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        class_counts = result.get("class_counts", {})
        y_pos = 70
        for cls_id, count in class_counts.items():
            name = self.cfg.model.class_names[cls_id] if cls_id < len(self.cfg.model.class_names) else f"cls{cls_id}"
            cv2.putText(frame, f"  {name}: {count}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20

        # Draw newly counted highlight
        for event in result.get("counted", []):
            centroid = event.get("centroid")
            if centroid:
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])),
                           15, (0, 255, 255), 2)

        # Resize for display
        disp_w = self.cfg.output.display_width
        disp_h = int(h * (disp_w / w))
        if disp_w != w:
            frame = cv2.resize(frame, (disp_w, disp_h))

        return frame
