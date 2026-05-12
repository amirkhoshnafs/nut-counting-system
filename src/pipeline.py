"""Threading scaffold: LatestFrameBuffer + four-stage pipeline.

    CameraThread → [LatestFrameBuffer] → DetectorThread
        → [detection_queue] → TrackerThread
            → [result_queue] → VizThread
"""

from __future__ import annotations

import queue
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from super_gradients.training import models

from src.config import Config, CountingConfig, CONVEYOR_UNIT_VECTORS
from src.tracking import ByteTrackWrapper
from src.counting import CountingZone
from src.visualization import Visualizer
from src.exporters import CSVLogger


# ── Threads ──────────────────────────────────────────────

class CameraThread(threading.Thread):
    """Reads frames from a video source and puts them in a queue."""

    def __init__(self, source: str | int, frame_queue: queue.Queue,
                 config: Config, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.config = config
        self._stop_event = stop_event

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_queue.put(frame)  # blocks if queue full (detector is slower)

        cap.release()
        self.frame_queue.put(None)  # sentinel to signal end of video


class DetectorThread(threading.Thread):
    """Runs YOLO-NAS on frames from the queue."""

    def __init__(self, frame_queue: queue.Queue, det_queue: queue.Queue,
                 config: Config, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.det_queue = det_queue
        self.cfg = config
        self._stop_event = stop_event

    def run(self) -> None:
        model = models.get(
            self.cfg.model.name,
            pretrained_weights=None,
            num_classes=self.cfg.model.num_classes,
            checkpoint_path=self.cfg.model.checkpoint,
        )
        device = self.cfg.pipeline.device
        model.to(device)
        model.eval()

        conf = self.cfg.pipeline.confidence_threshold

        while not self._stop_event.is_set():
            frame = self.frame_queue.get()
            if frame is None:  # sentinel — end of video
                break

            result = model.predict(frame, conf=conf, iou=0.65, fuse_model=False, fp16=(device == "cuda"))
            pred = result.prediction

            boxes = pred.bboxes_xyxy
            scores = pred.confidence
            labels = pred.labels

            detections = []
            if boxes is not None and len(boxes) > 0:
                for box, score, label in zip(boxes, scores, labels):
                    if int(label) == 0:  # nut only
                        detections.append((box, float(score)))

            # Push to tracker queue (non-blocking, drop if full)
            try:
                self.det_queue.put({"frame": frame, "detections": detections}, timeout=0.5)
            except queue.Full:
                pass

        self.det_queue.put(None)  # sentinel to tracker


class TrackerThread(threading.Thread):
    """Runs ByteTrack + counting zone logic."""

    def __init__(self, det_queue: queue.Queue, result_queue: queue.Queue,
                 config: Config, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.det_queue = det_queue
        self.result_queue = result_queue
        self.cfg = config
        self._stop_event = stop_event

    def run(self) -> None:
        tracker = ByteTrackWrapper(self.cfg.tracking)
        counter = CountingZone(self.cfg.counting)
        logger = CSVLogger(self.cfg.output.csv_path)

        # Load calibrator, alerts, reporter
        from src.calibration import Calibrator
        from src.statistics import BatchReporter
        from src.alerts import AlertManager
        from pathlib import Path as _Path

        calibrator_path = _Path(__file__).resolve().parent.parent / "checkpoints/calibrator.pkl"
        calibrator = Calibrator(calibrator_path) if calibrator_path.exists() else None
        self.reporter = BatchReporter()
        alerts = AlertManager(
            jam_threshold=3,
            gap_timeout=self.cfg.counting.gap_timeout,
            stale_frames=self.cfg.counting.max_stale_frames,
            movement_thresh=self.cfg.counting.stale_movement_thresh,
        )
        frame_idx = 0

        while not self._stop_event.is_set():
            data = self.det_queue.get()
            if data is None:  # sentinel — end of video
                break

            frame = data["frame"]
            detections = data["detections"]

            tracks = tracker.update(detections)
            counted = counter.update(tracks, frame.shape)

            # Apply calibration to counted events
            for c in counted:
                raw_conf = c["confidence"]
                if calibrator:
                    c["calibrated_confidence"] = float(calibrator.calibrate_single(raw_conf))
                    c["reliability"] = calibrator.reliability_label(c["calibrated_confidence"])
                    c["needs_review"] = calibrator.needs_review(c["calibrated_confidence"])
                else:
                    c["calibrated_confidence"] = raw_conf
                    c["reliability"] = "MEDIUM" if raw_conf >= 0.5 else "LOW"
                    c["needs_review"] = raw_conf < 0.5
                logger.log(c)
                self.reporter.add_count(c)

            # Check alerts
            new_alerts = alerts.update(tracks, counted, frame_idx)
            for a in new_alerts:
                print(f"  ⚠ {a['severity']}: {a['message']}")

            result = {
                "frame": frame,
                "tracks": tracks,
                "counted": counted,
                "total_count": counter.total_count,
                "class_counts": counter.class_counts,
                "alerts": new_alerts,
            }
            frame_idx += 1

            try:
                self.result_queue.put(result, timeout=0.5)
            except queue.Full:
                pass

        self.result_queue.put(None)  # sentinel to viz


class VizThread(threading.Thread):
    """Renders overlay and shows/saves output."""

    def __init__(self, result_queue: queue.Queue, config: Config,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.cfg = config
        self._stop_event = stop_event

    def run(self) -> None:
        vis = Visualizer(self.cfg)
        video_writer = None

        while not self._stop_event.is_set():
            result = self.result_queue.get()
            if result is None:  # sentinel — end of video
                break

            frame = vis.draw(result)

            # Lazy-init video writer on first frame
            if self.cfg.output.video_path and video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    self.cfg.output.video_path, fourcc, 30.0, (w, h),
                )

            if self.cfg.output.display:
                cv2.imshow("Nut Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop_event.set()

            if video_writer:
                video_writer.write(frame)

        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


# ── Pipeline runner ──────────────────────────────────────

def run_pipeline(config_path: str | None = None) -> None:
    """Start all threads and run until video ends or user quits."""
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "configs/default.yaml")
    cfg = Config.from_yaml(config_path)

    if cfg.pipeline.source is None:
        raise ValueError("pipeline.source must be set in config (video path, 0 for webcam, or rtsp://)")

    stop_event = threading.Event()

    # Handle Ctrl+C
    def _handle_sigint(signum, frame):
        print("\nStopping pipeline ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    # Create queues
    frame_queue = queue.Queue(maxsize=64)
    det_queue = queue.Queue(maxsize=8)
    result_queue = queue.Queue(maxsize=32)

    # Create threads
    camera = CameraThread(cfg.pipeline.source, frame_queue, cfg, stop_event)
    detector = DetectorThread(frame_queue, det_queue, cfg, stop_event)
    tracker = TrackerThread(det_queue, result_queue, cfg, stop_event)
    viz = VizThread(result_queue, cfg, stop_event)

    threads = [camera, detector, tracker, viz]
    for t in threads:
        t.start()

    # Wait for all threads to finish naturally (sentinel propagation)
    for t in threads:
        t.join()

    # Print batch report
    if hasattr(tracker, 'reporter'):
        print("\n" + tracker.reporter.summary())
    print("Pipeline stopped.")
