"""Typed configuration loader. Reads configs/default.yaml into frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "yolo_nas_s"
    checkpoint: str | None = None
    num_classes: int = 2
    class_names: list[str] = field(default_factory=lambda: ["nut"])


@dataclass(frozen=True)
class PipelineConfig:
    confidence_threshold: float = 0.35
    device: str = "cuda"
    source: str | int | None = None


@dataclass(frozen=True)
class TrackingConfig:
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"
    track_buffer: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass(frozen=True)
class CountingConfig:
    roi_points: list[list[int]] = field(default_factory=lambda: [
        [125, 430], [205, 405], [300, 378], [400, 335], [470, 355],
        [500, 390], [435, 425], [330, 465], [220, 520], [140, 500],
    ])
    count_line: list[list[int]] = field(default_factory=lambda: [[330, 395], [365, 468]])
    count_direction: str = "any"  # "left_to_right", "right_to_left", or "any"
    min_track_age: int = 4
    min_mean_confidence: float = 0.6
    direction_cos_threshold: float = 0.5
    min_speed: float = 0.3
    max_speed: float = 50.0
    min_area: float = 100
    max_area: float = 250_000
    max_stale_frames: int = 30
    stale_movement_thresh: float = 5.0
    gap_timeout: float = 10.0


@dataclass(frozen=True)
class SahiConfig:
    enabled: bool = False
    slice_height: int = 640
    slice_width: int = 640
    overlap_ratio: float = 0.2


@dataclass(frozen=True)
class OutputConfig:
    csv_path: str = "logs/counts.csv"
    video_path: str | None = None
    display: bool = True
    display_width: int = 960


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    sahi: SahiConfig = field(default_factory=SahiConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            pipeline=PipelineConfig(**raw.get("pipeline", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
            counting=CountingConfig(**raw.get("counting", {})),
            sahi=SahiConfig(**raw.get("sahi", {})),
            output=OutputConfig(**raw.get("output", {})),
        )


CONVEYOR_UNIT_VECTORS = {
    "down":  (0.0, 1.0),
    "up":    (0.0, -1.0),
    "left":  (-1.0, 0.0),
    "right": (1.0, 0.0),
}
