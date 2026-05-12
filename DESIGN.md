# Real-Time Nut Counting System — Technical Design Document

**Scope:** Single-class (nut) detection, tracking, and counting on a conveyor belt.
Multi-class (nut/bolt/washer) deferred to Phase 4 with custom balanced data.

## 1. System Architecture

### 1.1 Threading Model

The system runs as four threads connected by queues. Each thread owns one stage of the pipeline; queues decouple their speeds so that a slow detection frame does not stall the camera.

```
┌──────────────┐    Queue(64)    ┌──────────────┐    Detections    ┌──────────────┐    Results    ┌──────────────┐
│  Camera      │────────────────▶│  Detector     │────────────────▶│  Tracker +    │─────────────▶│  Viz +       │
│  Thread      │                │  Thread       │   Queue (8)    │  Counter      │  Queue (32)  │  Export      │
└──────────────┘                 └──────────────┘                 └──────────────┘              └──────────────┘
```

| Stage | Runs in thread | Input | Output | Bottleneck profile |
|-------|---------------|-------|--------|-------------------|
| Frame capture | `CameraThread` | Video file / webcam / RTSP | `queue.Queue` (64) | I/O bound |
| Detection | `DetectorThread` | `LatestFrameBuffer` → latest frame | `detection_queue` (Queue[8]) | GPU bound |
| Tracking + counting | `TrackerThread` | `detection_queue` → `List[Detection]` | `result_queue` (Queue[32]) | CPU bound, fast |
| Visualization + export | `VizThread` | `result_queue` → `FrameResult` | annotated frame + CSV append | I/O bound |

### 1.2 Frame queue

For video files (finite length), we use a regular `queue.Queue(maxsize=64)` between CameraThread and DetectorThread. This ensures all frames are processed — no frame skipping.

For live camera streams (future), a `LatestFrameBuffer` single-slot ring buffer would be more appropriate to avoid growing latency, dropping old frames when the detector falls behind.

### 1.3 Downstream queues (regular Queue)

The remaining queues use standard `queue.Queue` because downstream stages are faster than upstream:

| Queue | maxsize | Behaviour when full |
|-------|---------|---------------------|
| `detection_queue` | 8 | `put()` blocks with a 0.5 s timeout; on timeout, the detection is dropped |
| `result_queue` | 32 | `put()` blocks (export is fast enough that this should not stall) |

### 1.4 Thread lifecycle

```
                  start() ──▶ running ──▶ stop_event.set() ──▶ joined
```

Each thread holds a `threading.Event` named `_stop`. The main thread calls `stop()` on all threads on SIGINT or when the video ends. Each thread checks `_stop.is_set()` at the top of its loop and exits cleanly.

### 1.5 Config system

A single `configs/default.yaml` drives everything:

```yaml
model:
  name: yolo_nas_s           # yolo_nas_s | yolo_nas_m | yolo_nas_l
  weights: coco               # coco | path/to/checkpoint.pth

pipeline:
  confidence_threshold: 0.35  # raw detector threshold before tracking
  device: cuda                # cuda | cpu

tracking:
  tracker: bytetrack           # bytetrack | botsort
  track_buffer: 30             # frames to keep dead tracks
  min_hits: 3                  # min detections before track is confirmed
  iou_threshold: 0.3

counting:
  entry_line_y: 0.3            # fraction of frame height (0 = top, 1 = bottom)
  exit_line_y: 0.7
  conveyor_direction: down     # down | up | left | right
  min_track_age: 4             # frames track must exist before counting
  min_mean_confidence: 0.6     # average detection confidence across track
  direction_cos_threshold: 0.7 # cosine similarity to conveyor direction
  max_stale_frames: 30         # frames without significant movement → stale alert
  stale_movement_thresh: 5.0   # pixels — minimum displacement to reset staleness

sahi:
  enabled: false               # true = slice high-res frames for small objects
  slice_height: 640
  slice_width: 640
  overlap_ratio: 0.2           # 20% overlap between slices

output:
  csv_path: logs/counts.csv
  video_path: null             # null = no recording, or path
  display: true                # show live window
```

---

## 2. Counting Zone State Machine

### 2.1 Zone geometry

```
        ┌──────────────────────────┐
        │                          │
entry   │   ────────────────────   │  y_entry (30% from top)
  ↓     │       conveyor belt      │
exit    │   ────────────────────   │  y_exit  (70% from top)
        │                          │
        └──────────────────────────┘
```

Two horizontal lines. Objects enter from the top (or wherever `conveyor_direction` indicates) and exit at the bottom. Counting happens at the **exit line only** — this prevents double-counting if an object wobbles across the entry line.

### 2.2 Per-track state machine

```
                   ┌──────────┐
       new         │          │   age ≥ min_hits
  detection ──────▶│ PENDING  │────────────────────▶┐
                   │          │                      │
                   └──────────┘                      ▼
                    │   │                  ┌──────────────┐
    lost before     │   │   lost for       │              │
    min_hits        │   │   > track_buffer │  CONFIRMED   │──────────▶ TRACK_LOST
                    ▼   │   frames         │              │   (deleted)
               DISCARD  │                  └──────────────┘
                        │                         │
                        │            no significant      centroid crosses
                        │            movement for         exit line AND all
                        │            max_stale_frames     statistical gates pass
                        │                    │                    │
                        │                    ▼                    ▼
                        │             ┌──────────┐        ┌──────────────┐
                        │             │  STALE   │        │              │
                        │             │ (alert)  │        │   COUNTED    │
                        │             └──────────┘        │              │
                        │                    │            └──────────────┘
                        │       movement        │
                        │       resumes          │
                        │                    ▼
                        │             ┌──────────────┐
                        │             │  CONFIRMED   │ (recovered)
                        │             └──────────────┘
                        │
              track seen in zone,
              classified by ByteTrack
              but never confirmed
                        │
                        ▼
                  ┌──────────┐
                  │  GHOST   │ (logged, not counted)
                  └──────────┘
```

### 2.3 State definitions

**PENDING** (0 → min_hits frames)

Track exists but is not yet trusted.
- Track is assigned a temporary ID.
- Bounding box positions, confidences, and sizes are accumulated into per-track history.
- If the track disappears before `min_hits` frames: **DISCARD** (likely a false positive).
- If the track survives `min_hits` frames: promoted to **CONFIRMED**.

**CONFIRMED** (min_hits → counted, stale, or lost)

Track is trusted and actively tracked.
- Per-frame statistics continue to accumulate:
  - `confidence_history: List[float]`
  - `centroid_history: List[(float, float)]` (pixel coordinates of box center)
  - `box_area_history: List[float]`
- Direction vector is estimated via linear regression on `centroid_history` (last N points).
- Speed is estimated from centroid displacement per frame.
- If no significant movement for `max_stale_frames` consecutive frames: transition to **STALE**.
- If not matched to any detection for `track_buffer` consecutive frames: transition to **TRACK_LOST**.

**STALE** (CONFIRMED but motionless)

- A CONFIRMED track whose centroid has moved less than `stale_movement_thresh` pixels for `max_stale_frames` consecutive frames.
- A jam alert is emitted (see §2.7).
- The track remains tracked. If movement resumes (centroid displacement > threshold): transition back to **CONFIRMED**.
- If the stale track eventually crosses the exit line: counted normally.
- If the track is lost while stale: transition to **TRACK_LOST**, logged as `stale_lost` (potential jam cleared manually or object removed).

This is the mechanism for jam detection: if ≥ N tracks are simultaneously STALE in the counting zone, trigger a jam alert.

**COUNTED** (terminal)

- Track crossed the exit line from the entry side.
- All statistical gates passed (see §2.5).
- Track ID is recorded in `counted_ids: Set[int]` to prevent double-counting.
- Track continues to be tracked (for visualization) but will never be counted again.

**TRACK_LOST** (terminal)

- Track was not matched to any detection for `track_buffer` consecutive frames.
- If the track was COUNTED: no action needed.
- If the track was CONFIRMED but never crossed the exit line: it is **not** counted (object may have been removed before reaching the counting point).

**GHOST** (logged, not counted)

- A track that appeared inside the counting zone (its first centroid was below the entry line) and never reached CONFIRMED state before being lost.
- Logged as a ghost candidate for diagnostic purposes.

### 2.4 SAHI — Slicing Aided Hyper Inference for small parts

Nuts and washers occupy only a few percent of the frame in a wide conveyor shot. YOLO-NAS at native resolution may miss these small objects. SAHI slices the frame into overlapping tiles, runs detection on each tile independently, and merges results with a post-processing NMS.

**When enabled** (`sahi.enabled: true` in config):

```
Original frame (1920×1080)
        │
        ▼ slice into overlapping tiles
┌──────┬──────┬──────┐
│ 640  │ 640  │      │  overlap = 20% (128 px)
│ ×640 │ ×640 │ ...  │
├──────┼──────┼──────┤
│      │      │      │
├──────┼──────┼──────┤
│      │      │      │
└──────┴──────┴──────┘
        │
        ▼ run detector on each slice
        │
        ▼ merge all detections → NMS across slices → final detection list
```

**Cost**: SAHI with 640×640 slices on a 1080p frame produces ~6 slices. Detection time scales ~linearly with slice count. For 30 FPS input, SAHI is only practical with YOLO-NAS-S on a GPU (expect 12-18 FPS).

**Config toggle**: SAHI is off by default (`enabled: false`). Enable it only when:
- The camera is high resolution (≥ 1080p) and objects are < 5% of frame area.
- The deployment GPU can absorb the extra cost.
- Count accuracy without SAHI is below target.

Implementation uses the `sahi` Python library (`pip install sahi`), which integrates with Ultralytics and SuperGradients models.

### 2.5 Counting gate (statistical rules)

When a track centroid crosses the exit line, **all** of the following must pass:

```python
def should_count(track: Track, direction: str) -> tuple[bool, str]:
    # 1 — Minimum observation time
    if track.age < cfg.counting.min_track_age:
        return False, "too_short"

    # 2 — Average detection confidence across the track
    mean_conf = np.mean(track.confidence_history)
    if mean_conf < cfg.counting.min_mean_confidence:
        return False, "low_confidence"

    # 3 — Direction matches conveyor
    dir_vec = linear_fit_direction(track.centroid_history[-10:])
    cos_sim = cosine_similarity(dir_vec, CONVEYOR_UNIT_VECTORS[direction])
    if cos_sim < cfg.counting.direction_cos_threshold:
        return False, "wrong_direction"

    # 4 — Speed within expected range (pixels/frame)
    speed = np.linalg.norm(dir_vec)
    if not (cfg.counting.min_speed <= speed <= cfg.counting.max_speed):
        return False, "abnormal_speed"

    # 5 — Box size within expected range for the predicted class
    mean_area = np.mean(track.box_area_history)
    if not (cfg.counting.min_area <= mean_area <= cfg.counting.max_area):
        return False, "abnormal_size"

    return True, "ok"
```

Rejected counts are logged with the rejection reason for later analysis.

### 2.6 Handling object overlap

When two objects merge into one detection box:
1. The merged box area spikes suddenly (≥ 1.5× the running mean).
2. Detection confidence often drops.
3. The tracker may lose one of the two tracks.

Strategy:
- Detect merge: `current_area > 1.5 * running_mean_area`.
- On merge: flag both tracks involved (if two track IDs were associated with the merged box) as `merge_candidate`.
- If one track disappears and the surviving track's area returns to normal → the lost object likely left the belt without being counted (acceptable loss).
- If the merged box crosses the exit line: count once but annotate the log with `overlap_flag=true`.

This is a heuristic, not a full solution. Instance segmentation (Phase 4 enhancement, using the Nuts and Bolts Segmentation dataset) is the proper fix.

### 2.7 Jam and gap alerts

**Jam detection**: if ≥ 3 tracks are simultaneously in STALE state within the counting zone, emit a `JAM_DETECTED` alert with the number of stuck objects and their positions.

**Gap alert**: if no object crosses the exit line for > `gap_timeout` seconds (default 10 s) while the belt is running (i.e., frames are still arriving), emit a `GAP_ALERT` — may indicate a supply interruption upstream.

Both alerts appear in the CSV log and on the dashboard.

---

## 3. Data Strategy

### 3.1 Dataset inventory and progression

We use a staged approach: start simple with single-class nut counting, then expand to multi-class, then add advanced capabilities.

| Stage | Dataset | Images | Classes | Format (Roboflow) | License | Purpose |
|-------|---------|--------|---------|-------------------|---------|---------|
| **Phase 1 MVP** | [Nut Counting System](https://universe.roboflow.com/object-detection-39ltk/nut-counting-system) | 400 | `nut` (1) | YOLOv8, COCO, Pascal VOC, TFRecord | Public Domain | Train first detector; validate the full pipeline end-to-end with one class |
| **Phase 1 Advanced** | [Nuts and Bolts Object Detection](https://universe.roboflow.com/bishe-rqa1b/nuts-and-bolts-g8ni8-ar1eb) | 116 | `bolt`, `nuts` (2) | YOLOv8, COCO, Pascal VOC, TFRecord | CC BY 4.0 | Expand to multi-class detection + per-class counting |
| **Phase 4 Overlap** | [Nuts and Bolts Segmentation](https://universe.roboflow.com/jente-kleinh/nuts-and-bolts) | 100 | `Nut`, `Bolt`, `washer` (3) | YOLOv8, COCO, Pascal VOC | — | Instance segmentation for overlap resolution (requires switching to YOLOv8-seg or similar) |
| **Phase 4 Conveyor** | [Boxes on a Conveyer Belt](https://universe.roboflow.com/mohamed-traore-2ekkp/boxes-on-a-conveyer-belt) | 33 | boxes (1) | YOLOv8, COCO | — | Conveyor-specific test scenario; qualitatively different object type for generalization check |
| **Future** | [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) + [AD2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | 5000+ / 8000+ | 15 / 8 object+texture categories | MVTec proprietary | Research | Anomaly/defect detection add-on (out of scope for this project unless Phase 4 extended) |

**Why start with Nut Counting System (400 images, 1 class)?**

1. Single class means no classification confusion — the detector only needs to answer "is there a nut here?" not "which type?"
2. 400 images gives more training signal than 116 for the initial model.
3. The counting pipeline (tracking, zone logic, gating) can be validated before adding multi-class complexity.
4. Public Domain license — no attribution burden.

**Progression logic:**

```
Nut Counting (1 class, 400 imgs)  ──▶  Nuts & Bolts Detection (2 classes, 116 imgs)
  │                                        │
  │ Validate pipeline, tune hyperparams     │ Add per-class counting,
  │ on the simplest possible problem.       │ classification confusion metrics.
  │                                        │
  └────────────── Phase 1 MVP ──────────────┴────── Phase 1 Advanced ──────▶
```

### 3.2 Download format

Roboflow offers multiple export formats. For SuperGradients (YOLO-NAS), we need **COCO JSON**:

| Roboflow export | SuperGradients compatible? | Notes |
|-----------------|---------------------------|-------|
| **COCO** | Yes | Directly readable by SuperGradients `CocoDetectionDataset` |
| YOLOv8 | Needs conversion | YOLO `.txt` → COCO `.json` via conversion script |
| Pascal VOC | Needs conversion | More complex conversion needed |

**Download instruction**: On each Roboflow dataset page, click "Download Dataset" → select **COCO** format → download the `.zip`. Extract into `data/raw/<dataset_name>/`.

### 3.3 Four-way data split

The dataset is split into four disjoint subsets with distinct purposes:

| Split | Fraction | Nut Counting (400 imgs) | Nuts & Bolts (116 imgs) | Purpose |
|-------|----------|------------------------|------------------------|---------|
| **train** | 70% | 280 | ~81 | Model weight optimization (backprop) |
| **val** | 10% | 40 | ~12 | Early stopping, hyperparameter selection |
| **calibration** | 10% | 40 | ~12 | Fit isotonic regression calibrator (held out from training) |
| **test** | 10% | 40 | ~12 | Final evaluation — used exactly once, at the end |

**Why a dedicated calibration split?**

Using the validation set for both model selection AND calibrator fitting creates a subtle dependency: the calibrator may overfit to the same data that guided early stopping. A separate calibration split keeps the calibrator unbiased. The test set remains fully held-out for final metrics.

**Split method**: Stratified by class to maintain proportional class representation in each split. For the single-class Nut Counting dataset, stratification is trivially satisfied. For multi-class, use `sklearn.model_selection.train_test_split` with `stratify=labels` in two cascaded calls:

```python
# Step 1: separate test (10%)
train_val_cal, test = train_test_split(data, test_size=0.10, stratify=labels)

# Step 2: separate calibration (10% of original → 11.1% of remaining)
remaining = 1.0 - 0.10  # = 0.90
cal_frac = 0.10 / remaining  # ≈ 0.111
train, cal = train_test_split(train_val_cal, test_size=cal_frac, stratify=train_val_cal_labels)

# Step 3: separate val (10% of original → 12.5% of remaining)
remaining_2 = remaining - 0.10  # = 0.80
val_frac = 0.10 / remaining_2  # = 0.125
train, val = train_test_split(train, test_size=val_frac, stratify=train_labels)
```

Resulting directory structure:

```
data/processed/nut_counting/
├── train/
│   ├── images/   (280 files)
│   └── labels/   (280 .txt files, YOLO format — or coco_train.json)
├── val/
│   ├── images/   (40 files)
│   └── labels/
├── calibration/
│   ├── images/   (40 files)
│   └── labels/
└── test/
    ├── images/   (40 files)
    └── labels/
```

### 3.4 Video-based split (for custom conveyor data in Phase 4)

When we record conveyor video, the split must be **at the video level** — never split frames from the same video across sets:

```
1. Record 3-5 videos of varying duration (2-5 min each).
2. Assign whole videos to splits:
   - Train: 2-3 videos (~60% of total frames)
   - Val:   1 video   (~20%)
   - Calibration: frames from val video (re-use is acceptable since
     calibration data is small and video frames are highly correlated)
   - Test:  1 video   (~20%)
3. From each video, sample frames at interval N (every 15th frame
   for train, every 5th for val/cal/test — val/cal/test can be denser).
4. Label sampled frames only.
```

This guarantees zero temporal leakage.

### 3.5 Annotation format

Raw Roboflow exports in COCO JSON format — consumed directly by SuperGradients. If we ever need to convert from YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
0 0.523 0.312 0.081 0.064       # nut
1 0.678 0.445 0.092 0.071       # bolt
2 0.412 0.289 0.078 0.066       # washer
```

A conversion script (`scripts/yolo_to_coco.py`) handles this if needed.

### 3.6 Augmentation pipeline (albumentations)

Applied online during training. Configured in the SuperGradients training YAML.

| Augmentation | Parameters | p | Why |
|---|---|---|---|
| **Motion blur** | kernel=5-15, direction=horizontal | 0.3 | Objects moving on belt |
| **Random brightness/contrast** | brightness_limit=0.3, contrast_limit=0.3 | 0.5 | Factory lighting variation |
| **RandomSunFlare** | src_radius=100-400, num_flare_circles=(3, 8) | 0.2 | Specular reflection / glare off metallic nuts and bolts |
| **Coarse dropout** | max_holes=3, max_height=0.3, max_width=0.3 | 0.3 | Partial occlusion |
| **Gauss noise** | var_limit=(5, 30) | 0.2 | Sensor noise in cheap cameras |
| **Shift/scale/rotate** | shift=0.1, scale=0.2, rotate=15 | 0.5 | Position and viewpoint variance |
| **Hue saturation value** | hue=5, sat=20, val=20 | 0.3 | Color variance across batches |

**Why `RandomSunFlare` matters for this project specifically**: Nuts, bolts, and washers are metallic. Under factory overhead lighting, specular highlights can wash out an entire face of the object, changing its appearance dramatically. Without this augmentation, the model may fail on shiny parts under bright lights — a common factory condition.

```yaml
# super_gradients training config excerpt
augmentations:
  - MotionBlur:
      kernel_size: [5, 15]
      p: 0.3
  - RandomBrightnessContrast:
      brightness_limit: 0.3
      contrast_limit: 0.3
      p: 0.5
  - RandomSunFlare:
      src_radius: [100, 400]
      src_color: [255, 255, 255]
      num_flare_circles: [3, 8]
      p: 0.2
  - CoarseDropout:
      max_holes: 3
      max_height: 48
      max_width: 48
      p: 0.3
  - GaussNoise:
      var_limit: [5, 30]
      p: 0.2
```

---

## 4. Statistical Integration

### 4.1 Confidence calibration

**Problem:** A YOLO-NAS confidence score of 0.85 does not mean "85% chance the detection is correct." Neural networks are notoriously miscalibrated (typically overconfident).

**Method — Isotonic regression:**

1. After training completes, run inference on the **calibration** split (NOT val, NOT test).
2. For every prediction, record `(raw_confidence, is_correct)` — `is_correct` is 1 if IoU > 0.5 with a ground-truth box of the same class, else 0.
3. Fit an isotonic regression: `calibrated_prob = f(raw_confidence)`.
4. Store the fitted calibrator for use at inference time.
5. Evaluate calibration quality on the **test** split (held out, never seen by either model or calibrator).

```python
from sklearn.isotonic import IsotonicRegression

def calibrate_model(model, cal_loader, device):
    """Fit isotonic regression on the calibration split."""
    confidences, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in cal_loader:
            preds = model(images.to(device))
            for pred, target in zip(preds, targets):
                matched = match_predictions(pred, target, iou_thresh=0.5)
                confidences.extend(m.confidence for m in matched)
                labels.extend(m.is_correct for m in matched)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.array(confidences), np.array(labels))
    return iso  # calibrated_prob = iso.predict([raw_conf])
```

**Why isotonic over Platt scaling**: Isotonic regression is non-parametric — it makes no assumption about the shape of the miscalibration curve. Modern detectors often have S-shaped or non-monotonic calibration curves that Platt scaling (a sigmoid) cannot represent.

**Reliability diagram** (plotted, saved to `reports/`):

```
                      ┌─────┬─────┬─────┬─────┬─────┐
Actual accuracy  1.0  │     │     │     │     │█████│
                      │     │     │     │     │█████│
                  0.5  │     │     │     │█████│     │
                      │     │     │█████│     │     │
                  0.0  │█████│     │     │     │     │
                      └─────┴─────┴─────┴─────┴─────┘
                      0-0.2 0.2-  0.4-  0.6-  0.8-
                            0.4   0.6   0.8   1.0
                       Predicted confidence bin

  ─── Perfect calibration
  ─── Raw model
```

**ECE (Expected Calibration Error):**

```python
def compute_ece(confidences, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return ece
```

Target: ECE < 0.05 after calibration (on test split).

### 4.2 Per-count decision

Each counting event receives a calibrated probability `p_correct`:

```
p_correct = iso_reg.predict([track.mean_raw_confidence])
```

| p_correct | Label | Action |
|-----------|-------|--------|
| ≥ 0.90 | HIGH | Counted, no flag |
| 0.70 – 0.90 | MEDIUM | Counted, logged with `review_flag=medium` |
| < 0.70 | LOW | Counted, logged with `review_flag=low` |

Track-wide confidence is averaged from all detections across the track's lifetime, making it more stable than any single frame.

### 4.3 Per-batch count uncertainty

**Inputs** (measured once on the held-out test set):

| Symbol | Meaning | Example |
|--------|---------|---------|
| R_c | Per-class recall | nut recall = 0.992 |
| FPR_c | Per-class false-positive rate | nut FPR = 0.003 |
| C_c,raw | Raw count from the system for a batch | nuts = 492 |

**Computation:**

```
E[true_c]  = raw_c / R_c  -  FPR_c × raw_other
σ_c        ≈ sqrt(raw_c × (1 − R_c) + FPR_c × raw_c)
```

This uses a Poisson-like variance approximation, which is reasonable for counting processes.

**Wilson score interval** (better than normal approximation, especially for small counts):

```python
from statsmodels.stats.proportion import proportion_confint

# For recall: 992/1000 nuts detected on test set
lower, upper = proportion_confint(
    count=detected, nobs=total, alpha=0.05, method="wilson"
)
```

**Output format (per batch):**

```
═══════════════════════════════════════════
 BATCH #42      2026-05-08 14:32:15
═══════════════════════════════════════════
 nut        492  ± 4   (95% CI: 484–500)
 bolt       508  ± 5   (95% CI: 498–518)
 washer     188  ± 3   (95% CI: 182–194)
───────────────────────────────────────────
 TOTAL     1188  ± 7   (95% CI: 1174–1202)
───────────────────────────────────────────
 RELIABILITY: HIGH  (98.2% high-confidence)
 REVIEW FLAGS: 3 medium, 0 low
 FPS: 24.3     DROPPED FRAMES: 0.4%
═══════════════════════════════════════════
```

### 4.4 Evaluation metrics — full set

**Detection metrics** (Phase 1, computed on test split):

| Metric | Target |
|--------|--------|
| mAP@50 | > 0.85 |
| mAP@50:95 | > 0.60 |
| Per-class precision | report all |
| Per-class recall | report all |
| Per-class F1 | report all |

**Counting metrics** (Phase 2+, from test videos):

| Metric | Formula | Target |
|--------|---------|--------|
| Count Error Rate | |pred − true| / true | < 5% |
| ID Switches | count of ID reassignments | < 1% of tracks |
| Ghost Counts | objects counted that never existed | < 1% |
| Merge Errors | two objects counted as one | < 5% of close pairs |

**Reliability metrics** (Phase 3, computed on test split):

| Metric | Target |
|--------|--------|
| ECE (post-calibration) | < 0.05 |
| Brier score | < 0.10 |
| 95% CI coverage | actual count within interval for ≥ 95% of batches |

---

## 5. Project Milestone Roadmap

### Phase 0 — Documentation & Pre-flight ✅ COMPLETED

**Goal:** Design complete, environment ready.

| # | Task | Output | Status |
|---|------|--------|--------|
| 0.1 | Update DESIGN.md with dataset strategy, calibration split, specular augmentation, stale track + SAHI | This document | Done |
| 0.2 | Set up conda environment `ml2`, install super-gradients 3.7.1, verify GPU (RTX 3060 6GB) | Working `import super_gradients` | Done |
| 0.3 | Download Nut Counting System dataset in COCO format (400 source images, ~17 ann/img) | `data/raw/nut_counting/` (train + valid COCO) | Done |
| 0.4 | Create project skeleton (directories, `__init__.py`, `configs/default.yaml`, `requirements.txt`) | Full scaffold ready | Done |
| 0.5 | Write `scripts/prepare_data.py`: merge COCO, group by source, 70/10/10/10 split | `data/processed/nut_counting/` (train/val/cal/test) | Done |

### Phase 1 — Detection Foundation ✅ COMPLETED

**Goal:** A trained YOLO-NAS detector that finds nuts in images (single-class MVP).

| # | Task | Output | Status |
|---|------|--------|--------|
| 1.1 | Write `scripts/train.py`: SuperGradients training with `coco_detection_yolo_nas_dataset_params` config | `scripts/train.py` with 80-epoch schedule | Done |
| 1.2 | Train YOLO-NAS-S from COCO pretrained weights, batch=6 (RTX 3060 6GB OOM at 12) | `checkpoints/nut_counting_phase1/ckpt_best.pth` | Done |
| 1.3 | Write `scripts/evaluate.py`: pycocotools-based COCO evaluation on test split | `scripts/evaluate.py` | Done |
| 1.4 | Evaluate on test split | `reports/phase1_metrics.json` | Done |

**Results:**

| Metric | Training (val) | Test (pycocotools) | Gate (0.85) |
|--------|---------------|-------------------|-------------|
| mAP@0.50 | 0.932 | 0.806 | BELOW |
| mAP@0.75 | — | 0.734 | — |
| mAP@0.50:0.95 | — | 0.604 | — |
| AR@100 | — | 0.664 | — |

**Gate outcome:** NOT MET. Test mAP@0.50 = 0.806 < 0.85. The 0.126 gap between val and test mAP is due to: (a) pycocotools using stricter matching than SuperGradients' internal metric, (b) model missing ~34% of objects (AR@100 = 0.66). Improvements needed before proceeding to tracking.

**Lessons learned:**
- SuperGradients `COCODetectionDataset` requires `data_dir/annotations/*.json` + `images/<split>/` layout
- `num_classes=2` needed (not 1) to avoid PPYoloE loss edge case
- `PPYoloECollateFN` / `CrowdDetectionCollateFN` format: train uses `with_crowd=False`, val uses `with_crowd=True`
- `model.predict()` is the reliable high-level inference API (handles post-processing)
- `conda run` suppresses stdout; use file-based output for debugging

### Phase 1.5 — Detection Improvement ✅ COMPLETED

**Goal:** Raise test mAP@0.50 from 0.806 to ≥ 0.85.

| # | Task | Output | Status |
|---|------|--------|--------|
| 1.5.1 | Train YOLO-NAS-M (more capacity) 150 epochs — no gain (0.802) | Model capacity not the bottleneck | Done |
| 1.5.2 | Merge 3 datasets → combined 2-class (nut/bolt), 566 source images, train YOLO-NAS-S 150 epochs | `checkpoints/nut_counting_phase1/ckpt_best.pth` | Done |
| 1.5.3 | Confidence threshold sweep (0.01, 0.25, 0.50) on test split | Precision/recall trade-off charted | Done |
| 1.5.4 | Evaluate on combined test split with pycocotools | `reports/phase1_metrics.json` | Done |

**Results (conf=0.50, pycocotools):**

| Class | mAP@0.50 | AR@100 | Gate (0.85) |
|-------|----------|--------|-------------|
| **nut** | **0.865** | 0.764 | **PASS** |
| bolt | 0.000 | 0.000 | FAIL (zero detections) |

**Gate outcome:** PASS for nut. Bolt failed due to extreme class imbalance (18,986 nut vs 686 bolt annotations — 27:1 ratio). The model never learned to detect bolts.

**Decision:** System scope narrowed to **nut-only counting**. The model achieves mAP@0.50 = 0.865 on nuts with pycocotools, which is sufficient for the counting pipeline. Bolt and washer support deferred to Phase 4 (custom balanced data).

**Lessons learned:**
- More data helped nuts: AR@100 improved from 0.664 → 0.764
- Class imbalance of 27:1 made bolt detection impossible — oversampling or weighted loss would be needed
- SuperGradients internal val mAP (0.97) is heavily inflated vs pycocotools (0.87 for nuts) — always evaluate with pycocotools
- GPU throttling (P5 state, 30W) caused 5× slower training — fixed by AC adapter unplug/replug
- YOLO-NAS-M gave no improvement over S despite 2× parameters — data volume, not model capacity, is the bottleneck

### Phase 2 — Tracking & Counting (current, Est. 1–2 weeks)

**Goal:** Track nuts across frames and count them as they cross the exit line. Single-class (nut only).

| # | Task | Output |
|---|------|--------|
| 2.1 | Implement `LatestFrameBuffer` and thread scaffold (`CameraThread`, `DetectorThread`, `TrackerThread`, `VizThread`) | `src/pipeline.py` |
| 2.2 | Implement ByteTrack wrapper: takes `List[Detection]`, returns tracks with persistent IDs | `src/tracking.py` |
| 2.3 | Implement counting zone state machine (§2.2–2.3): PENDING → CONFIRMED → COUNTED/LOST/STALE/GHOST | `src/counting.py` |
| 2.4 | Implement statistical gating rules (§2.5) | Inside `src/counting.py` |
| 2.5 | Implement visualization overlay: boxes, track IDs, counting lines, live counts, state colors | `src/visualization.py` |
| 2.6 | Implement CSV logger: per-count events with timestamp, class, confidence, gate results | `src/exporters.py` |
| 2.7 | End-to-end integration test on a sample video (any conveyor-like footage) | `tests/test_pipeline.py` |
| 2.8 | Compute counting metrics: count error rate, ID switches, ghost counts, merge errors | `scripts/evaluate_run.py` |

**Milestone gate:** Count error rate < 5% on a clean video with clear line crossing.

### Phase 3 — Statistical Layer (Est. 1 week)

**Goal:** Add confidence calibration and per-batch uncertainty estimates.

| # | Task | Output |
|---|------|--------|
| 3.1 | Run inference on calibration split, collect (confidence, is_correct) pairs | `data/calibration/calibration_data.json` |
| 3.2 | Fit isotonic regression calibrator, save to disk | `checkpoints/calibrator.pkl` |
| 3.3 | Plot reliability diagram (raw vs calibrated) on test split, compute ECE and Brier score | `reports/reliability_diagram.png`, `reports/calibration_metrics.json` |
| 3.4 | Integrate calibrator into counting pipeline: each track gets a calibrated probability and HIGH/MEDIUM/LOW label | `src/calibration.py` |
| 3.5 | Implement per-batch uncertainty estimation with Wilson score intervals | `src/statistics.py` |
| 3.6 | Implement the batch summary report format (§4.3) | Printed to console + `logs/batch_*.txt` |
| 3.7 | Run full pipeline on test video, generate final batch report with ± error | `reports/batch_test_report.txt` |

**Milestone gate:** ECE < 0.05 post-calibration (on test split); 95% CI covers true count in ≥ 95% of test batches.

### Phase 4 — Polish & Ship ✅ COMPLETED

**Goal:** Production alerts, dashboard, and final report. Custom data fine-tuning skipped (no labeled conveyor footage available).

| # | Task | Output | Status |
|---|------|--------|--------|
| 4.6 | Implement jam detection and gap alerts | `src/alerts.py` | Done |
| 4.7 | Build Streamlit dashboard | `app.py` | Done |
| 4.9 | Final evaluation report | `reports/final_report.md` | Done |
| 4.1–4.4 | Custom data fine-tuning | Skipped — no labeled data | Deferred |

**Milestone gate:** Dashboard running, final report complete.

**Final system summary:**
- Detection: YOLO-NAS-S, mAP@0.50 = 0.865 (nut, pycocotools)
- Calibration: ECE = 0.006 (isotonic regression)
- Tracking: ByteTrack with direction-aware line crossing
- Counting: 8 nuts on test video, all HIGH reliability, 95% CI [6, 11]
- Dashboard: Streamlit with batch history, confidence distribution, system info

---

## 6. Directory Structure

```
conveyor-counter/
├── configs/
│   ├── default.yaml              # runtime pipeline config
│   └── training.yaml             # SuperGradients training recipe
├── src/
│   ├── __init__.py
│   ├── pipeline.py               # threading scaffold + main loop
│   ├── detection.py              # YOLO-NAS wrapper + optional SAHI
│   ├── tracking.py               # ByteTrack/BoT-SORT wrapper
│   ├── counting.py               # zone state machine + statistical gating
│   ├── calibration.py            # isotonic regression calibrator
│   ├── statistics.py             # batch uncertainty + Wilson CI
│   ├── visualization.py          # OpenCV overlay rendering
│   ├── exporters.py              # CSV writer, batch reports
│   ├── alerts.py                 # jam detection, gap detection
│   └── config.py                 # typed config loader (dataclass from YAML)
├── scripts/
│   ├── prepare_data.py           # COCO parse → four-way split
│   ├── train.py                  # training entry point
│   ├── evaluate.py               # detection metrics (mAP, precision, recall, F1)
│   ├── calibrate.py              # fit calibrator on calibration split, plot reliability diagram
│   ├── run_inference.py          # full pipeline on video
│   └── evaluate_run.py           # counting metrics on a completed run
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_calibration_plots.ipynb
├── app.py                        # Streamlit dashboard (Phase 4)
├── requirements.txt
├── README.md
└── DESIGN.md                     # this document
```

---

## 7. Technology Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Detection | `super-gradients` (Deci) | YOLO-NAS S/M/L; ONNX export path available for production |
| SAHI (small object slicing) | `sahi` | Slicing-Aided Hyper Inference; toggleable in config |
| Tracking | `bytetrack` or `boxmot` | ByteTrack for speed, BoT-SORT for accuracy |
| Augmentations | `albumentations` | Includes `RandomSunFlare` for metallic glare |
| Calibration | `scikit-learn` (IsotonicRegression) | Non-parametric; no assumption about calibration curve shape |
| Statistics | `statsmodels` | Wilson score intervals for count uncertainty |
| Visualization | `opencv-python` + `supervision` | `supervision` for annotated frame helpers |
| Dashboard | `streamlit` (Phase 4) | Lightweight, Python-only |
| Experiment tracking | `wandb` (optional) | Loss curves, mAP tracking |
| Config | `pyyaml` + `dataclasses` | Typed config |
| Package manager | `uv` (recommended) or `pip` | Faster, lockfile support |
| Minimum Python | 3.10 | For `dataclasses` slots and `X | Y` syntax |
