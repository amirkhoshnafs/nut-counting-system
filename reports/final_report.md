# Nut Counting System — Final Evaluation Report

**Date:** 2026-05-12
**Model:** YOLO-NAS-S, trained on combined datasets (Nut Counting + Nuts & Bolts Detection + Nuts & Bolts Segmentation)
**Scope:** Single-class nut detection, tracking, and counting

---

## 1. Detection Performance (pycocotools, test split)

| Metric | Nut | Gate |
|--------|-----|------|
| mAP@0.50 | **0.865** | PASS (≥ 0.85) |
| mAP@0.75 | 0.805 | — |
| mAP@0.50:0.95 | 0.685 | — |
| AR@100 | 0.764 | — |

Trained on 566 unique source images across 3 Roboflow datasets. Bolt class excluded due to 27:1 class imbalance.

---

## 2. Calibration Quality

| Metric | Raw | Calibrated | Gate |
|--------|-----|-----------|------|
| ECE | 0.015 | **0.006** | PASS (< 0.05) |
| Brier Score | 0.044 | 0.043 | — |

Model is well-calibrated. Raw confidences ~0.90 map to ~0.98 calibrated probability.

---

## 3. Counting Performance (cropped_nuts.mp4)

| Metric | Value |
|--------|-------|
| Video | 121 frames, 864×360 |
| Nuts counted | 8 |
| Mean confidence | 0.906 |
| Reliability | HIGH (8/8 events) |
| Expected count | 8 ± 1 |
| 95% CI | [6, 11] |

All counts rated HIGH reliability. No false positives, no ghost counts.

---

## 4. System Architecture

- **Detection:** YOLO-NAS-S (19M params, ONNX-exportable)
- **Tracking:** ByteTrack (supervision 0.27)
- **Counting:** Direction-aware line crossing (left→right at x=432)
- **Calibration:** Isotonic regression (scikit-learn)
- **Statistics:** Wilson score intervals, per-batch uncertainty
- **Dashboard:** Streamlit

---

## 5. What Works

- Real-time nut detection with mAP@0.50 = 0.865
- Reliable line-crossing counting with direction filtering
- Calibrated confidence scores (ECE = 0.006)
- Batch summary reports with ± error and confidence intervals
- Jam detection and gap alerts

## 6. Known Limitations

- Bolt detection not operational (27:1 class imbalance — 18,986 nut vs 686 bolt annotations). Combined nut+bolt mAP@0.50 = 0.451 vs nut-only = 0.865
- Rotating/vibrating nuts not supported (requires consistent conveyor flow for line-crossing counting)
- No instance segmentation for overlapping nuts
- Model trained on static images, not conveyor video frames
- Jam detection is experimental (stalled-track heuristic, not true stationary-object detection)
- Counting accuracy claims limited to 8-count demo video — larger labeled dataset needed for rigorous validation

---

## 7. Next Steps

1. Record balanced nut/bolt conveyor video, label 500 frames
2. Fine-tune YOLO-NAS-M on custom data
3. Add instance segmentation for overlap resolution
4. Deploy with ONNX Runtime / TensorRT for production FPS
