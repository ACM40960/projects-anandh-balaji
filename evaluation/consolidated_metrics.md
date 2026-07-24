# BóSight — Consolidated Evaluation Summary (all 5 stages)

A single place pulling together the quantitative results across the whole pipeline,
with an explicit statement of which stages have ground truth and which do not.

_Data: MmCows, annotated day 2023-07-25, camera cam_1, 16 Holstein-Friesian cows._

---

## 1. Results at a glance

| Stage | Model | Metric | Result | Ground truth? |
|---|---|---|---|---|
| 1. Detection | YOLOv8s | mAP@0.5 | **0.992** | ✅ manual bboxes |
| 2. Re-identification | ResNet-50 (16-class) | Test accuracy | **97.0%** | ✅ manual cow IDs |
| 3. Tracking | Detection + re-ID | Overall accuracy | **98.4%** | ✅ derived from GT boxes |
| 4. Behaviour | ResNet-50 (4-class) | Accuracy / macro-F1 | **95.0% / 0.807** | ✅ 1 Hz behaviour labels |
| 5a. BCS | bbox-size proxy | — (relative ranking) | qualitative | ❌ none in MmCows |
| 5b. Health alerts | rule-based screening | — (12/3/1 split) | qualitative | ❌ no in-window events |

**The three vision stages and the behaviour classifier are metric-backed. BCS and health
alerts are demonstrative** — MmCows contains no BCS scores and no health events during the
deployment window, so no MAE / RMSE / precision / recall can be reported for those two.
This is a dataset limitation, stated openly rather than worked around.

---

## 2. Stage detail

### Stage 1 — Detection (YOLOv8s)
- Fine-tuned from COCO weights on cam_1 7/25 frames (single "cow" class).
- **mAP@0.5 = 0.992.** Detection rate at IoU ≥ 0.5 in the tracking eval: 99.29%.

### Stage 2 — Re-identification (ResNet-50, 16-class)
- Coat-pattern classifier over the 16 cow identities.
- **Test accuracy = 97.0%.** ID accuracy of matched detections (tracking eval): 99.07%.

### Stage 3 — Tracking (appearance-based)
- Frame-gap of ~17 s makes motion trackers unsuitable; identity is assigned per-frame by re-ID.
- On 5,040 frames / 54,657 detections: **overall accuracy 98.4%** (correct-and-identified / GT boxes).

### Stage 4 — Behaviour classification (ResNet-50, 4-class)
Test set = 8,736 crops. Overall **accuracy 95.0%**, **macro-F1 0.807**, weighted-F1 0.951
(best validation macro-F1 0.819).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| lying | 0.997 | 0.996 | **0.997** | 3389 |
| feeding | 0.941 | 0.978 | **0.959** | 2108 |
| standing | 0.950 | 0.916 | **0.933** | 3054 |
| moving | 0.315 | 0.368 | **0.339** | 185 |

- Three of four classes exceed 0.93 F1.
- **`moving` is weak (0.34)** — only ~1.5% of samples, and locomotion is a temporal behaviour
  poorly captured in single still frames. Complemented by IMU activity in the fusion stage.

### Stage 5a — Body condition (proxy)
- Median standing-frame bbox area per cow, min-max scaled to a conservative 2.0–4.0 band.
- Produces a **relative ranking** (e.g. C10 highest, C07 lowest), not a clinical BCS.
- No ground truth → no MAE/R². Limitation: perspective confound from the isometric camera.

### Stage 5b — Health alerts (rule-based screening)
- Veterinary-threshold severity scoring over the fused feature vector (behaviour + BCS + IMU + CBT + UWB).
- Output on 7/25: **12 healthy, 3 suspect (C01, C07, C08), 1 critical (C11)**.
- No in-window health events → thresholds are literature-grounded but **unvalidated**;
  no precision/recall. Vision-only cows (C11–C16) are screened on behaviour + BCS alone.

---

## 3. Data-scope decisions affecting evaluation

- **Milk yield excluded** from fusion/alerts: single-day data has no baseline for trend detection.
- **Only C01–C10 have wearables** → full multimodal feature vector exists for 10 of 16 cows.
- **Only 7/25 has visual annotations** → behaviour/detection metrics are single-day.

---

## 4. One-paragraph summary (for the abstract / conclusion)

> The BóSight pipeline achieves strong quantitative performance on its supervised stages:
> 0.992 mAP@0.5 detection, 97% re-identification accuracy, 98.4% tracking accuracy, and
> 95.0% accuracy (0.807 macro-F1) on four-class behaviour classification. Body-condition
> scoring and health alerting are implemented as a geometric proxy and a veterinary-knowledge
> rule-based screening system respectively, because the MmCows dataset provides no ground
> truth for either; these stages are evaluated qualitatively and their limitations documented.
