# BóSight — Future Work and Research Directions

This document outlines how the BóSight pipeline could be extended beyond the constraints of
the single-day MmCows evaluation. Each direction follows directly from a limitation observed
during the project rather than being speculative — see `consolidated_metrics.md` for the
results these build on.

---

## 1. Re-introduce milk yield with multi-day data

Milk yield was deliberately excluded from the health-alert stage. On a single observation day
there is no per-cow baseline, so a raw daily yield carries little information, and cross-cow
comparison confounds genuine health signals with natural variation (age, lactation stage,
breed). With continuous multi-day monitoring, **milk-yield trend** — a drop relative to a
cow's own rolling baseline — becomes one of the strongest early indicators of illness,
metabolic stress, and lameness, and should be re-introduced as a fused feature.

## 2. Replace the BCS proxy with a ground-truth-trained model

Body condition is currently a **geometric proxy** (bounding-box area of standing cows), scaled
to a relative 2.0–4.0 band, because MmCows contains no body-condition labels. This yields only
a relative ranking and is confounded by the isometric camera perspective. Two upgrades:

- **Calibrated regression** against real body-weight or vet-assigned BCS scores, giving an
  absolute, validated estimate (with MAE / R² instead of a ranking).
- **3D / depth-based body measurement** (point clouds) to derive biometric stature features
  that are robust to viewing angle — the standard approach in modern livestock weight
  estimation.

## 3. Move from rule-based alerts to a supervised health model

The health-alert stage is an interpretable **rule-based screening system** because the dataset
has no health-event labels within the deployment window. Its veterinary-threshold weights are
literature-grounded but unvalidated, so no precision/recall can be reported. With a dataset
that includes recorded health events (treatments, diagnoses, lameness scores):

- Train a **supervised classifier** (e.g. gradient-boosted trees) on the fused per-cow-day
  features, enabling proper precision/recall/AUC evaluation.
- Keep the rule-based system as a **transparent baseline** and an explainability layer
  alongside the learned model.

## 4. Add temporal modelling for movement behaviours

The behaviour classifier reaches 0.807 macro-F1, but the `moving` class is weak (F1 ≈ 0.34)
because locomotion is a **temporal** behaviour that a single still frame cannot capture — a
walking cow looks almost identical to a standing one in one image. Extensions:

- **Short video clips** with temporal models (3D CNNs, ConvLSTM, or transformer-based video
  classifiers) instead of single-frame classification.
- **Fusing the IMU accelerometer signal**, which measures motion directly and is strongest
  exactly where vision is weakest — a concrete demonstration of complementary modalities.

## 5. Expand the behavioural ethogram

The current four classes (lying / standing / feeding / moving) are an operational grouping of
the dataset's finer annotations. A richer ethogram — including health-relevant behaviours such
as isolation, reduced rumination, or abnormal posture — would increase the clinical value of
the behaviour budget, at the cost of needing more labelled examples per (often rare) class.

## 6. Scale beyond a single day, camera, and site

MmCows provides one fully-annotated day from one camera view. Robust deployment requires:

- **Multi-day** aggregation for trend-based signals (feeds directly into §1).
- **Multi-camera** fusion to reduce occlusion and recover cows missing from any single view.
- **Multi-site validation** across barns with different layouts, lighting, and breed mixes, to
  test generalisation of the detector, re-identification model, and behaviour classifier.

## 7. Toward real-time, on-farm operation

The current pipeline runs offline over pre-collected frames. A production system would need
near-real-time inference (edge or streaming), automated identity maintenance over long
horizons (handling appearance drift as animals grow or moult), and integration of the
dashboard's decision-support view into farmers' existing herd-management workflows.

---

## Summary

BóSight demonstrates that a full detect → identify → track → behaviour → condition → alert
pipeline can be assembled and made interpretable end-to-end. The clearest path to impact is
**better ground truth** (real weights and health events, unlocking §2 and §3) and **temporal
data** (unlocking §1 and §4) — precisely the data a continuous, multi-day, on-farm deployment
would provide.
