[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23935060&assignment_repo_type=AssignmentRepo)

# BóSight — Multimodal Dairy Cow Health Monitoring

An automated monitoring system for dairy-cow health and behaviour, built on the
**MmCows** multimodal dataset (NeurIPS 2024). A multi-stage deep-learning pipeline
turns overhead barn-camera frames + wearable-sensor streams into per-cow behaviour,
body-condition estimates, and health alerts, presented in an interactive dashboard.

## Pipeline

```
frame → [1] detect → [2] re-ID → [3] track → [4] behaviour → [5a] BCS ┐
                                                     └──────→ [5b] sensor fusion → alert → dashboard
```

| Stage | Description | Metric |
|---|---|---|
| 1. Detection | YOLOv8 cow detector | mAP@0.5 = 0.992 |
| 2. Re-identification | ResNet-50, 16 identities | 97% accuracy |
| 3. Tracking | Detection + re-ID per frame | 98.4% accuracy |
| 4. Behaviour | ResNet-50, 4-class (lying/standing/feeding/moving) | 95.0% acc, 0.807 macro-F1 |
| 5a. Body condition | Geometric proxy (no ground truth) | relative ranking |
| 5b. Health alerts | Rule-based screening over fused features | healthy / suspect / critical |

## Repository layout

```
├── week5_behaviour_classifier.ipynb        # Week 5 — behaviour classifier (local variant)
├── _bosight_week5_kaggle_upd_.ipynb        # Week 5 — behaviour classifier (Kaggle/GPU)
├── week6_bcs_KAGGLE.ipynb                   # Week 6 — body-condition-score proxy
├── week7_make_cow_day_features.py          # Week 7 — multimodal sensor fusion
├── week8_make_alerts.py                    # Week 8 — rule-based health alerts
├── app.py                                  # Week 9 — Streamlit dashboard
├── report/                                 # evaluation figures + writeups
├── outputs/                                # pipeline result tables + dashboard screenshots
├── KAGGLE_GUIDE.md                         # how to run training on Kaggle
└── requirements.txt
```

Large artefacts (trained model weights, raw crop images, the Python venv) are excluded
from git — the notebooks regenerate them.

## Running the dashboard

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads the precomputed parquet files in `outputs/` — no GPU or model
inference needed to view it.

## Dataset notes

- **4-class behaviour mapping:** MmCows codes 1→moving, 2→standing, 3/4→feeding, 7→lying; codes 0/5/6 excluded.
- **Milk yield excluded** from the health pipeline: only one annotated day exists, so there is no multi-day baseline for a yield-trend signal.
- **Wearables cover C01–C10 only** (IMU/CBT/UWB); C11–C16 are camera-only.
- **Timestamps** are Unix epoch in CDT (UTC−5); integer match across all modalities.

## Team

Two-person project: one half owns the vision front-end (detection, re-identification,
tracking); the other owns behaviour classification, body condition, sensor fusion, and
the dashboard.
