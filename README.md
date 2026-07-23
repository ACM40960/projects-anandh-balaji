[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23935060&assignment_repo_type=AssignmentRepo)

# BóSight — Multimodal Monitoring of Dairy Cow Health and Behaviour

*MSc in Data & Computational Science — University College Dublin*

> *"Bó"* is the Irish word for cow. BóSight combines computer vision and wearable-sensor
> data to give farmers automated "sight" into the health and behaviour of a dairy herd.

---

## 1. Problem statement

Modern dairy farming operates at a scale where continuous manual observation of every
animal is impractical. Yet many health problems — lameness, metabolic stress, illness,
reduced feeding — first show up as subtle, gradual changes in an individual cow's
behaviour and physiology. These early signs are easily missed, and by the time a cow is
visibly unwell, welfare and milk yield have often already suffered.

Precision Livestock Farming (PLF) aims to close this gap using sensing and machine
learning. The challenge is that no single data source tells the whole story: cameras
capture what a cow *does* but not its internal state, while wearable sensors capture
movement and temperature but not context. **BóSight** implements and evaluates an
automated, per-animal monitoring pipeline that fuses these complementary signals, and
demonstrates — via case studies — that the fusion catches things no single modality
would catch alone.

## 2. Objectives

BóSight is an end-to-end pipeline that, for each cow in a barn, does the following:

1. **Detect and localise** individual animals in overhead camera footage.
2. **Identify** each animal consistently by its unique coat pattern (re-identification).
3. **Track** identities across a full day's frames.
4. **Classify behaviour** into four daily activities (lying, standing, feeding, moving).
5. **Estimate body condition** as a proxy indicator of nutritional state.
6. **Fuse vision with wearable-sensor data** (motion, body temperature, location) to
   flag animals that may need attention, via a rule-based health screening layer.
7. **Present** the results in an interactive Streamlit dashboard.

## 3. Dataset

The project uses **MmCows** (Vu et al., *NeurIPS 2024*), a publicly available multimodal
dataset from a 14-day deployment with 16 Holstein-Friesian dairy cows in a Wisconsin
barn. It provides synchronised overhead camera footage (cam_1, ~5,040 frames), manual
behaviour annotations for one fully-annotated day (2023-07-25), and wearable-sensor
streams (IMU acceleration, core body temperature, UWB positioning) for 10 of the 16
cows. The Holstein-Friesian breed is also the dominant dairy breed in Ireland, making
the work directly relevant to the Irish agricultural context represented in the
VistaMilk research programme.

## 4. Pipeline and results

BóSight is a multi-stage pipeline. Each stage consumes the output of the previous one,
moving from raw pixels to a per-cow health alert.

| Stage | Method | Key result | Ground truth available? |
|---|---|---|---|
| Detection | Fine-tuned YOLOv8s | mAP50 = **0.992** | Yes |
| Re-identification | ResNet-50, 16-class classification | **97.0%** test accuracy | Yes |
| Tracking | Detection + re-ID (appearance-based, no motion tracker) | **98.4%** overall accuracy | Yes |
| Behaviour classification | ResNet-50, 4-class (lying/standing/feeding/moving) | macro-F1 **0.807** (acc 95.0%) | Yes |
| Body condition (BCS) | Bounding-box-area proxy, standing frames only | Relative ranking (2.0–4.0 band) | **No** |
| Sensor fusion + health alerts | Rule-based severity scoring across 7 veterinary-threshold flags | 12 healthy / 3 suspect / 1 critical | **No** |
| Dashboard | Streamlit + Plotly, reads precomputed outputs only | Herd Overview + Per-Cow Detail pages | — |

**Why two stages have no ground truth:** MmCows contains no body-condition-score labels
and no health-event labels within the 14-day deployment window (its health records are
lifetime cow records spanning 2014–2024, with zero medically relevant events near the
annotated day). BCS and health alerts are therefore implemented as **documented,
interpretable proxies** rather than trained/validated models — this is a scope reality
of the dataset, not a shortcut, and is stated explicitly wherever these outputs appear
in the dashboard and report.

Full per-class metrics, the behaviour classifier's confusion matrix, and case-study
walk-throughs of individual cows through the entire pipeline are in `notebooks/10_evaluation.ipynb`.

## 5. Dashboard

The Streamlit dashboard (`app.py`) reads precomputed parquet outputs only — no live
model inference — so it runs instantly with no GPU.

- **Herd Overview:** alert summary, herd behaviour distribution, "cows needing
  attention" panel, sortable/colour-coded All Cows table, CSV export.
- **Per-Cow Detail:** reference photo, 24h behaviour donut, body/sensor metrics with
  proxy/no-ground-truth tooltips, full rule-flag breakdown, herd comparison chart.

Vision-only cows (C11–C16, no wearable sensors) are handled explicitly throughout —
sensor fields are shown as clearly marked "N/A" rather than blank or `NaN`.

```bash
streamlit run app.py
```

## 6. Repository structure

```
bosight/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_yolov8_detection.ipynb
│   ├── 03_reid.ipynb
│   ├── 04_tracking.ipynb
│   ├── 05_behaviour_classifier.ipynb
│   ├── 06_bcs.ipynb
│   └── 10_evaluation.ipynb        # confusion matrix, case studies, consolidated metrics
├── src/
│   ├── week7_make_cow_day_features.py
│   ├── week8_make_alerts.py
│   └── schemas.py
├── models/
│   ├── best.pt                     # YOLOv8s cow detector
│   ├── reid.pt                     # ResNet-50 cow re-ID (16-class)
│   └── behaviour_best.pt           # ResNet-50 behaviour classifier (4-class)
├── outputs/
│   ├── tracked_cows.parquet
│   ├── behaviour_daily.parquet
│   ├── bcs_daily.parquet
│   ├── cow_daily_features.parquet
│   └── alerts_daily.parquet
├── app.py                          # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 7. Limitations

- **Single camera view** (cam_1 only) — occlusion and partial visibility reduce
  detection performance relative to multi-view literature benchmarks.
- **Single annotated day** (2023-07-25) — all quantitative results are for this day;
  milk yield was excluded from sensor fusion because a single day gives no baseline
  for detecting a meaningful trend.
- **BCS is an unvalidated geometric proxy** (bounding-box area), not a calibrated
  clinical score.
- **Health alerts are rule-based**, grounded in veterinary literature thresholds but
  not validated against real diagnoses, since none exist in this dataset's window.
- **Vision-only cows (C11–C16)** are screened on behaviour and BCS alone, with no
  physiological (temperature/activity) confirmation available.
- The **`moving` behaviour class underperforms** (0.34 F1) due to severe class
  imbalance and the fact that locomotion is a temporal signal a single still frame
  cannot reliably capture — addressed conceptually by IMU accelerometer fusion.

## 8. Technologies

Python · PyTorch / torchvision · Ultralytics YOLOv8 · scikit-learn · pandas / NumPy ·
Streamlit · Plotly · Jupyter · Kaggle / Google Colab (GPU training).

See `requirements.txt` for the full pinned dependency list.

## 9. References

Vu, H., Prabhune, O., Raskar, U., Panditharatne, D., Chung, H., Choi, C. Y., & Kim, Y.
(2024). *MmCows: A Multimodal Dataset for Dairy Cattle Monitoring.* Advances in Neural
Information Processing Systems (NeurIPS) 37, 59451–59467.

Yu, R. et al. (2024). *Research on Automatic Recognition of Dairy Cow Daily Behaviors
Based on Deep Learning.* Animals 14(3):458.

Khan, A. et al. (2024). *Development of a real-time cattle lameness detection system
using a single side-view camera.* Scientific Reports 14:13734.

Andrew, W. et al. (2021). *Visual identification of individual Holstein-Friesian cattle
via deep metric learning.* Computers and Electronics in Agriculture 185:106133.

Paulauskaite-Taraseviciene, A. et al. (2026). *AI-Driven Multimodal Sensing for Early
Detection of Health Disorders in Dairy Cows.* Animals 16(3):411.
