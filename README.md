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
movement and temperature but not context. **BóSight** investigates how these
complementary signals can be combined into an automated, per-animal monitoring system.

## 2. Objectives

The project aims to build an end-to-end pipeline that, for each cow in a barn, can:

1. **Detect and localise** individual animals in overhead camera footage.
2. **Identify** each animal consistently by its unique coat pattern (re-identification).
3. **Track** identities across time.
4. **Classify behaviour** into key daily activities (lying, standing, feeding, moving).
5. **Estimate body condition** as an indicator of nutritional state.
6. **Fuse vision with wearable-sensor data** (motion, body temperature, location) to
   flag animals that may need attention.
7. **Present** the results in an interactive dashboard usable by a non-technical farmer.

## 3. Dataset

The project uses **MmCows** (Vu et al., *NeurIPS 2024*), a publicly available multimodal
dataset from a 14-day deployment with a herd of Holstein-Friesian dairy cows. It provides
synchronised overhead camera footage, manual behaviour annotations, and wearable-sensor
streams (inertial motion, core body temperature, and ultra-wideband positioning). The
Holstein-Friesian breed is also the dominant dairy breed in Ireland, making the work
directly relevant to the Irish agricultural context.

## 4. Approach

BóSight is structured as a multi-stage pipeline. Each stage consumes the output of the
previous one, moving from raw pixels to an actionable, per-cow health summary.

| Stage | Goal | Method (planned) |
|---|---|---|
| Detection | Locate every cow in a frame | Fine-tuned object detector (YOLO family) |
| Re-identification | Assign the correct identity to each cow | Deep metric / classification model on coat patterns |
| Tracking | Maintain identity over time | Appearance-based association across frames |
| Behaviour classification | Label each cow's activity | CNN image classifier (transfer learning) |
| Body condition | Estimate nutritional state | Vision-based estimation |
| Sensor fusion & alerts | Combine vision + sensors into a health signal | Feature fusion + screening logic |
| Dashboard | Communicate results | Interactive web app |

## 5. Methodology notes

- **Behaviour categories** are derived from the dataset's fine-grained annotations and
  grouped into four operationally meaningful classes (lying, standing, feeding, moving).
- **Class imbalance** is expected (resting behaviours dominate) and will be addressed
  during training.
- **Sensor alignment** relies on the dataset's shared timestamp convention, enabling an
  exact join between visual and wearable data per animal.
- **Scope decisions** (e.g. which modalities are informative given the available data)
  are documented as the project progresses.

## 6. Technologies

Python · PyTorch / torchvision · scikit-learn · pandas / NumPy · Streamlit ·
Jupyter · Kaggle / Google Colab (GPU training).

## 7. Project organisation

This is a two-person project. The work is split into a **vision front-end** (detection,
re-identification, tracking) and a **behaviour, body-condition, sensor-fusion and
dashboard** component, with joint integration and evaluation.

## 8. References

Vu, H., Prabhune, O., Raskar, U., Panditharatne, D., Chung, H., Choi, C. Y., & Kim, Y.
(2024). *MmCows: A Multimodal Dataset for Dairy Cattle Monitoring.* Advances in Neural
Information Processing Systems (NeurIPS) 37.
