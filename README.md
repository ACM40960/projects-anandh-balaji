Skip to content
ACM40960
projects-anandh-balaji
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Security and quality
Insights
Settings
Files
Go to file
t
T
.gitignore
README.md
requirements.txt
test.txt
projects-anandh-balaji
/
README.md
in
main

Edit

Preview
Indent mode

Spaces
Indent size

2
Line wrap mode

Soft wrap
Editing README.md file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
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
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
No file chosen
Attach files by dragging & dropping, selecting or pasting them.
 
