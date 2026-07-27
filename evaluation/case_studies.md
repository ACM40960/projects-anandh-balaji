# BóSight: Case Studies (end-to-end pipeline walk-through)

Three cows tracing the full pipeline: detection, ID, tracking, behaviour, BCS,
sensor fusion, alert, chosen to span the alert spectrum. Data: 2023-07-25, cam_1.

---

## Case 1: C11, 🔴 Critical (behavioural anomaly, vision-only cow)

**Pipeline trace**
- **Detection + ID + tracking:** C11 detected and identified across the day (3,507 crops).
- **Behaviour budget:** lying **64%**, standing 21%, feeding **14%**, moving 1.1%.
- **BCS proxy:** 2.69 (mid-range).
- **Sensors:** none, C11 is a vision-only cow (no wearable).

**Why flagged critical**
Two behavioural rules fire: **excessive lying (≥60%)** [+2 total via feeding] and
**reduced feeding (≤15%)** [+2], giving severity 3, critical.

**Interpretation**
C11 is the most sedentary cow in the herd and eats the least, a combination that, in a
real barn, warrants a physical check (lameness, off-feed, early illness). Note the caveat:
this alert rests on behaviour alone, with no temperature/activity confirmation available.

---

## Case 2: C01, 🟡 Suspect (physiological, lethargy)

**Pipeline trace**
- **Behaviour budget:** lying 39%, standing 36%, feeding 21%, moving 2.9%, unremarkable.
- **BCS proxy:** 2.47.
- **Sensors (wearable):** CBT mean 38.48 °C (max 39.07, normal), **IMU active fraction 15%**
  (lowest activity band of the herd), distance travelled 1,595 m (also lowest).

**Why flagged suspect**
Behaviour looks normal, but the **IMU tells a different story**: activity z-score ≤ -1.5
means the lethargy rule fires (+2), giving suspect.

**Interpretation**
This is the multimodal pipeline earning its keep, a cow that looks behaviourally ordinary
on camera is caught by its wearable showing abnormally low movement. Exactly the case where
sensor fusion adds information vision alone would miss.

---

## Case 3: C07, 🟡 Suspect (low body condition)

**Pipeline trace**
- **Behaviour budget:** lying 42%, standing 22%, feeding 33% (feeds well), moving 2.8%.
- **BCS proxy:** **2.00**, the lowest in the herd.
- **Sensors:** CBT mean 38.43 °C (normal), activity 20% (normal-low), distance 1,794 m.

**Why flagged suspect**
Only the **low body-condition rule** fires (BCS ≤ 2.2, +1), giving suspect.

**Interpretation**
Temperature and activity are fine and the cow feeds actively, but it carries the least
apparent body condition. Over multiple days this would be the cow to watch for weight loss.
Caveat: BCS here is a bbox-size proxy, so this flag is a screening hint, not a diagnosis.

---

## Contrast: C05, 🟢 Healthy (clean baseline)

- Behaviour balanced (lying 36%, standing 40%, feeding 22%, moving 1.9%).
- BCS proxy 3.65 (one of the highest).
- CBT mean 38.54 °C, activity 25%, distance 2,407 m, all mid-to-high, normal.
- **No rules fire, healthy.** A good reference profile for what "normal" looks like.

---

## Summary table

| Cow | Alert | Lying | Feeding | BCS | CBT max | Activity | Fired rule(s) |
|---|---|---|---|---|---|---|---|
| C11 | 🔴 critical | 64% | 14% | 2.69 | n/a | n/a | excessive lying + reduced feeding |
| C01 | 🟡 suspect | 39% | 21% | 2.47 | 39.07 | 15% | lethargy (low activity) |
| C07 | 🟡 suspect | 42% | 33% | 2.00 | 39.18 | 20% | low body condition |
| C05 | 🟢 healthy | 36% | 22% | 3.65 | 39.16 | 25% | none |
