"""
BóSight Week 8: Health Alert Screening

Turns the fused per-cow-day feature vector (Week 7) into a health status —
healthy / suspect / critical — for each cow.

MmCows contains no health-event labels within the deployment window (its vet records are
lifetime histories, with no medically relevant event near the annotated day), so there is
no ground truth to train a classifier against. This stage is therefore a rule-based,
veterinary-knowledge screening system rather than a trained/validated model: thresholds
are literature-grounded but unvalidated, so precision/recall are not reported.

Pipeline:
  1. Load cow_day_features (Week 7)
  2. Score each cow against seven veterinary-threshold flags, accumulating a severity score
  3. Map the score to an alert level (>=3 critical, >=1 suspect, 0 healthy)
  4. Save alerts_daily.parquet (schema per two_person_plan section 2.5)

Results (2023-07-25): 12 healthy, 3 suspect, 1 critical.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

# outputs/ lives at the repo root (this script is in src/), so resolve relative to here
OUT_DIR  = str(Path(__file__).resolve().parent.parent / 'outputs')
FEATURES = f'{OUT_DIR}/cow_day_features.parquet'   # Week 7 output
DATE     = '2023-07-25'

df = pd.read_parquet(FEATURES)

# Herd reference for anomaly detection. Only the 10 wearable cows have IMU, so the
# activity z-score is computed over that group only.
wear = df[df['has_wearable']].copy()
def z(series, value):
    mu, sd = series.mean(), series.std()
    return (value - mu) / sd if sd and not np.isnan(sd) else 0.0

# Veterinary thresholds (the interpretable "knobs" of the screening system).
CBT_FEVER     = 39.5   # C: fever / heat-stress territory (data guide section 6.3)
CBT_HIGH_MEAN = 39.0   # C: warm all day; milder concern than a single spike
BCS_LOW       = 2.2    # low end of the 2.0-4.0 proxy scale
LYING_HIGH    = 0.60   # lying >60% of the day: possible lameness/illness
FEEDING_LOW   = 0.15   # feeding <15% of the day: reduced appetite


def assess(row):
    """Score one cow against the seven flags.
    Returns (alert_level, list_of_reasons, severity_score). Severity accumulates:
    0 = healthy, 1-2 = suspect, 3+ = critical. A genuine fever alone reaches critical.
    """
    reasons = []
    severity = 0

    # Temperature — a fever is the single most trustworthy illness signal, so it alone
    # is weighted enough (+3) to reach critical.
    if row['has_wearable'] and not np.isnan(row.get('cbt_max', np.nan)):
        if row['cbt_max'] >= CBT_FEVER:
            reasons.append(f"cbt_elevated (max {row['cbt_max']:.1f}C)"); severity += 3
        elif row['cbt_mean'] >= CBT_HIGH_MEAN:
            reasons.append(f"cbt_high_mean ({row['cbt_mean']:.1f}C)"); severity += 1

    # Activity — abnormally low suggests lethargy (weighted higher); abnormally high can
    # indicate restlessness or estrus (weighted lower, as it is often benign).
    if row['has_wearable'] and not np.isnan(row.get('imu_active_frac', np.nan)):
        za = z(wear['imu_active_frac'], row['imu_active_frac'])
        if za <= -1.5:
            reasons.append("low_activity (lethargy)"); severity += 2
        elif za >= 2.0:
            reasons.append("high_activity (restless/estrus)"); severity += 1

    # Behaviour flags apply to every cow (including vision-only ones), which is why a
    # cow without wearables can still be flagged.
    if row['frac_lying'] >= LYING_HIGH:
        reasons.append(f"excessive_lying ({row['frac_lying']*100:.0f}%)"); severity += 1
    if row['frac_feeding'] <= FEEDING_LOW:
        reasons.append(f"reduced_feeding ({row['frac_feeding']*100:.0f}%)"); severity += 2

    # Body condition — a light contribution, since this is only a proxy, not a clinical BCS.
    if row['bcs_estimate'] <= BCS_LOW:
        reasons.append(f"low_body_condition ({row['bcs_estimate']:.1f})"); severity += 1

    # Map the accumulated score to a traffic-light level.
    if severity >= 3:
        level = 'critical'
    elif severity >= 1:
        level = 'suspect'
    else:
        level = 'healthy'
    return level, reasons, severity


rows = []
for _, r in df.iterrows():
    level, reasons, sev = assess(r)
    rows.append({
        'cow_id': r['cow_id'],
        'date': DATE,
        'alert_level': level,
        'severity_score': sev,
        'reasons': reasons,
        'has_wearable': bool(r['has_wearable']),
        # Carry a few raw values through so the dashboard need not reopen the Week 7 table.
        'cbt_max':         r.get('cbt_max', np.nan),
        'imu_active_frac': r.get('imu_active_frac', np.nan),
        'frac_lying':      r['frac_lying'],
        'frac_feeding':    r['frac_feeding'],
        'bcs_estimate':    r['bcs_estimate'],
    })

# Sort worst-first, the natural reading order for a health list.
alerts = pd.DataFrame(rows).sort_values(
    ['severity_score', 'cow_id'], ascending=[False, True]).reset_index(drop=True)

# Parquet cannot store a column of Python lists, so flatten reasons to a string.
alerts_out = alerts.copy()
alerts_out['reasons'] = alerts_out['reasons'].apply(lambda L: '; '.join(L) if L else '')
out = f'{OUT_DIR}/alerts_daily.parquet'
alerts_out.to_parquet(out, index=False)

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)
print(alerts_out[['cow_id', 'alert_level', 'severity_score', 'reasons']].to_string())
print()
print('alert level counts:', dict(alerts_out['alert_level'].value_counts()))
print('SAVED', out)
