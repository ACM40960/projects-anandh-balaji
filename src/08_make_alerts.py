"""
Week 8 — Health alerts.

Important caveat up front: MmCows has no health-event labels during our study window
(I checked — the vet records are lifetime histories, and nothing illness-related lands
on or near 7/25). So there's nothing to train a classifier against. Instead this is a
rule-based screening system: take the fused feature vector from Week 7, run each cow
past a handful of vet-sensible thresholds, and add up a severity score.

It's honest to call this a "screening demo", not a validated model — the thresholds are
grounded in the literature but we can't report precision/recall without ground truth.

Output: alerts_daily.parquet (schema follows two_person_plan §2.5).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

# outputs/ lives at the repo root (this script is in src/), so resolve relative to here
OUT_DIR  = str(Path(__file__).resolve().parent.parent / 'outputs')
FEATURES = f'{OUT_DIR}/cow_day_features.parquet'   # the one file Week 7 produced
DATE     = '2023-07-25'

df = pd.read_parquet(FEATURES)

# For "is this cow unusual?" we compare against the herd. Only the 10 wearable cows
# have IMU, so the activity z-score is computed over that group only.
wear = df[df['has_wearable']].copy()
def z(series, value):
    mu, sd = series.mean(), series.std()
    return (value - mu) / sd if sd and not np.isnan(sd) else 0.0

# (kept for reference / possible future behaviour z-scores — all 16 cows have behaviour)
beh_ref = {c: (df[c].mean(), df[c].std()) for c in
           ['frac_lying', 'frac_standing', 'frac_feeding', 'frac_moving']}

# ---- the thresholds. these are the knobs a vet would recognise ----
CBT_FEVER      = 39.5   # C — fever / heat-stress territory (see data guide §6.3)
CBT_HIGH_MEAN  = 39.0   # C — warm all day, milder concern than a single spike
BCS_LOW        = 2.2    # on our 2–4 proxy scale, this is the thin end
LYING_HIGH     = 0.60   # lying >60% of the day is a lot — possible lameness/illness
FEEDING_LOW    = 0.15   # eating <15% of the day — off her feed


def assess(row):
    """Score one cow. Returns (alert_level, list_of_reasons, severity_score)."""
    reasons = []
    severity = 0   # 0 = healthy, 1–2 = suspect, 3+ (or a fever) = critical

    # Temperature first — a genuine fever is the single most trustworthy sick signal,
    # so it alone is enough to push a cow to critical (+3).
    if row['has_wearable'] and not np.isnan(row.get('cbt_max', np.nan)):
        if row['cbt_max'] >= CBT_FEVER:
            reasons.append(f"cbt_elevated (max {row['cbt_max']:.1f}C)"); severity += 3
        elif row['cbt_mean'] >= CBT_HIGH_MEAN:
            reasons.append(f"cbt_high_mean ({row['cbt_mean']:.1f}C)"); severity += 1

    # Movement. Too little = lethargy (a real concern). Too much can mean estrus or
    # restlessness — worth noting but less alarming, so it scores lower.
    if row['has_wearable'] and not np.isnan(row.get('imu_active_frac', np.nan)):
        za = z(wear['imu_active_frac'], row['imu_active_frac'])
        if za <= -1.5:
            reasons.append("low_activity (lethargy)"); severity += 2
        elif za >= 2.0:
            reasons.append("high_activity (restless/estrus)"); severity += 1

    # Behaviour flags work for every cow (camera-only ones included), which is why
    # a vision-only cow like C11 can still be flagged.
    if row['frac_lying'] >= LYING_HIGH:
        reasons.append(f"excessive_lying ({row['frac_lying']*100:.0f}%)"); severity += 1
    if row['frac_feeding'] <= FEEDING_LOW:
        reasons.append(f"reduced_feeding ({row['frac_feeding']*100:.0f}%)"); severity += 2

    # Body condition — a gentle nudge, since it's only a proxy, not a real BCS.
    if row['bcs_estimate'] <= BCS_LOW:
        reasons.append(f"low_body_condition ({row['bcs_estimate']:.1f})"); severity += 1

    # Turn the running score into a traffic light.
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
        # carry a few raw numbers through so the dashboard doesn't have to reopen Week 7
        'cbt_max': r.get('cbt_max', np.nan),
        'imu_active_frac': r.get('imu_active_frac', np.nan),
        'frac_lying': r['frac_lying'],
        'frac_feeding': r['frac_feeding'],
        'bcs_estimate': r['bcs_estimate'],
    })

# Worst cows first — that's the order you'd want to read them in.
alerts = pd.DataFrame(rows).sort_values(
    ['severity_score', 'cow_id'], ascending=[False, True]).reset_index(drop=True)

# Parquet doesn't love a column of python lists, so flatten reasons to a string.
alerts_out = alerts.copy()
alerts_out['reasons'] = alerts_out['reasons'].apply(lambda L: '; '.join(L) if L else '')
out = f'{OUT_DIR}/alerts_daily.parquet'
alerts_out.to_parquet(out, index=False)

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)
print(alerts_out[['cow_id', 'alert_level', 'severity_score', 'reasons']].to_string())
print()
print('alert level counts:', dict(alerts_out['alert_level'].value_counts()))
print('SAVED', out)
