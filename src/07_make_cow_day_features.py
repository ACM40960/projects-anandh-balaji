"""
Week 7 — Sensor fusion.

Takes everything we know about each cow on 7/25 and squashes it into a single row
per cow: the vision side (how it spent its day + a body-condition guess) plus the
wearable-sensor side (movement, temperature, position). That one-row-per-cow table
is what the Week 8 alert rules run on.

A few things to keep in mind while reading this:
  - Only C01–C10 wore sensors. C11–C16 are camera-only, so their sensor columns
    come out as NaN (handled by the has_wearable flag, not by crashing).
  - Milk yield is deliberately left out. We only have one day, and milk only means
    anything as a trend against a cow's own baseline — one number tells us nothing.
  - Everything is pure pandas. No GPU, no torch, so it just runs locally in seconds.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

# outputs/ lives at the repo root (this script is in src/), so resolve relative to here
OUT_DIR   = str(Path(__file__).resolve().parent.parent / 'outputs')
# The MmCows wearable-sensor data is external (too large for the repo). Point this at
# your local copy of sensor_data/main_data.
SENS      = 'D:/D2/sensor_data/sensor_data/main_data'
BEH_DAILY = f'{OUT_DIR}/behavior_daily.parquet'   # from Week 5
BCS_DAILY = f'{OUT_DIR}/bcs_daily.parquet'         # from Week 6

DATE       = '2023-07-25'
DAY_START  = 1690261200                            # midnight 7/25 CDT, in Unix seconds
DAY_END    = DAY_START + 86400                     # +24h
GRAVITY    = 9.81                                   # m/s^2 — a still cow's IMU reads ~this
WEARABLE   = [f'C{i:02d}' for i in range(1, 11)]   # C01–C10 have tags; note T0n folder == C0n cow

# ---------- vision side: behaviour budget + BCS, all 16 cows ----------
# These two parquets already cover every cow, so a left-merge keeps all 16 even
# though the sensor joins below only add data for the 10 wearable ones.
beh = pd.read_parquet(BEH_DAILY)
bcs = pd.read_parquet(BCS_DAILY)[['cow_id', 'bcs_estimate']]
vision = beh.merge(bcs, on='cow_id', how='left')
print('vision rows:', len(vision))


# ---------- IMU: how much the cow moved ----------
def imu_features(cow):
    # sensor folders use the T-prefix, so C03 -> T03 etc.
    t = 'T' + cow[1:]
    f = f'{SENS}/immu/{t}/{t}_0725.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    # Total acceleration magnitude. When the cow is still this hovers around gravity
    # (~9.81); real movement pushes it above/below. So the *spread* of this signal,
    # not its mean, is what tells activity apart from rest.
    mag = np.sqrt(d['accel_x_mps2']**2 + d['accel_y_mps2']**2 + d['accel_z_mps2']**2)
    dev = (mag - GRAVITY).abs()
    return {
        'imu_accel_mean':   float(mag.mean()),
        'imu_accel_std':    float(mag.std()),          # our main "how busy was she" number
        'imu_active_frac':  float((dev > 0.5).mean()),  # share of the day clearly moving
    }


# ---------- CBT: core body temperature ----------
def cbt_features(cow):
    f = f'{SENS}/cbt/{cow}.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    # Unlike IMU/UWB, the CBT file holds all 14 days in one go, so trim to just 7/25.
    d = d[(d['timestamp'] >= DAY_START) & (d['timestamp'] < DAY_END)]
    # C09 had a 21.9 C reading — obviously a sensor dropout (a live cow can't be that
    # cold), and it wrecked the std. Toss anything outside a plausible cow range.
    d = d[(d['temperature_C'] >= 35.0) & (d['temperature_C'] <= 42.0)]
    if len(d) == 0:
        return None
    return {
        'cbt_mean': float(d['temperature_C'].mean()),
        'cbt_max':  float(d['temperature_C'].max()),
        'cbt_min':  float(d['temperature_C'].min()),
        'cbt_std':  float(d['temperature_C'].std()),
    }


# ---------- UWB: where the cow went ----------
def uwb_features(cow):
    t = 'T' + cow[1:]
    f = f'{SENS}/uwb/{t}/{t}_0725.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).sort_values('timestamp')   # sort first or the diffs are nonsense
    # Distance walked = sum of straight-line hops between consecutive positions.
    # Coordinates are in cm, so divide by 100 to report metres.
    step = np.sqrt(d['coord_x_cm'].diff()**2 + d['coord_y_cm'].diff()**2)
    return {
        'uwb_total_dist_m':  float(step.sum() / 100.0),
        'uwb_x_range_m':     float((d['coord_x_cm'].max() - d['coord_x_cm'].min()) / 100.0),
        'uwb_y_range_m':     float((d['coord_y_cm'].max() - d['coord_y_cm'].min()) / 100.0),
    }


# ---------- stitch it all together, one row per cow ----------
rows = []
for _, v in vision.iterrows():
    cow = v['cow_id']
    row = v.to_dict()
    row['has_wearable'] = cow in WEARABLE
    # For camera-only cows we simply skip the sensor extractors — their columns stay
    # absent here and land as NaN once this becomes a DataFrame.
    for extractor in (imu_features, cbt_features, uwb_features):
        feats = extractor(cow) if cow in WEARABLE else None
        if feats:
            row.update(feats)
    rows.append(row)

features = pd.DataFrame(rows)

# Put the human-friendly columns first so the printed table reads nicely.
front = ['cow_id', 'date', 'has_wearable',
         'frac_lying', 'frac_standing', 'frac_feeding', 'frac_moving',
         'bcs_estimate', 'n_observations']
cols = front + [c for c in features.columns if c not in front]
features = features[cols].sort_values('cow_id').reset_index(drop=True)

out = f'{OUT_DIR}/cow_day_features.parquet'
features.to_parquet(out, index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
print(features.to_string())
print('\nSAVED', out)
print('shape:', features.shape, '| wearable cows:', int(features['has_wearable'].sum()))
