"""
BóSight: Sensor Fusion

Combines the vision-derived outputs (behaviour time-budget from Week 5 and the BCS proxy
from Week 6) with the wearable-sensor streams into a single feature vector per cow per day.
This per-cow-day table is the input to the Week 8 health-alert stage.

Pipeline:
  1. Load the per-cow behaviour budget (behavior_daily) and BCS proxy (bcs_daily)
  2. For each wearable cow (C01-C10), derive features for 2023-07-25 from:
       - IMU  : acceleration-magnitude statistics and an activity fraction
       - CBT  : core body temperature mean/max/min/std
       - UWB  : total distance travelled and spatial range
  3. Merge vision + sensor features into one row per cow
  4. Save cow_day_features.parquet

Notes:
  - Milk yield is excluded: a single observation day has no baseline for a yield trend.
  - Only C01-C10 wore sensors; C11-C16 are vision-only and their sensor columns are NaN
    (marked by the has_wearable flag rather than dropped).
  - Pure pandas, no GPU. SENS points at the external MmCows sensor data (not in the repo);
    update it to your local copy.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

# outputs/ lives at the repo root (this script is in src/), so resolve relative to here
OUT_DIR   = str(Path(__file__).resolve().parent.parent / 'outputs')
SENS      = 'D:/D2/sensor_data/sensor_data/main_data'   # external sensor data; update to your path
BEH_DAILY = f'{OUT_DIR}/behavior_daily.parquet'         # Week 5 output
BCS_DAILY = f'{OUT_DIR}/bcs_daily.parquet'              # Week 6 output

DATE      = '2023-07-25'
DAY_START = 1690261200                            # midnight 2023-07-25 CDT, Unix seconds
DAY_END   = DAY_START + 86400                     # +24h
GRAVITY   = 9.81                                  # m/s^2; a stationary IMU reads ~this
WEARABLE  = [f'C{i:02d}' for i in range(1, 11)]   # C01-C10 wore tags (sensor folder T0n == cow C0n)

# Vision features for all 16 cows.
# Both parquets already cover every cow, so a left-merge keeps all 16 even though the
# sensor joins below only add data for the 10 wearable ones.
beh = pd.read_parquet(BEH_DAILY)
bcs = pd.read_parquet(BCS_DAILY)[['cow_id', 'bcs_estimate']]
vision = beh.merge(bcs, on='cow_id', how='left')
print('vision rows:', len(vision))


def imu_features(cow):
    """IMU activity features for one cow on 2023-07-25.
    Acceleration magnitude sits near gravity (~9.81) when the cow is still and deviates
    when it moves, so the spread of the signal — not its mean — separates activity from rest.
    """
    t = 'T' + cow[1:]                             # C0n -> T0n (sensor folder naming)
    f = f'{SENS}/immu/{t}/{t}_0725.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    mag = np.sqrt(d['accel_x_mps2']**2 + d['accel_y_mps2']**2 + d['accel_z_mps2']**2)
    dev = (mag - GRAVITY).abs()
    return {
        'imu_accel_mean':  float(mag.mean()),
        'imu_accel_std':   float(mag.std()),           # primary activity index
        'imu_active_frac': float((dev > 0.5).mean()),  # fraction of the day clearly moving
    }


def cbt_features(cow):
    """Core body temperature features for one cow on 2023-07-25."""
    f = f'{SENS}/cbt/{cow}.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    # The CBT file holds all 14 days in one file, so trim to the annotated day.
    d = d[(d['timestamp'] >= DAY_START) & (d['timestamp'] < DAY_END)]
    # Drop physiologically impossible readings (sensor dropouts, e.g. C09's 21.9C) that
    # would otherwise distort the daily statistics.
    d = d[(d['temperature_C'] >= 35.0) & (d['temperature_C'] <= 42.0)]
    if len(d) == 0:
        return None
    return {
        'cbt_mean': float(d['temperature_C'].mean()),
        'cbt_max':  float(d['temperature_C'].max()),
        'cbt_min':  float(d['temperature_C'].min()),
        'cbt_std':  float(d['temperature_C'].std()),
    }


def uwb_features(cow):
    """UWB movement features for one cow on 2023-07-25.
    Distance travelled is the sum of straight-line steps between consecutive positions;
    coordinates are in centimetres and converted to metres.
    """
    t = 'T' + cow[1:]
    f = f'{SENS}/uwb/{t}/{t}_0725.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).sort_values('timestamp')   # sort first, or consecutive diffs are wrong
    step = np.sqrt(d['coord_x_cm'].diff()**2 + d['coord_y_cm'].diff()**2)
    return {
        'uwb_total_dist_m': float(step.sum() / 100.0),
        'uwb_x_range_m':    float((d['coord_x_cm'].max() - d['coord_x_cm'].min()) / 100.0),
        'uwb_y_range_m':    float((d['coord_y_cm'].max() - d['coord_y_cm'].min()) / 100.0),
    }


# Assemble one row per cow.
# Vision-only cows skip the sensor extractors, so those columns stay absent here and become
# NaN once the list of rows is turned into a DataFrame.
rows = []
for _, v in vision.iterrows():
    cow = v['cow_id']
    row = v.to_dict()
    row['has_wearable'] = cow in WEARABLE
    for extractor in (imu_features, cbt_features, uwb_features):
        feats = extractor(cow) if cow in WEARABLE else None
        if feats:
            row.update(feats)
    rows.append(row)

features = pd.DataFrame(rows)

# Order the human-readable columns first so the printed table is easy to scan.
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
