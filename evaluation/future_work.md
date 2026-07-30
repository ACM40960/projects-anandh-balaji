# BóSight — Future Work

Three directions for developing BóSight from a proof-of-concept into a deployable system.

## 1. Real-time, on-farm deployment

Extend the offline pipeline to run continuously on live camera feeds, with near-real-time
inference at the barn (edge or streaming). This includes multi-camera coverage to reduce
occlusion and robust long-horizon identity maintenance as animals change in appearance over
time, so that a farmer sees an always-current herd health view rather than a retrospective one.

## 2. Depth-based body measurement

Add a 3D/depth sensing modality to estimate body weight and stature directly from point-cloud
geometry. Trained against real weigh-scale data, this replaces the current relative
body-condition estimate with a calibrated, angle-invariant measurement — turning body
condition into a quantitative growth and welfare signal.

## 3. Temporal behaviour modelling and richer ethogram

Move from single-frame classification to short-clip, temporal models (e.g. video CNNs) fused
with the wearable motion signal, which better captures movement-based and health-relevant
behaviours. This also supports expanding the behaviour set beyond the current four classes to a
finer ethogram, increasing the clinical value of the per-animal behaviour profile.
