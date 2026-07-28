"""
BóSight — Herd Health Dashboard (Week 9).

The front end for the whole pipeline. It does NOT run any models — by this point
everything (behaviour, BCS, sensor features, alerts) is already sitting in the parquet
files under outputs/, so the dashboard just reads those and draws them. That keeps it
snappy and means it'll happily run on a laptop with no GPU.

Two pages:
  - Herd Overview  : the "how's the whole barn doing" screen — alert counts, a
                     behaviour breakdown, and a colour-coded table you can sort.
  - Per-Cow Detail : drill into one cow — photo, behaviour, sensors, and a plain-English
                     breakdown of which health rules fired and why.

Run:  streamlit run app.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Config & paths
# ----------------------------------------------------------------------------
st.set_page_config(page_title="BóSight — Herd Health", page_icon="🐄", layout="wide")

# outputs/ sits at the repo root (this file is in src/), so go up one level
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
# External cow-gallery reference photos (not in the repo). Optional — the per-cow page
# shows a placeholder if a photo isn't found. Point this at your local MmCows gallery.
GALLERY = Path("D:/D2/MmCows-20260630T220432Z-3-001/MmCows/extracted/cows_gallery")

BEHAVIOURS = ["frac_lying", "frac_standing", "frac_feeding", "frac_moving"]
BEH_LABELS = {"frac_lying": "Lying", "frac_standing": "Standing",
              "frac_feeding": "Feeding", "frac_moving": "Moving"}
BEH_COLORS = {"Lying": "#4C72B0", "Standing": "#55A868",
              "Feeding": "#C44E52", "Moving": "#8172B3"}

ALERT_STYLE = {
    "critical": ("🔴", "#C44E52"),
    "suspect":  ("🟡", "#DD9F40"),
    "healthy":  ("🟢", "#55A868"),
}

# veterinary thresholds (mirror week8_make_alerts.py)
CBT_FEVER, CBT_HIGH_MEAN = 39.5, 39.0
BCS_LOW, LYING_HIGH, FEEDING_LOW = 2.2, 0.60, 0.15


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    feats  = pd.read_parquet(OUT_DIR / "cow_day_features.parquet")
    alerts = pd.read_parquet(OUT_DIR / "alerts_daily.parquet")
    df = feats.merge(
        alerts[["cow_id", "alert_level", "severity_score", "reasons"]],
        on="cow_id", how="left")
    return df


@st.cache_data
def gallery_photo(cow_id: str):
    hits = sorted(glob.glob(str(GALLERY / f"{cow_id} *" / "*.jpg")) +
                  glob.glob(str(GALLERY / f"{cow_id} *" / "*.JPG")))
    return hits[0] if hits else None


df = load_data()
DATE = df["date"].iloc[0]

# herd reference stats for activity z-scores (wearable cows only)
_wear = df[df["has_wearable"]]
_imu_mu, _imu_sd = _wear["imu_active_frac"].mean(), _wear["imu_active_frac"].std()


def rule_flags(row):
    """The same 7 rules Week 8 uses, re-evaluated here so the per-cow page can show
    exactly which ones fired (not just the final label). Kept deliberately in sync with
    week8_make_alerts.py — if you change a threshold there, change it here too.
    Returns a list of (rule_name, fired?, severity_if_fired)."""
    def za(v):   # activity z-score vs the wearable herd
        return (v - _imu_mu) / _imu_sd if _imu_sd else 0.0
    w = bool(row["has_wearable"])   # sensor rules only apply to the 10 tagged cows
    flags = [
        ("Fever — CBT max ≥ 39.5°C",         w and row["cbt_max"] >= CBT_FEVER,          3),
        ("Elevated temp — CBT mean ≥ 39.0°C", w and row["cbt_mean"] >= CBT_HIGH_MEAN,     1),
        ("Lethargy — activity z ≤ −1.5",      w and za(row["imu_active_frac"]) <= -1.5,   2),
        ("Restless/estrus — activity z ≥ 2.0", w and za(row["imu_active_frac"]) >= 2.0,   1),
        ("Excessive lying — ≥ 60%",           row["frac_lying"] >= LYING_HIGH,            1),
        ("Reduced feeding — ≤ 15%",           row["frac_feeding"] <= FEEDING_LOW,         2),
        ("Low body condition — BCS ≤ 2.2",    row["bcs_estimate"] <= BCS_LOW,             1),
    ]
    return flags


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🐄 BóSight")
st.sidebar.caption("Herd Health Dashboard")
page = st.sidebar.radio("Navigate", ["Herd Overview", "Per-Cow Detail"])
st.sidebar.markdown("---")
st.sidebar.metric("Date", DATE)
st.sidebar.metric("Cows monitored", len(df))

# sidebar herd summary stats
st.sidebar.markdown("**Herd averages**")
s1, s2 = st.sidebar.columns(2)
s1.metric("Avg BCS", f"{df['bcs_estimate'].mean():.1f}")
s2.metric("Avg lying", f"{df['frac_lying'].mean()*100:.0f}%")
s3, s4 = st.sidebar.columns(2)
s3.metric("Avg feeding", f"{df['frac_feeding'].mean()*100:.0f}%")
s4.metric("Avg activity", f"{_wear['imu_active_frac'].mean()*100:.0f}%")
st.sidebar.markdown("---")
st.sidebar.caption("Data: MmCows 2023-07-25 (cam_1). Alerts are a rule-based "
                   "screening demo — BCS is a proxy and there is no clinical "
                   "ground truth, so no precision/recall is reported.")


# ----------------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------------
def behaviour_pie(row):
    vals = [row[b] * 100 for b in BEHAVIOURS]
    labels = [BEH_LABELS[b] for b in BEHAVIOURS]
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.45,
                           marker_colors=[BEH_COLORS[l] for l in labels]))
    fig.update_traces(textinfo="label+percent", sort=False)
    fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    return fig


def herd_behaviour_bar(df):
    d = df.copy()
    for b in BEHAVIOURS:
        d[BEH_LABELS[b]] = d[b] * 100
    long = d.melt(id_vars="cow_id", value_vars=list(BEH_LABELS.values()),
                  var_name="Behaviour", value_name="pct")
    fig = px.bar(long, x="cow_id", y="pct", color="Behaviour",
                 color_discrete_map=BEH_COLORS, labels={"pct": "% of day", "cow_id": ""})
    fig.update_layout(height=400, barmode="stack", legend_title="", margin=dict(t=10, b=10))
    return fig


# ============================================================================
# PAGE 1 — HERD OVERVIEW
# ============================================================================
if page == "Herd Overview":
    st.title("Herd Health Overview")
    st.caption(f"{DATE} · cam_1")

    counts = df["alert_level"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Healthy", int(counts.get("healthy", 0)))
    c2.metric("🟡 Suspect", int(counts.get("suspect", 0)))
    c3.metric("🔴 Critical", int(counts.get("critical", 0)))
    c4.metric("Active alerts", int(counts.get("suspect", 0) + counts.get("critical", 0)))

    st.markdown("---")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Herd behaviour distribution")
        st.plotly_chart(herd_behaviour_bar(df), use_container_width=True)
    with right:
        st.subheader("Cows needing attention")
        flagged = df[df["alert_level"] != "healthy"].sort_values("severity_score", ascending=False)
        if len(flagged) == 0:
            st.success("No cows flagged today.")
        for _, r in flagged.iterrows():
            emoji, color = ALERT_STYLE[r["alert_level"]]
            st.markdown(
                f"<div style='padding:8px 12px;border-left:4px solid {color};"
                f"background:rgba(128,128,128,0.08);margin-bottom:8px;border-radius:4px'>"
                f"<b>{emoji} {r['cow_id']}</b> — {r['alert_level'].title()}<br>"
                f"<span style='font-size:0.85em;color:gray'>{r['reasons'] or '—'}</span>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("---")

    hdr, dl = st.columns([4, 1])
    hdr.subheader("All cows")

    # build display table
    order = {"critical": 0, "suspect": 1, "healthy": 2}
    tbl = df.copy()
    tbl["_ord"] = tbl["alert_level"].map(order)
    tbl = tbl.sort_values(["_ord", "cow_id"])
    tbl["Alert"]     = tbl["alert_level"].map(lambda x: f"{ALERT_STYLE[x][0]} {x.title()}")
    tbl["Lying %"]   = (tbl["frac_lying"] * 100).round(0)
    tbl["Feeding %"] = (tbl["frac_feeding"] * 100).round(0)
    tbl["BCS"]       = tbl["bcs_estimate"].round(1)
    tbl["CBT max"]   = tbl["cbt_max"].round(1)          # NaN for C11–C16
    tbl["Activity %"] = (tbl["imu_active_frac"] * 100).round(0)
    tbl["Wearable"]  = tbl["has_wearable"].map({True: "✓", False: "—"})
    show = tbl[["cow_id", "Alert", "Lying %", "Feeding %", "BCS",
                "CBT max", "Activity %", "Wearable", "reasons"]].rename(
                    columns={"cow_id": "Cow", "reasons": "Reasons"})

    # Colour the cells that cross a threshold so the table reads like a heatmap —
    # you can spot the worrying cows without reading a single number.
    def style_row(row):
        styles = {c: "" for c in show.columns}
        if pd.notna(row["CBT max"]) and row["CBT max"] >= CBT_FEVER:
            styles["CBT max"] = "background-color: rgba(196,78,82,0.35)"
        if pd.notna(row["Feeding %"]) and row["Feeding %"] <= FEEDING_LOW * 100:
            styles["Feeding %"] = "background-color: rgba(221,159,64,0.35)"
        if pd.notna(row["Lying %"]) and row["Lying %"] >= LYING_HIGH * 100:
            styles["Lying %"] = "background-color: rgba(221,159,64,0.35)"
        if pd.notna(row["BCS"]) and row["BCS"] <= BCS_LOW:
            styles["BCS"] = "background-color: rgba(196,78,82,0.25)"
        # grey out sensor columns for vision-only cows
        if row["Wearable"] == "—":
            for c in ["CBT max", "Activity %"]:
                styles[c] = "color: #999; background-color: rgba(128,128,128,0.12)"
        return pd.Series(styles)

    styled = show.style.apply(style_row, axis=1).format(
        {"Lying %": "{:.0f}", "Feeding %": "{:.0f}", "BCS": "{:.1f}",
         "CBT max": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
         "Activity %": lambda v: "—" if pd.isna(v) else f"{v:.0f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("🔴 red = elevated CBT / low BCS · 🟠 amber = high lying / low feeding · "
               "grey = vision-only cow (no wearable sensors). Click a column header to sort.")

    # download button
    csv = df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(index=False)
    dl.download_button("⬇ Alerts CSV", csv, file_name=f"bosight_alerts_{DATE}.csv",
                       mime="text/csv", use_container_width=True)


# ============================================================================
# PAGE 2 — PER-COW DETAIL
# ============================================================================
else:
    st.title("Per-Cow Detail")

    cow = st.selectbox("Select cow", sorted(df["cow_id"]))
    row = df[df["cow_id"] == cow].iloc[0]
    emoji, color = ALERT_STYLE[row["alert_level"]]

    st.markdown(f"## {emoji} {cow} — {row['alert_level'].title()}")
    if row["reasons"]:
        st.markdown(f"**Alert reasons:** {row['reasons']}")

    st.markdown("---")

    col_img, col_beh, col_stat = st.columns([1, 1.3, 1.2])

    with col_img:
        st.subheader("Reference")
        photo = gallery_photo(cow)
        if photo:
            st.image(photo, use_container_width=True)
        else:
            st.info("No gallery photo.")

    with col_beh:
        st.subheader("Behaviour (24h)")
        st.plotly_chart(behaviour_pie(row), use_container_width=True)

    with col_stat:
        st.subheader("Body & sensors")
        st.metric("BCS (proxy)", f"{row['bcs_estimate']:.1f}",
                  help="Relative body-condition proxy from bbox size (2.0–4.0). "
                       "Not a clinical score — MmCows has no BCS ground truth.")
        if row["has_wearable"]:
            st.metric("CBT mean", f"{row['cbt_mean']:.2f} °C", delta=f"max {row['cbt_max']:.1f}",
                      help="Core body temperature. ≥39.5°C flags fever/heat stress.")
            st.metric("Activity (active frac)", f"{row['imu_active_frac']*100:.0f}%",
                      help="Fraction of the day the neck IMU shows active movement.")
            st.metric("Distance travelled", f"{row['uwb_total_dist_m']:.0f} m",
                      help="Total path length from UWB positioning.")
        else:
            st.info("No wearable sensors (C11–C16). Vision-only cow.")

    st.markdown("---")

    # ---- rule-flag breakdown (all 7 rules, which fired) ----
    st.subheader("Health screening — rule breakdown")
    flags = rule_flags(row)
    fired_total = sum(sev for _, fired, sev in flags if fired)
    fb = pd.DataFrame([{
        "Rule": name,
        "Fired": "🔴 yes" if fired else "— no",
        "Severity if fired": sev,
    } for name, fired, sev in flags])
    st.dataframe(fb, use_container_width=True, hide_index=True)
    st.caption(f"Total severity score: **{fired_total}** "
               f"→ {row['alert_level'].title()}  (≥3 critical · ≥1 suspect · 0 healthy). "
               "Sensor-based rules are inactive for vision-only cows.")

    st.markdown("---")

    # ---- feature vector ----
    st.subheader("Feature vector")
    cols_show = ["frac_lying", "frac_standing", "frac_feeding", "frac_moving",
                 "bcs_estimate", "n_observations",
                 "imu_accel_mean", "imu_accel_std", "imu_active_frac",
                 "cbt_mean", "cbt_max", "cbt_min", "cbt_std",
                 "uwb_total_dist_m", "uwb_x_range_m", "uwb_y_range_m"]
    fv = row[cols_show].to_frame("value")
    fv["value"] = pd.to_numeric(fv["value"], errors="coerce").round(3)
    st.dataframe(fv, use_container_width=True)

    # ---- herd comparison ----
    if row["has_wearable"]:
        st.subheader("How this cow compares (wearable cows)")
        wear = df[df["has_wearable"]]
        metric = st.selectbox("Metric", ["imu_active_frac", "cbt_max", "uwb_total_dist_m",
                                          "frac_lying", "frac_feeding"])
        wsorted = wear.sort_values(metric)
        fig = px.bar(wsorted, x="cow_id", y=metric, labels={"cow_id": "", metric: metric})
        fig.update_traces(marker_color=[color if c == cow else "#B0B0B0"
                                        for c in wsorted["cow_id"]])
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
