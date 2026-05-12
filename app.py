"""Streamlit dashboard for Nut Counting System.

Usage:
    conda run -n ml2 streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Nut Counter",
    page_icon="🥜",
    layout="wide",
)

st.title("Nut Counting System")
st.caption("Conveyor Belt Object Counter")

# ── Load data ─────────────────────────────────────────

CSV_PATH = Path(__file__).resolve().parent / "logs" / "counts.csv"

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    counted = df[df["counted"] == True]
    total = len(counted)
    mean_conf = counted["confidence"].mean() if len(counted) > 0 else 0.0
    high_count = (counted["reliability"] == "HIGH").sum() if "reliability" in counted.columns else total
    reliability = "HIGH" if high_count == total else ("MEDIUM" if high_count / max(total, 1) > 0.9 else "LOW")
else:
    df = pd.DataFrame()
    total = 0
    mean_conf = 0.0
    high_count = 0
    reliability = "N/A"

# ── Metrics ───────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Counted", total)
with col2:
    st.metric("Mean Confidence", f"{mean_conf:.3f}")
with col3:
    st.metric("Reliability", reliability, delta=f"{high_count}/{total} high" if total > 0 else None)

# ── Tabs ──────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Count Log", "Confidence", "About"])

with tab1:
    st.subheader("Counting Events")
    if not df.empty:
        st.dataframe(
            df[["timestamp", "track_id", "confidence", "calibrated_confidence", "reliability"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No counting log found. Run the pipeline first.")

with tab2:
    st.subheader("Confidence Distribution")
    if not df.empty and len(counted) > 0:
        st.bar_chart(counted["confidence"], height=200)
    else:
        st.info("No data to plot.")

with tab3:
    st.markdown("""
    ### System Details

    - **Model:** YOLO-NAS-S, trained on combined nut dataset (3 Roboflow sources)
    - **Nut mAP@0.50:** 0.865 (pycocotools, nut-only)
    - **ECE:** 0.006 (isotonic regression calibration)
    - **Tracking:** ByteTrack (supervision 0.27)
    - **Counting:** Direction-aware line crossing

    ### Statistical Features
    - Isotonic regression confidence calibration
    - Wilson score intervals for count uncertainty
    - Per-count HIGH/MEDIUM/LOW reliability labels
    - Batch summary reports with ± error

    ### Alerts
    - Gap alert: triggers when no nut counted for > N seconds
    - Jam detection: experimental (stalled-track heuristic)
    """)

st.divider()
st.caption("YOLO-NAS + ByteTrack + Streamlit")
