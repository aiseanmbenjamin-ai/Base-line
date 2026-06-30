"""Streamlit dashboard: positions.json -> coverage heatmap + the four core
movement metrics, per player.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import streamlit as st

from analytics.engine import compute_all_metrics, coverage_heatmap
from grounding.position_builder import load_positions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render_heatmap(track, court_dims_m, title):
    heatmap, _, _ = coverage_heatmap(track, court_dims_m)
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[0, court_dims_m["width"], 0, court_dims_m["length"]],
        cmap="hot",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("court width (m)")
    ax.set_ylabel("court length (m)")
    return fig


def main():
    st.set_page_config(page_title="Baseline", layout="wide")
    st.title("Baseline — movement analytics")

    default_path = os.environ.get(
        "BASELINE_POSITIONS_PATH", os.path.join(REPO_ROOT, "data", "sample_positions.json")
    )
    path = st.text_input("positions.json path", value=default_path)

    if not os.path.exists(path):
        st.warning(f"No file at {path}")
        return

    data = load_positions(path)
    court_dims_m = data["court_dims_m"]

    cols = st.columns(2)
    for col, side in zip(cols, ["near", "far"]):
        track = data["players"].get(side, [])
        with col:
            st.subheader(f"{side.title()} player")
            if not track:
                st.info("No data for this player.")
                continue
            st.pyplot(render_heatmap(track, court_dims_m, f"{side} coverage"))
            metrics = compute_all_metrics(track, court_dims_m)
            st.metric("Distance covered", f"{metrics['distance_covered_m']:.1f} m")
            st.write("Zone split (%)", metrics["zone_split_pct"])
            st.write("Recovery position", metrics["recovery_position"])
            if metrics["work_rate"]:
                st.line_chart({"distance_m": [b["distance_m"] for b in metrics["work_rate"]]})


if __name__ == "__main__":
    main()
