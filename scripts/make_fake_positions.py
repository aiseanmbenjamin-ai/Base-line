"""Generate a hand-written-equivalent fake positions.json for testing the
analytics + dashboard half before the real CV pipeline produces one.

Per the brief: this is a normal stand-in for the data contract during
development. Never present its output as real model output.
"""
import json
import os

import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_positions.json")

COURT_DIMS_M = {"width": 8.23, "length": 23.77}
FPS_SAMPLED = 4
DURATION_S = 60


def make_track(rng, baseline_y, center_x, lateral_amplitude, depth_amplitude):
    n = DURATION_S * FPS_SAMPLED
    track = []
    for i in range(n):
        t = i / FPS_SAMPLED
        x = center_x + lateral_amplitude * np.sin(t / 3.0) + rng.normal(0, 0.05)
        y = baseline_y + depth_amplitude * np.sin(t / 5.0 + 1.0) + rng.normal(0, 0.05)
        x = float(np.clip(x, 0.2, COURT_DIMS_M["width"] - 0.2))
        y = float(np.clip(y, 0.2, COURT_DIMS_M["length"] - 0.2))
        track.append({"frame": i, "t": round(t, 2), "x": round(x, 3), "y": round(y, 3), "conf": round(float(rng.uniform(0.85, 0.98)), 3)})
    return track


def main():
    rng = np.random.default_rng(42)
    positions = {
        "video": "match01.mp4",
        "fps_sampled": FPS_SAMPLED,
        "court_dims_m": COURT_DIMS_M,
        "players": {
            "near": make_track(rng, baseline_y=3.0, center_x=COURT_DIMS_M["width"] / 2, lateral_amplitude=2.5, depth_amplitude=2.0),
            "far": make_track(rng, baseline_y=COURT_DIMS_M["length"] - 3.0, center_x=COURT_DIMS_M["width"] / 2, lateral_amplitude=2.0, depth_amplitude=2.5),
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(positions, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
