"""Generate realistic-looking fake positions.json for testing the analytics
+ dashboard half before the real LocateAnything grounding pipeline runs.

Movement model: Ornstein-Uhlenbeck mean-reverting random walk centred on
each player's home position, with periodic "chase" excursions to simulate
lateral recovery runs and occasional net approaches.
"""
import json
import os

import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_positions.json")

COURT = {"width": 8.23, "length": 23.77}
FPS_SAMPLED = 4
DURATION_S = 120


def ou_step(x, mu, theta, sigma, rng):
    return x + theta * (mu - x) + rng.normal(0, sigma)


def make_track(rng, home_x, home_y, court, lateral_sigma=0.5, depth_sigma=0.3, approach_prob=0.015):
    """Simulate a player around their baseline home position for DURATION_S seconds.

    lateral_sigma: per-step lateral noise (high = wide coverage)
    depth_sigma:   per-step depth noise (low = stays near baseline)
    approach_prob: per-frame chance of starting a net approach / recovery run
    """
    n = DURATION_S * FPS_SAMPLED
    W, L = court["width"], court["length"]
    track = []
    x, y = home_x, home_y
    approach_countdown = 0

    for i in range(n):
        t = i / FPS_SAMPLED

        # Occasional approach to mid-court or net then recovery back
        if approach_countdown == 0 and rng.random() < approach_prob:
            approach_countdown = int(rng.uniform(8, 20))

        if approach_countdown > 0:
            net = L / 2
            target_y = home_y + (net - home_y) * 0.6
            y = ou_step(y, target_y, 0.25, depth_sigma * 1.5, rng)
            approach_countdown -= 1
        else:
            y = ou_step(y, home_y, 0.12, depth_sigma, rng)

        # Lateral: weak mean-reversion so player sweeps the full width
        x = ou_step(x, home_x, 0.05, lateral_sigma, rng)
        x = float(np.clip(x, 0.3, W - 0.3))
        y = float(np.clip(y, 0.3, L - 0.3))

        track.append({
            "frame": i,
            "t": round(t, 3),
            "x": round(x, 3),
            "y": round(y, 3),
            "conf": round(float(rng.uniform(0.85, 0.98)), 3),
        })

    return track


def main():
    rng = np.random.default_rng(42)
    W, L = COURT["width"], COURT["length"]

    positions = {
        "video": "match01.mp4",
        "fps_sampled": FPS_SAMPLED,
        "court_dims_m": COURT,
        "players": {
            "near": make_track(rng, home_x=W / 2, home_y=2.5, court=COURT,
                               lateral_sigma=0.55, depth_sigma=0.35, approach_prob=0.02),
            "far":  make_track(rng, home_x=W / 2, home_y=L - 2.5, court=COURT,
                               lateral_sigma=0.55, depth_sigma=0.35, approach_prob=0.015),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(positions, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    for side, track in positions["players"].items():
        xs = [p["x"] for p in track]
        ys = [p["y"] for p in track]
        print(f"  {side}: x=[{min(xs):.2f}, {max(xs):.2f}] y=[{min(ys):.2f}, {max(ys):.2f}]")


if __name__ == "__main__":
    main()
