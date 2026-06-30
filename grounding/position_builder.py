"""Boxes + homography -> positions.json (the data contract between the GPU
half of the pipeline and the pure-math analytics half).
"""
import json

from calibrate.calibration_tool import apply_homography


def build_positions(video, fps_sampled, court_dims_m, frames, calibration, backend, prompts):
    """frames: list of {"frame", "t", "image"} from the frame sampler.
    prompts: {"near": "...", "far": "..."} text prompts per player/side.

    Returns a dict matching the positions.json contract.
    """
    homography = calibration["homography"]
    players = {side: [] for side in prompts}

    for frame in frames:
        for side, prompt in prompts.items():
            box = backend.locate(frame["image"], prompt)
            if box is None:
                continue
            x, y = apply_homography(homography, box.feet_point())
            players[side].append({
                "frame": frame["frame"],
                "t": frame["t"],
                "x": x,
                "y": y,
                "conf": box.conf,
            })

    return {
        "video": video,
        "fps_sampled": fps_sampled,
        "court_dims_m": {"width": court_dims_m[0], "length": court_dims_m[1]},
        "players": players,
    }


def smooth_positions(track, window=3):
    """Moving-average smoothing on a single player's track, to de-jitter
    frame-to-frame grounding noise.
    """
    if len(track) < window:
        return track
    xs = [p["x"] for p in track]
    ys = [p["y"] for p in track]
    half = window // 2
    smoothed = []
    for i, p in enumerate(track):
        lo, hi = max(0, i - half), min(len(track), i + half + 1)
        smoothed.append({**p, "x": sum(xs[lo:hi]) / (hi - lo), "y": sum(ys[lo:hi]) / (hi - lo)})
    return smoothed


def save_positions(path, positions):
    with open(path, "w") as f:
        json.dump(positions, f, indent=2)


def load_positions(path):
    with open(path) as f:
        return json.load(f)
