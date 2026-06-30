"""Pure-NumPy analytics on a positions.json track: heatmap + the four core
movement metrics. No GPU dependency - this is the half you can iterate on
locally all day.
"""
import numpy as np


def _xy(track):
    return np.array([[p["x"], p["y"]] for p in track], dtype=np.float64)


def coverage_heatmap(track, court_dims_m, bins=(20, 40)):
    """2D histogram of court occupancy. Returns (heatmap, xedges, yedges)."""
    xy = _xy(track)
    width, length = court_dims_m["width"], court_dims_m["length"]
    heatmap, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=bins, range=[[0, width], [0, length]]
    )
    return heatmap, xedges, yedges


def distance_covered_m(track):
    """Total distance covered, in meters, from frame-to-frame displacement."""
    xy = _xy(track)
    if len(xy) < 2:
        return 0.0
    deltas = np.diff(xy, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def zone_split(track, court_dims_m, baseline_frac=0.25, net_frac=0.25):
    """% of samples in baseline / mid-court / forecourt zones, split along
    the court's length (y=0 and y=length are the two baselines, y=length/2
    is the net).
    """
    xy = _xy(track)
    if len(xy) == 0:
        return {"baseline": 0.0, "midcourt": 0.0, "forecourt": 0.0}
    length = court_dims_m["length"]
    y = xy[:, 1]
    in_baseline = (y < baseline_frac * length) | (y > (1 - baseline_frac) * length)
    near_net = np.abs(y - length / 2) < (net_frac * length / 2)
    forecourt = near_net & ~in_baseline
    midcourt = ~in_baseline & ~forecourt
    n = len(y)
    return {
        "baseline": float(np.sum(in_baseline) / n * 100),
        "midcourt": float(np.sum(midcourt) / n * 100),
        "forecourt": float(np.sum(forecourt) / n * 100),
    }


def recovery_position(track, court_dims_m):
    """Mean position and lateral offset (meters) from the court's center line."""
    xy = _xy(track)
    if len(xy) == 0:
        return {"mean_x": None, "mean_y": None, "lateral_offset_m": None}
    mean_x, mean_y = xy.mean(axis=0)
    center_x = court_dims_m["width"] / 2
    return {
        "mean_x": float(mean_x),
        "mean_y": float(mean_y),
        "lateral_offset_m": float(mean_x - center_x),
    }


def work_rate_over_time(track, bin_seconds=30):
    """Distance covered (m) per time bin. Returns a list of
    {"t_start", "distance_m"}; a late drop is a conditioning signal.
    """
    if len(track) < 2:
        return []
    xy = _xy(track)
    t = np.array([p["t"] for p in track])
    deltas = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    seg_t = t[1:]

    bins = []
    bin_start = t[0]
    bin_end = bin_start + bin_seconds
    acc = 0.0
    for seg_time, d in zip(seg_t, deltas):
        while seg_time >= bin_end:
            bins.append({"t_start": float(bin_start), "distance_m": float(acc)})
            bin_start = bin_end
            bin_end += bin_seconds
            acc = 0.0
        acc += d
    bins.append({"t_start": float(bin_start), "distance_m": float(acc)})
    return bins


def compute_all_metrics(track, court_dims_m):
    return {
        "distance_covered_m": distance_covered_m(track),
        "zone_split_pct": zone_split(track, court_dims_m),
        "recovery_position": recovery_position(track, court_dims_m),
        "work_rate": work_rate_over_time(track),
    }
