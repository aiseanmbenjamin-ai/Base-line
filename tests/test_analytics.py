import pytest

from analytics.engine import (
    compute_all_metrics,
    coverage_heatmap,
    distance_covered_m,
    recovery_position,
    work_rate_over_time,
    zone_split,
)

COURT_DIMS_M = {"width": 8.0, "length": 24.0}


def make_track(points):
    return [{"frame": i, "t": i * 0.25, "x": x, "y": y, "conf": 0.9} for i, (x, y) in enumerate(points)]


def test_distance_covered_straight_line():
    track = make_track([(0, 0), (0, 3), (0, 6)])
    assert distance_covered_m(track) == pytest.approx(6.0)


def test_distance_covered_needs_two_points():
    assert distance_covered_m(make_track([(1, 1)])) == 0.0
    assert distance_covered_m([]) == 0.0


def test_recovery_position_center():
    track = make_track([(4, 5), (4, 7)])
    pos = recovery_position(track, COURT_DIMS_M)
    assert pos["mean_x"] == pytest.approx(4.0)
    assert pos["lateral_offset_m"] == pytest.approx(0.0)


def test_recovery_position_offset():
    track = make_track([(6, 5), (6, 7)])
    pos = recovery_position(track, COURT_DIMS_M)
    assert pos["lateral_offset_m"] == pytest.approx(2.0)


def test_zone_split_sums_to_100():
    track = make_track([(4, y) for y in range(0, 24)])
    split = zone_split(track, COURT_DIMS_M)
    assert sum(split.values()) == pytest.approx(100.0)


def test_zone_split_all_baseline():
    track = make_track([(4, 0.5), (4, 23.5)])
    split = zone_split(track, COURT_DIMS_M)
    assert split["baseline"] == pytest.approx(100.0)
    assert split["midcourt"] == pytest.approx(0.0)


def test_coverage_heatmap_shape_and_mass():
    track = make_track([(1, 1), (2, 2), (3, 3)])
    heatmap, xedges, yedges = coverage_heatmap(track, COURT_DIMS_M, bins=(8, 24))
    assert heatmap.shape == (8, 24)
    assert heatmap.sum() == pytest.approx(len(track))


def test_work_rate_bins_are_chronological_and_sum_to_total():
    track = make_track([(0, 0), (0, 1), (0, 2), (0, 3)])
    bins = work_rate_over_time(track, bin_seconds=0.5)
    assert [b["t_start"] for b in bins] == sorted(b["t_start"] for b in bins)
    assert sum(b["distance_m"] for b in bins) == pytest.approx(distance_covered_m(track))


def test_work_rate_empty_for_short_track():
    assert work_rate_over_time(make_track([(0, 0)])) == []


def test_compute_all_metrics_has_expected_keys():
    track = make_track([(1, 1), (2, 2), (3, 3), (4, 4)])
    metrics = compute_all_metrics(track, COURT_DIMS_M)
    assert set(metrics) == {"distance_covered_m", "zone_split_pct", "recovery_position", "work_rate"}
