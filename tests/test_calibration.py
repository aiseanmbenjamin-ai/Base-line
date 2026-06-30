import pytest

from calibrate.calibration_tool import apply_homography, compute_homography

COURT_DIMS_M = (8.23, 23.77)
COURT_POINTS_M = [
    [0, 0],
    [COURT_DIMS_M[0], 0],
    [COURT_DIMS_M[0], COURT_DIMS_M[1]],
    [0, COURT_DIMS_M[1]],
]
# A plausible fixed-tripod view: far corners appear higher and closer together (perspective).
IMAGE_POINTS = [
    [200, 900],
    [1100, 900],
    [950, 300],
    [350, 300],
]


def test_corners_map_back_to_court_points():
    homography = compute_homography(IMAGE_POINTS, COURT_POINTS_M)
    for image_pt, court_pt in zip(IMAGE_POINTS, COURT_POINTS_M):
        mapped = apply_homography(homography, image_pt)
        assert mapped[0] == pytest.approx(court_pt[0], abs=1e-3)
        assert mapped[1] == pytest.approx(court_pt[1], abs=1e-3)


def test_center_of_image_maps_inside_court():
    homography = compute_homography(IMAGE_POINTS, COURT_POINTS_M)
    cx = sum(p[0] for p in IMAGE_POINTS) / 4
    cy = sum(p[1] for p in IMAGE_POINTS) / 4
    x, y = apply_homography(homography, (cx, cy))
    assert 0 <= x <= COURT_DIMS_M[0]
    assert 0 <= y <= COURT_DIMS_M[1]
