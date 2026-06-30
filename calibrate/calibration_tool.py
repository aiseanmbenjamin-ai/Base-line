"""Court calibration: 4 clicked image corners -> homography (pixels -> court meters).

The camera is assumed fixed for the whole clip, so this runs once per video.
Corner order must be consistent between image_points and court_points_m:
near-left, near-right, far-right, far-left (walking the court boundary).

Interactive corner-picking needs a GUI (cv2.imshow), so it only works on a
machine with a display and a non-headless OpenCV build. The programmatic
functions below (compute_homography, apply_homography, save/load) have no
GUI dependency and are what the rest of the pipeline and the tests use.
"""
import json

import cv2
import numpy as np


def compute_homography(image_points, court_points_m):
    """image_points / court_points_m: 4 (x, y) pairs each, in matching order.

    Returns the 3x3 homography matrix mapping image pixels to court meters.
    """
    src = np.array(image_points, dtype=np.float32)
    dst = np.array(court_points_m, dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst, method=0)
    return homography


def apply_homography(homography, point):
    """Map a single (x, y) pixel point to court meters."""
    px = np.array([[point]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(px, homography)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def save_calibration(path, video, frame_size, court_dims_m, image_points, court_points_m):
    homography = compute_homography(image_points, court_points_m)
    data = {
        "video": video,
        "frame_width": frame_size[0],
        "frame_height": frame_size[1],
        "court_dims_m": {"width": court_dims_m[0], "length": court_dims_m[1]},
        "image_points": image_points,
        "court_points_m": court_points_m,
        "homography": homography.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_calibration(path):
    with open(path) as f:
        data = json.load(f)
    data["homography"] = np.array(data["homography"], dtype=np.float64)
    return data


def pick_corners_interactive(image):
    """Open a window for the user to click 4 corners in order:
    near-left, near-right, far-right, far-left. Returns a list of (x, y).

    Requires a display and a non-headless OpenCV build.
    """
    points = []
    window = "Calibration - click 4 corners (near-left, near-right, far-right, far-left), Esc to cancel"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    try:
        while True:
            display = image.copy()
            for i, p in enumerate(points):
                cv2.circle(display, p, 5, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if len(points) == 4 or key == 27:
                break
    finally:
        cv2.destroyWindow(window)

    if len(points) != 4:
        raise RuntimeError("Calibration cancelled before 4 corners were picked")
    return points
