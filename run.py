"""CLI glue for the Baseline pipeline.

Usage:
  python run.py calibrate video.mp4 --out calibration.json
  python run.py track video.mp4 --calibration calibration.json --out positions.json --mock
  python run.py dashboard positions.json
"""
import argparse
import os

import cv2

from calibrate.calibration_tool import load_calibration, pick_corners_interactive, save_calibration
from grounding.mock_backend import MockGroundingBackend
from grounding.position_builder import build_positions, save_positions, smooth_positions
from sampler.frame_sampler import sample_frames

DEFAULT_COURT_DIMS_M = (8.23, 23.77)  # singles court: width x length


def cmd_calibrate(args):
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {args.video}")

    height, width = frame.shape[:2]
    court_dims_m = (args.width, args.length)
    court_points_m = [
        [0, 0],
        [court_dims_m[0], 0],
        [court_dims_m[0], court_dims_m[1]],
        [0, court_dims_m[1]],
    ]
    image_points = pick_corners_interactive(frame)
    save_calibration(args.out, args.video, (width, height), court_dims_m, image_points, court_points_m)
    print(f"Saved calibration to {args.out}")


def cmd_track(args):
    calibration = load_calibration(args.calibration)
    frames = sample_frames(args.video, args.fps)

    if not args.mock:
        raise NotImplementedError(
            "Real LocateAnything backend isn't wired up yet (no GPU env in this build). "
            "Pass --mock to use the stand-in backend, or implement "
            "grounding/locate_anything_backend.py against a GPU runner."
        )
    backend = MockGroundingBackend(calibration["frame_width"], calibration["frame_height"])

    prompts = {"near": "the player on the near side", "far": "the player on the far side"}
    court_dims_m = (calibration["court_dims_m"]["width"], calibration["court_dims_m"]["length"])
    positions = build_positions(args.video, args.fps, court_dims_m, frames, calibration, backend, prompts)
    for side, track in positions["players"].items():
        positions["players"][side] = smooth_positions(track)

    save_positions(args.out, positions)
    print(f"Saved positions to {args.out}")


def cmd_dashboard(args):
    os.environ["BASELINE_POSITIONS_PATH"] = args.positions
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "dashboard.py")
    os.execvp("streamlit", ["streamlit", "run", dashboard_path])


def main():
    parser = argparse.ArgumentParser(description="Baseline: phone-camera tennis movement analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="Pick 4 court corners and save calibration.json")
    p_cal.add_argument("video")
    p_cal.add_argument("--out", default="calibration.json")
    p_cal.add_argument("--width", type=float, default=DEFAULT_COURT_DIMS_M[0])
    p_cal.add_argument("--length", type=float, default=DEFAULT_COURT_DIMS_M[1])
    p_cal.set_defaults(func=cmd_calibrate)

    p_track = sub.add_parser("track", help="Sample frames, run grounding, write positions.json")
    p_track.add_argument("video")
    p_track.add_argument("--calibration", default="calibration.json")
    p_track.add_argument("--out", default="positions.json")
    p_track.add_argument("--fps", type=float, default=4.0)
    p_track.add_argument("--mock", action="store_true", help="Use the mock grounding backend")
    p_track.set_defaults(func=cmd_track)

    p_dash = sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    p_dash.add_argument("positions", nargs="?", default="positions.json")
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
