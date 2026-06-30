"""Extract frames from a video at a fixed sampling rate.

Movement analysis doesn't need 30fps, so sampling down to ~3-5fps cuts the
GPU grounding workload (the expensive step) by 6-10x with no real loss of
signal for player position tracking.
"""
import os

import cv2


def sample_frames(video_path, fps_sampled, out_dir=None):
    """Sample frames from video_path at a fixed rate.

    Returns a list of dicts: {"frame": sampled_idx, "t": seconds, "image": ndarray, "path": str|None}.
    If out_dir is given, each sampled frame is also written to disk as a JPEG.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(source_fps / fps_sampled)), 1)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames = []
    frame_idx = 0
    sampled_idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            path = None
            if out_dir:
                path = os.path.join(out_dir, f"frame_{sampled_idx:05d}.jpg")
                cv2.imwrite(path, image)
            frames.append({
                "frame": sampled_idx,
                "t": frame_idx / source_fps,
                "image": image,
                "path": path,
            })
            sampled_idx += 1
        frame_idx += 1

    cap.release()
    return frames
