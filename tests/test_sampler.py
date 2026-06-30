import cv2
import numpy as np
import pytest

from sampler.frame_sampler import sample_frames


@pytest.fixture
def synthetic_video(tmp_path):
    path = str(tmp_path / "synthetic.mp4")
    source_fps = 30
    num_frames = 60  # 2 seconds at 30fps
    width, height = 64, 48

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_sample_frames_rate_matches_request(synthetic_video):
    frames = sample_frames(synthetic_video, fps_sampled=5)
    # ~2s of video sampled at 5fps should yield roughly 10 frames.
    assert 8 <= len(frames) <= 12
    assert frames[0]["frame"] == 0
    assert frames[0]["t"] == pytest.approx(0.0, abs=1e-6)


def test_sample_frames_writes_to_disk(tmp_path, synthetic_video):
    out_dir = str(tmp_path / "frames")
    frames = sample_frames(synthetic_video, fps_sampled=5, out_dir=out_dir)
    for frame in frames:
        assert frame["path"] is not None
        import os

        assert os.path.exists(frame["path"])


def test_sample_frames_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        sample_frames("does_not_exist.mp4", fps_sampled=4)
