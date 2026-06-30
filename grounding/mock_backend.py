"""Deterministic fake grounding backend.

Lets the sampler -> position builder -> analytics -> dashboard chain be
built and tested end-to-end before LocateAnything is wired in. Per the
brief: never present this output as if the real model produced it.
"""
import numpy as np

from .base import BoundingBox, GroundingBackend


class MockGroundingBackend(GroundingBackend):
    """Returns synthetic boxes that drift smoothly across the frame."""

    def __init__(self, frame_width, frame_height, seed=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._rng = np.random.default_rng(seed)

    def locate(self, image, prompt):
        cx = self.frame_width * (0.3 if "near" in prompt else 0.7)
        cx += self._rng.normal(0, 15)
        cy = self.frame_height * 0.7 + self._rng.normal(0, 10)
        box_w, box_h = 60, 140
        return BoundingBox(cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, conf=0.9)
