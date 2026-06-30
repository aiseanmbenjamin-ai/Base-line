"""Grounding backend interface: image + text prompt -> a bounding box.

This is the only GPU-dependent step in the pipeline (component 3 in the
architecture brief). Everything upstream and downstream of it is
conventional CV / plain math and has no GPU dependency.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    def feet_point(self):
        """Bottom-center of the box - used as the player's court position."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


class GroundingBackend(ABC):
    @abstractmethod
    def locate(self, image, prompt):
        """Return the best-matching BoundingBox for `prompt` in `image`, or None."""
        raise NotImplementedError
