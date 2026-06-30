"""Real grounding backend, backed by NVIDIA LocateAnything.

Not implemented here: this environment has no GPU and no LocateAnything
weights. Wire this up once a GPU runner is available (Modal, RunPod,
Colab, or a hosted inference endpoint):

  1. Load the model + processor once in __init__ (HF weights, device="cuda").
  2. In locate(), run inference with `prompt`, parse the model's box output,
     and return a BoundingBox (or None if nothing matched).

LocateAnything ships under NVIDIA's non-commercial research license -
fine for this PoC, but flag it before any commercial path.
"""
from .base import GroundingBackend


class LocateAnythingBackend(GroundingBackend):
    def __init__(self, model_path=None, device="cuda"):
        raise NotImplementedError(
            "LocateAnythingBackend requires a GPU environment and the LocateAnything "
            "weights. Load the model in __init__ and implement locate() to call it."
        )

    def locate(self, image, prompt):
        raise NotImplementedError
