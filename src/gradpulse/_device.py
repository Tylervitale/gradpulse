"""gradpulse._device - single source of truth for the compute device.

Set ``GRADPULSE_DEVICE`` to choose the torch device explicitly ('cpu', 'cuda',
'cuda:1', ...); unset (the default) falls back to CUDA when available, else CPU.
Resolved once at import time, so set it *before* importing gradpulse -- the same
way ``CUDA_VISIBLE_DEVICES`` works.

Centralised here (a dependency-free leaf module) so ``parametric``, ``analysis``,
``crossresonance``, and every module that imports ``DEVICE`` from ``parametric``
all resolve the device identically -- previously each defined its own copy.

Note: the paper's benchmark shows the 9-D model's small multi-seed searches run
fastest on CPU; the GPU only wins for large batches. On a GPU workstation, set
``GRADPULSE_DEVICE=cpu`` to match that guidance without touching
``CUDA_VISIBLE_DEVICES``.
"""
from __future__ import annotations

import os

import torch


def resolve_device() -> torch.device:
    """Resolve the torch device from ``GRADPULSE_DEVICE``, else CUDA-if-available."""
    requested = os.environ.get("GRADPULSE_DEVICE", "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = resolve_device()
