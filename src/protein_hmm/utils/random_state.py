"""Deterministic random number helpers."""

from __future__ import annotations

import numpy as np


def get_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
