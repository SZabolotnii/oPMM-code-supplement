"""Shared Monte Carlo helpers for the three Phase-B gate tests."""

from __future__ import annotations

import numpy as np


def gal_symmetric_sample(
    n: int, alpha_gal: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` samples from a symmetric scalar GAL with shape ``alpha_gal``.

    GAL representation: ``X = sqrt(W) * Z`` with ``W ~ Gamma(alpha_gal, 1)``
    and ``Z ~ N(0, 1)``. This matches ``theory/opmm_monte_carlo.py``.
    """
    W = rng.gamma(shape=alpha_gal, scale=1.0, size=n)
    Z = rng.standard_normal(n)
    return np.sqrt(W) * Z
