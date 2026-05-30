"""Middleton Class A canonical impulsive-noise data-generating process.

Middleton's Class A model represents the instantaneous amplitude of symmetric
impulsive interference (co-channel devices, switching electronics, atmospheric
noise) as a Poisson-weighted mixture of zero-mean Gaussians. It is the
industry-standard canonical model for symmetric impulsive RF noise and is the
``Middleton Class A`` backbone reference invoked in the symmetric primary scope
of this paper (Spec.md §12 PIVOT 10/14).

Model
-----
For each scalar sample we draw an impulse count ``m ~ Poisson(A)`` and, given
``m``, draw ``x | m ~ N(0, V_m)`` with the canonical conditional variance

    V_m = sigma_tot^2 * (m / A + Gamma_prime) / (1 + Gamma_prime),

where ``A`` is the *impulsive index* (overlap index; small ``A`` => sparser,
heavier impulses) and ``Gamma_prime`` is the *Gaussian-to-impulsive intensity
ratio* (small ``Gamma_prime`` => more impulsive). Because ``E[m] = A`` we have
``E[V_m] = sigma_tot^2``, so the sampler is unit-variance by construction at
the default ``sigma_tot = 1``.

As a scale mixture of normals the distribution is symmetric with finite
moments; its excess kurtosis admits the closed form

    gamma_4 = 3 / (A * (1 + Gamma_prime)^2),

which the helper :func:`middleton_excess_kurtosis` returns. This places Class A
in the same symmetric heavy-tailed family as the Student-t cells of §V while
remaining a genuinely distinct, physically motivated DGP.

References
----------
Middleton, D. (1977). Statistical-physical models of electromagnetic
interference. *IEEE Trans. Electromagn. Compat.*, EMC-19(3), 106-127.

See also `Spec.md §4.2` (MIMO noise model) and `Spec.md §12` PIVOT 19.
"""

from __future__ import annotations

import numpy as np


def middleton_excess_kurtosis(A: float, Gamma_prime: float) -> float:
    """Closed-form excess kurtosis gamma_4 of Middleton Class A noise.

    Parameters
    ----------
    A : float
        Impulsive index (overlap index), ``A > 0``. Smaller ``A`` is more
        impulsive (heavier tails).
    Gamma_prime : float
        Gaussian-to-impulsive intensity ratio, ``Gamma_prime > 0``.

    Returns
    -------
    float
        Excess kurtosis ``gamma_4 = 3 / (A (1 + Gamma_prime)^2)``.
    """
    return 3.0 / (A * (1.0 + Gamma_prime) ** 2)


def sample_middleton_class_a(
    k: int,
    n: int,
    A: float,
    Gamma_prime: float,
    rng: np.random.Generator,
    *,
    sigma_tot: float = 1.0,
) -> np.ndarray:
    """Sample a ``(k, n)`` matrix of i.i.d. Middleton Class A noise.

    Each entry is an independent scalar Class A draw via the Poisson-Gaussian
    scale-mixture construction described in the module docstring. The result is
    unit-variance (per entry) when ``sigma_tot = 1``.

    Parameters
    ----------
    k, n : int
        Output shape (receive antennas x training length).
    A : float
        Impulsive index, ``A > 0``.
    Gamma_prime : float
        Gaussian-to-impulsive intensity ratio, ``Gamma_prime > 0``.
    rng : numpy.random.Generator
        Per the project RNG convention (``np.random.default_rng(seed)``).
    sigma_tot : float, optional
        Total per-entry standard deviation (default 1.0).

    Returns
    -------
    np.ndarray, shape (k, n), float64
        Symmetric heavy-tailed noise matrix.
    """
    if A <= 0.0:
        raise ValueError(f"Middleton Class A impulsive index A must be > 0, got {A}.")
    if Gamma_prime <= 0.0:
        raise ValueError(
            f"Middleton Class A intensity ratio Gamma_prime must be > 0, got {Gamma_prime}."
        )
    m = rng.poisson(lam=A, size=(k, n)).astype(np.float64)
    var_m = (sigma_tot**2) * (m / A + Gamma_prime) / (1.0 + Gamma_prime)
    return rng.standard_normal((k, n)) * np.sqrt(var_m)
