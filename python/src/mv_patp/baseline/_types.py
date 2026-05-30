"""Shared result type for baseline estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    """Result of fitting a baseline channel estimator.

    Attributes
    ----------
    H : np.ndarray
        Estimated channel matrix of shape ``(k, p)``.
    residuals : np.ndarray
        Final residual matrix ``Y - H @ S`` of shape ``(k, n)``.
    converged : bool
        True if the iterative procedure converged before ``max_iter`` (always
        True for closed-form estimators such as OLS).
    n_iter : int
        Number of iterations actually performed (0 for closed-form).
    log_history : dict
        Optional per-iteration diagnostics (frob deltas, weighted negative
        log-likelihoods, etc.). Empty dict for closed-form estimators.
    """

    H: np.ndarray
    residuals: np.ndarray
    converged: bool = True
    n_iter: int = 0
    log_history: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.log_history is None:
            self.log_history = {}
