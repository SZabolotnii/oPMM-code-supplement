"""Baseline channel estimators for ARE comparison.

Each module exposes ``fit(Y, S, ...) -> BaselineResult`` as the canonical
interface so that ``experiments/mv_patp_mimo/run.py`` can dispatch them
uniformly. See ``design.md`` Section "L2 -- Estimators and inference".
"""

from mv_patp.baseline import (
    huber_irls,
    matrix_f_mle,
    matrix_pmm2,
    matrix_pmm3,
    matrix_t_mle,
    ols,
    student_t_elementwise,
    tyler_opmm,
    whitened_huber,
    wishart_mle,
)
from mv_patp.baseline._types import BaselineResult

__all__ = [
    "BaselineResult",
    "huber_irls",
    "matrix_f_mle",
    "matrix_pmm2",
    "matrix_pmm3",
    "matrix_t_mle",
    "ols",
    "student_t_elementwise",
    "tyler_opmm",
    "whitened_huber",
    "wishart_mle",
]
