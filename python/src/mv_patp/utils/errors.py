"""Typed error classes for mv_patp.

All domain-specific exceptions are defined here.
Never raise plain ValueError or RuntimeError for these conditions.
See AGENTS.md §5.3.
"""


class AlphaOutOfRangeError(ValueError):
    """Raised when α ∉ [0.3, 3.0]."""


class IdentifiabilityViolation(RuntimeError):
    """Raised when tr(Γ̂) drift exceeds tolerance."""


class BrentNoConvergence(RuntimeError):
    """Raised when Brent's method fails; α-optimizer falls back to grid."""


class EMDivergence(RuntimeError):
    """Raised when outer EM did not converge within max_iter."""


class NumericalInstabilityError(ArithmeticError):
    """Raised when NaN or Inf is detected in any returned array."""
