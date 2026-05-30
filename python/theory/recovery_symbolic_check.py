r"""Symbolic verification of the Recovery Theorem (Theorem 2 / Appendix A).

Answers reviewer Q5: provides a machine-checkable, fully symbolic derivation
that the closed-form oPMM_alpha efficiency

    g(alpha; C) = (kappa2 * sigma2_a - c1a^2)
                  / (kappa2 * (sigma2_a - 2*alpha*nu_a*c1a + alpha^2*nu_a^2*kappa2))

collapses, at alpha = 3 on a symmetric distribution (kappa3 = 0), to the
classical Kunchenko PMM3 efficiency

    g(3; C) = 1 - gamma4^2 / (6 + 9*gamma4 + gamma6),

with gamma4 = kappa4/kappa2^2 and gamma6 = kappa6/kappa2^3.

This also confirms there is NO spurious ``kappa3^3'' term: the only kappa3
contribution to the sixth central moment is the +10*kappa3^2 term of the
moment-cumulant identity, which vanishes under kappa3 = 0.

Run (dev dependency ``sympy``, see Spec.md Section 12 PIVOT 21):

    PYTHONPATH=src python theory/recovery_symbolic_check.py
"""

from __future__ import annotations

import sympy as sp


def main() -> int:
    kappa2, kappa3, kappa4, kappa6 = sp.symbols(
        "kappa2 kappa3 kappa4 kappa6", positive=True
    )
    alpha = sp.Integer(3)

    # Central moments in terms of cumulants (zero mean). The sixth-moment
    # identity is mu6 = kappa6 + 15*kappa4*kappa2 + 10*kappa3^2 + 15*kappa2^3;
    # we keep kappa3 explicit to show exactly where it enters and then set it
    # to zero for the symmetric case.
    mu2 = kappa2
    mu4 = kappa4 + 3 * kappa2**2
    mu6 = kappa6 + 15 * kappa4 * kappa2 + 10 * kappa3**2 + 15 * kappa2**3

    # Fractional moments of the score basis at alpha = 3 (|eps|^3 = eps^3,
    # sign(eps)|eps|^3 = eps^3), on a symmetric law (odd central moments = 0).
    nu_a = mu2                         # E|eps|^{alpha-1} = E|eps|^2 = mu2
    c1a = mu4                          # E|eps|^{alpha+1} = E|eps|^4 = mu4
    mu_a = kappa3                      # E[sign(eps)|eps|^3] = E[eps^3] = kappa3
    sigma2_a = mu6 - mu_a**2           # E|eps|^{2alpha} - mu_a^2 = mu6 - kappa3^2

    g_alpha = (kappa2 * sigma2_a - c1a**2) / (
        kappa2 * (sigma2_a - 2 * alpha * nu_a * c1a + alpha**2 * nu_a**2 * kappa2)
    )

    # Symmetric case.
    g_sym = sp.simplify(g_alpha.subs(kappa3, 0))

    gamma4 = kappa4 / kappa2**2
    gamma6 = kappa6 / kappa2**3
    g_kunchenko = 1 - gamma4**2 / (6 + 9 * gamma4 + gamma6)

    difference = sp.simplify(g_sym - g_kunchenko)

    print("oPMM_3 efficiency (symmetric, kappa3=0):")
    sp.pprint(g_sym)
    print("\nKunchenko PMM3 efficiency:")
    sp.pprint(sp.simplify(g_kunchenko))
    print(f"\nSymbolic difference (must be 0): {difference}")
    assert difference == 0, "Recovery identity FAILED symbolically"
    print("PASS: g(3; C) == 1 - gamma4^2/(6 + 9*gamma4 + gamma6) symbolically.")

    # Numerical anchor: standard Laplace has gamma4 = 3, gamma6 = 30.
    g_laplace = g_kunchenko.subs({kappa2: 2, kappa4: 12, kappa6: 240})
    print(f"\nLaplace check: g(3) = {sp.nsimplify(g_laplace)} = {float(g_laplace):.6f} "
          f"(expected 54/63 = {54 / 63:.6f}).")
    assert sp.simplify(g_laplace - sp.Rational(54, 63)) == 0
    print("PASS: Laplace numerical anchor matches 54/63.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
