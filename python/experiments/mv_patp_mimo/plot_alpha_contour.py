"""Experiment 7: alpha*(gamma4) contour and heuristic ARE loss.

Computes and plots:
1. The oracle optimal alpha* as a function of excess kurtosis gamma4
   (equivalently, as a function of Student-t degrees of freedom).
2. The ARE loss when using the kurtosis heuristic instead of the oracle
   alpha*, for n=128 and n=512.

Expected theoretical results:
  alpha*(gamma4=0) = 2.0  (Gaussian optimum — quadratic score)
  alpha*(gamma4=3) ≈ 0.7–0.9  (Laplace — fractional score)
  alpha*(gamma4)   monotonically decreasing
  ARE_loss(n=512)  < 1% for gamma4 ∈ [1, 10]
  ARE_loss(n=128)  3–8% for gamma4 > 10

Output: ``manuscript/en/figures/fig_alpha_contour.pdf``
        (also saves ``fig_alpha_contour_data.npz`` for reproducibility)

CLI
---
  python -m experiments.mv_patp_mimo.plot_alpha_contour [options]

Options
-------
  --gamma4-max FLOAT    max gamma4 on x-axis (default: 15)
  --n-grid INT          number of gamma4 grid points (default: 50)
  --n-values INT [INT]  sample sizes for heuristic comparison (default: 128 512)
  --n-reps INT          replicates per (gamma4, n) cell (default: 1000)
  --seed INT            base random seed (default: 42)
  --output-dir PATH     output directory (default: manuscript/en/figures)
  --dry-run             smoke test: 5 points, 10 reps

Satisfies reviewer Exp 7 request for alpha*(gamma4) visualisation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def oracle_alpha_from_gamma4(
    gamma4: float,
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Find oracle alpha* for given excess kurtosis using large-n samples.

    Parameters
    ----------
    gamma4 : float
        Target excess kurtosis (gamma4 >= 0).
    n_samples : int
        Number of samples drawn for the oracle cumulant estimate.
    seed : int
        RNG seed.

    Returns
    -------
    (alpha_star, are_oracle) : tuple[float, float]
    """
    from mv_patp.core.alpha_optimizer import find_alpha_opt

    if gamma4 < 0.1:
        # Near-Gaussian: oracle is alpha=2.0 (quadratic score optimal)
        return 2.0, 1.0

    df = 6.0 / gamma4 + 4.0
    rng = np.random.default_rng(seed)
    eps = rng.standard_t(df=df, size=n_samples)
    try:
        alpha_star, g_min, _ = find_alpha_opt({"eps_white": eps})
        are_oracle = float(1.0 / max(float(g_min), 1e-8))
        return float(alpha_star), are_oracle
    except Exception as exc:
        logger.warning("oracle alpha failed at gamma4=%.2f: %s", gamma4, exc)
        return 2.0, 1.0


def heuristic_are_loss(
    gamma4: float,
    n: int,
    n_reps: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate ARE loss from using kurtosis heuristic vs oracle alpha*.

    Parameters
    ----------
    gamma4 : float
        Excess kurtosis of the target distribution.
    n : int
        Sample size for the heuristic alpha estimate.
    n_reps : int
        Number of Monte Carlo replicates.
    seed : int
        Base seed.

    Returns
    -------
    (are_loss_pct, are_heuristic_mean) : tuple[float, float]
        ARE_loss_pct = (ARE_oracle - ARE_heuristic_mean) / ARE_oracle * 100
    """
    from mv_patp.core.alpha_optimizer import find_alpha_opt
    from mv_patp.utils.cumulants import closed_form_g_alpha, empirical_cumulants

    if gamma4 < 0.1:
        return 0.0, 1.0

    df = 6.0 / gamma4 + 4.0

    # Oracle: large-sample cumulants; retry with different seeds if degenerate.
    are_oracle = float("nan")
    eps_large = None
    for oracle_trial in range(5):
        rng_oracle = np.random.default_rng(seed + 999_999 + oracle_trial * 17)
        eps_large_trial = rng_oracle.standard_t(df=df, size=100_000)
        try:
            alpha_oracle_t, g_oracle_t, _ = find_alpha_opt({"eps_white": eps_large_trial})
            if 0.0 < float(g_oracle_t) < 1.0 and not (0.92 < float(alpha_oracle_t) < 1.08):
                are_oracle = float(1.0 / float(g_oracle_t))
                eps_large = eps_large_trial
                break
        except Exception:
            pass
    if np.isnan(are_oracle) or eps_large is None:
        return float("nan"), float("nan")

    # Heuristic: estimate alpha from n samples, evaluate g at oracle cumulants
    are_heuristic_arr = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed + rep)
        eps_sample = rng.standard_t(df=df, size=n)
        try:
            alpha_hat, _, _ = find_alpha_opt({"eps_white": eps_sample})
            # Evaluate g(alpha_hat) at the oracle (large-sample) cumulants
            C_oracle = empirical_cumulants(eps_large, float(alpha_hat))
            g_at_hat = closed_form_g_alpha(float(alpha_hat), C_oracle)
            are_at_hat = float(1.0 / max(float(g_at_hat), 1e-8))
            are_heuristic_arr.append(are_at_hat)
        except Exception:
            pass

    if not are_heuristic_arr:
        return float("nan"), float("nan")

    # Clip ARE_heuristic to [0, 2*ARE_oracle] to exclude degenerate alpha estimates
    # where the estimator overshoots (g >> 1 near alpha=1 degenerate band).
    are_arr = np.array(are_heuristic_arr)
    are_arr = np.clip(are_arr, 0.0, 2.0 * are_oracle)
    # Use median to be robust to the degenerate-alpha tail.
    are_heuristic_median = float(np.median(are_arr))
    are_loss_pct = max(
        (are_oracle - are_heuristic_median) / max(are_oracle, 1e-8) * 100.0, 0.0
    )
    return are_loss_pct, are_heuristic_median


def run(
    gamma4_max: float = 15.0,
    n_grid: int = 50,
    n_values: list[int] | None = None,
    n_reps: int = 1000,
    seed: int = 42,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the full alpha contour computation and save outputs.

    Returns a dict of computed arrays for testing/inspection.
    """
    if n_values is None:
        n_values = [128, 512]
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "manuscript" / "en" / "figures"

    if dry_run:
        n_grid = 5
        n_reps = 10
        logger.info("Dry run: n_grid=%d, n_reps=%d", n_grid, n_reps)

    gamma4_grid = np.linspace(0.1, gamma4_max, n_grid)

    logger.info(
        "Computing oracle alpha* across %d gamma4 points (n_samples=100k, seed=%d)...",
        n_grid, seed,
    )
    alpha_oracle = np.zeros(n_grid)
    are_oracle = np.zeros(n_grid)
    for i, g4 in enumerate(gamma4_grid):
        a, are = oracle_alpha_from_gamma4(g4, n_samples=100_000, seed=seed)
        alpha_oracle[i] = a
        are_oracle[i] = are
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d: gamma4=%.2f -> alpha*=%.3f, ARE=%.3f",
                        i + 1, n_grid, g4, a, are)

    are_loss = {}
    are_heuristic = {}
    for n in n_values:
        logger.info("Computing ARE loss for n=%d (%d reps per cell)...", n, n_reps)
        loss_arr = np.zeros(n_grid)
        heur_arr = np.zeros(n_grid)
        for i, g4 in enumerate(gamma4_grid):
            loss, heur = heuristic_are_loss(g4, n=n, n_reps=n_reps, seed=seed + i)
            loss_arr[i] = loss
            heur_arr[i] = heur
        are_loss[n] = loss_arr
        are_heuristic[n] = heur_arr
        logger.info("  n=%d: mean ARE loss=%.2f%%", n, float(np.nanmean(loss_arr)))

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "fig_alpha_contour_data.npz"
    save_dict: dict[str, np.ndarray] = {
        "gamma4_grid": gamma4_grid,
        "alpha_oracle": alpha_oracle,
        "are_oracle": are_oracle,
    }
    for n in n_values:
        save_dict[f"are_loss_n{n}"] = are_loss[n]
        save_dict[f"are_heuristic_n{n}"] = are_heuristic[n]
    np.savez(npz_path, **save_dict)
    logger.info("Data saved: %s", npz_path)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Left panel: alpha*(gamma4)
        ax1.plot(gamma4_grid, alpha_oracle, "k-", lw=2, label=r"$\alpha^*(\gamma_4)$")
        ax1.axhline(2.0, color="gray", ls="--", lw=1, label=r"$\alpha=2$ (Gaussian)")
        ax1.axhline(1.0, color="gray", ls=":", lw=1, label=r"$\alpha=1$ (Laplace)")
        ax1.set_xlabel(r"Excess kurtosis $\gamma_4$")
        ax1.set_ylabel(r"Optimal $\alpha^*$")
        ax1.set_title(r"Oracle $\alpha^*$ vs tail heaviness")
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 3)
        ax1.grid(True, alpha=0.3)

        # Right panel: ARE loss (median, robust to degenerate-alpha outliers)
        colors = ["C0", "C1", "C2", "C3"]
        for ci, n in enumerate(n_values):
            ax2.plot(
                gamma4_grid, are_loss[n],
                color=colors[ci], lw=2,
                label=f"$n={n}$",
            )
        ax2.axhline(1.0, color="gray", ls="--", lw=1, label="1% threshold")
        ax2.set_xlabel(r"Excess kurtosis $\gamma_4$")
        ax2.set_ylabel(r"ARE loss, median (\%)")
        ax2.set_title(r"ARE loss: heuristic $\hat\alpha$ vs oracle $\alpha^*$")
        ax2.legend(fontsize=8)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf_path = output_dir / "fig_alpha_contour.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved: %s", pdf_path)
    except ImportError:
        logger.warning("matplotlib not available; skipping figure generation.")

    return {
        "gamma4_grid": gamma4_grid,
        "alpha_oracle": alpha_oracle,
        "are_oracle": are_oracle,
        "are_loss": are_loss,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute alpha*(gamma4) oracle contour and ARE-loss figure."
    )
    parser.add_argument("--gamma4-max", type=float, default=15.0)
    parser.add_argument("--n-grid", type=int, default=50)
    parser.add_argument("--n-values", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        gamma4_max=args.gamma4_max,
        n_grid=args.n_grid,
        n_values=args.n_values,
        n_reps=args.n_reps,
        seed=args.seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
