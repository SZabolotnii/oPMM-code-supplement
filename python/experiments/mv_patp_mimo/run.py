"""Monte Carlo orchestrator for mv-patp-mimo experiments.

Loads a YAML configuration, expands the experimental grid
``(k, n, alpha_MAL, M, Sigma, replicate)``, dispatches each cell to the
configured estimators, and writes per-cell CSV plus a reproducibility
``manifest.json`` per AGENTS.md Section 6.1.

Usage
-----
``python -m experiments.mv_patp_mimo.run --config <path-to-yaml>
[--output-dir <dir>] [--seed 42] [--dry-run]``

The estimator dispatch table currently raises ``NotImplementedError`` for
estimators that have not yet landed in ``src/mv_patp/`` (the EM-PATP family
is gated on Phase B in tasks.md). The ``--dry-run`` flag exercises the
manifest writer and the YAML loader without invoking estimators.

Satisfies Requirements R11.1 (single-command MC regeneration),
R11.3 (manifest with SHA-256 + git commit + package versions),
R11.4 (deterministic seeds via ``np.random.default_rng``).

Task: Section 15.2 (run.py orchestrator).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib.metadata as _meta
import itertools
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _worker(args_tuple: tuple) -> dict:
    """Top-level helper so multiprocessing spawn can pickle it."""
    cell, seed = args_tuple
    return run_mc_cell(cell, seed=seed)


# --- Estimator registry -------------------------------------------------------
# Maps estimator name (as used in config YAML) to a thunk that returns the
# fitting callable. Lazy imports avoid hard-failing on partially-implemented
# Kiro work in src/mv_patp/core/.

_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register(name: str):
    def _wrap(fn):
        _ESTIMATOR_REGISTRY[name] = fn
        return fn
    return _wrap


@register("ols")
def _load_ols():
    from mv_patp.baseline import ols
    return ols.fit


@register("wishart_mle")
def _load_wishart():
    from mv_patp.baseline import wishart_mle
    return wishart_mle.fit


@register("matrix_t_mle")
def _load_matrix_t():
    from mv_patp.baseline import matrix_t_mle
    return matrix_t_mle.fit


@register("matrix_f_mle")
def _load_matrix_f():
    from mv_patp.baseline import matrix_f_mle
    return matrix_f_mle.fit


@register("matrix_pmm2")
def _load_matrix_pmm2():
    from mv_patp.baseline import matrix_pmm2
    return matrix_pmm2.fit


@register("matrix_pmm3")
def _load_matrix_pmm3():
    from mv_patp.baseline import matrix_pmm3
    return matrix_pmm3.fit


@register("opmm_alpha")
def _load_opmm():
    from mv_patp.core import em_patp
    return em_patp.em_patp_estimate


@register("opmm_alpha_exact")
def _load_opmm_exact():
    from mv_patp.core import em_patp_exact
    return em_patp_exact.em_patp_exact_estimate


@register("huber_irls")
def _load_huber_irls():
    from mv_patp.baseline import huber_irls
    return huber_irls.fit


@register("whitened_huber")
def _load_whitened_huber():
    from mv_patp.baseline import whitened_huber
    return whitened_huber.fit


@register("student_t_elementwise")
def _load_student_t_elem():
    from mv_patp.baseline import student_t_elementwise
    return student_t_elementwise.fit


@register("tyler_opmm")
def _load_tyler_opmm():
    from mv_patp.baseline import tyler_opmm
    return tyler_opmm.fit


# --- eps_min sensitivity variants (Exp 5) ------------------------------------
# Each variant fixes alpha_p=0.5 (worst case for score floor sensitivity)
# and sweeps eps_min over four orders of magnitude.

import functools as _functools


def _make_opmm_eps(eps_val: float):
    from mv_patp.core.em_patp import em_patp_estimate
    return _functools.partial(
        em_patp_estimate,
        eps_min=eps_val,
        alpha_p_fixed=True,
        alpha_p_init=0.5,
    )


@register("opmm_alpha_eps1e10")
def _load_opmm_eps1e10():
    return _make_opmm_eps(1e-10)


@register("opmm_alpha_eps1e8")
def _load_opmm_eps1e8():
    return _make_opmm_eps(1e-8)


@register("opmm_alpha_eps1e6")
def _load_opmm_eps1e6():
    return _make_opmm_eps(1e-6)


@register("opmm_alpha_eps1e4")
def _load_opmm_eps1e4():
    return _make_opmm_eps(1e-4)


# --- Manifest writer ----------------------------------------------------------

_MANIFEST_PACKAGES = (
    "numpy", "scipy", "pandas", "pyyaml", "h5py",
    "matplotlib", "mpmath",
)


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    return sha


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _MANIFEST_PACKAGES:
        try:
            versions[pkg] = _meta.version(pkg)
        except _meta.PackageNotFoundError:
            versions[pkg] = "absent"
    return versions


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    output_dir: Path,
    config_path: Path,
    seed: int,
    wall_clock_seconds: float,
    pass_fail_summary: dict[str, str],
    input_data_paths: list[Path] | None = None,
) -> Path:
    """Write reproducibility manifest per AGENTS.md Section 6.1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "seed": seed,
        "config_path": str(config_path.resolve()),
        "input_data_sha256": {
            str(p): _file_sha256(p) for p in (input_data_paths or [])
        },
        "wall_clock_seconds": wall_clock_seconds,
        "pass_fail_summary": pass_fail_summary,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


# --- Config loader and grid expansion -----------------------------------------

@dataclasses.dataclass
class CellSpec:
    """One cell of the Monte Carlo grid.

    Two data sources are supported:

    * ``data_source="synthetic"`` (default) — sample matrix-MAL or Student-t
      noise via ``_draw_noise``; ``H_truth`` is drawn fresh per replicate.
    * ``data_source="deepmimo"`` — pull a ray-traced channel via
      ``data_gen.deepmimo_loader.load_deepmimo`` at the requested
      ``scenario`` and ``snr_db``; matrix-MAL noise is added at ``alpha_MAL``.
      For DeepMIMO cells, ``k`` is determined by the loader (``2 * k_ant``
      after Re/Im stack) — the cell-level ``k`` is ignored and recorded for
      provenance only.
    """
    k: int
    n: int
    alpha_MAL: float
    p: int
    M_norm: float
    sigma_structure: str
    replicates: int
    estimator: str
    data_source: str = "synthetic"
    scenario: str = ""
    snr_db: float = float("nan")

    def cell_id(self) -> str:
        if self.data_source == "deepmimo":
            return (
                f"deepmimo_{self.scenario}_snr{self.snr_db}_n{self.n}_p{self.p}"
                f"_alpha{self.alpha_MAL}_sigma{self.sigma_structure}"
                f"_est{self.estimator}"
            )
        return (
            f"k{self.k}_n{self.n}_alpha{self.alpha_MAL}_p{self.p}"
            f"_M{self.M_norm}_sigma{self.sigma_structure}_est{self.estimator}"
        )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def expand_grid(config: dict[str, Any]) -> list[CellSpec]:
    """Expand a YAML config into a flat list of cells.

    Branches on ``config["data_source"]``: the default ``"synthetic"`` keeps
    the existing matrix-MAL / Student-t grid (``k × n × alpha_MAL × p ×
    M_norm × sigma_structure × estimators``); ``"deepmimo"`` instead expands
    over ``scenarios × snr_db × n × p × alpha_MAL × estimators``.
    """
    data_source = str(config.get("data_source", "synthetic"))
    grid = config.get("grid", {})
    estimators: list[str] = config.get("estimators", [])
    replicates: int = int(config.get("replicates", 100))
    cells: list[CellSpec] = []

    if data_source == "deepmimo":
        for scenario, snr_db, n, p, alpha_MAL, sigma_structure, estimator in (
            itertools.product(
                grid.get("scenarios", []),
                grid.get("snr_db", [10.0]),
                grid.get("n", [256]),
                grid.get("p", [2]),
                grid.get("alpha_MAL", [1.0]),
                grid.get("sigma_structure", ["student_t_4"]),
                estimators,
            )
        ):
            cells.append(CellSpec(
                k=0, n=int(n), alpha_MAL=float(alpha_MAL), p=int(p),
                M_norm=0.0, sigma_structure=str(sigma_structure),
                replicates=replicates, estimator=str(estimator),
                data_source="deepmimo",
                scenario=str(scenario), snr_db=float(snr_db),
            ))
        return cells

    for k, n, alpha_MAL, p, M_norm, sigma_structure, estimator in itertools.product(
        grid.get("k", [2]),
        grid.get("n", [512]),
        grid.get("alpha_MAL", [1.0]),
        grid.get("p", [2]),
        grid.get("M_norm", [0.0]),
        grid.get("sigma_structure", ["I"]),
        estimators,
    ):
        cells.append(CellSpec(
            k=int(k), n=int(n), alpha_MAL=float(alpha_MAL), p=int(p),
            M_norm=float(M_norm), sigma_structure=str(sigma_structure),
            replicates=replicates, estimator=str(estimator),
        ))
    return cells


# --- Cell runner -------------------------------------------------------------

def _draw_noise(
    cell: CellSpec, rng: np.random.Generator
) -> np.ndarray:
    """Sample one ``(k, n)`` noise realisation per the cell DGP.

    The cell encodes the noise family via the ``estimator``-agnostic
    fields ``alpha_MAL`` (matrix MAL shape) and ``sigma_structure``.
    For the misspec primary scope (Spec §12 PIVOT 14) we extend the
    convention: ``sigma_structure`` values that begin with ``"student_t_"``
    select Student-t noise with the indicated degrees of freedom, e.g.
    ``"student_t_4"``.
    """
    k, n = cell.k, cell.n
    if cell.sigma_structure.startswith("student_t_"):
        df = float(cell.sigma_structure.split("_")[-1])
        z = rng.standard_normal((k, n))
        chi2 = rng.chisquare(df=df, size=(k, n))
        return z / np.sqrt(chi2 / df)
    if cell.sigma_structure.startswith("ar1_gamma_rho"):
        # "ar1_gamma_rho0p5" → rho=0.5; "ar1_gamma_rho0p9" → rho=0.9
        rho_str = cell.sigma_structure.split("rho")[-1].replace("p", ".")
        rho = float(rho_str)
        i_idx, j_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        Gamma_ar1 = rho ** np.abs(i_idx - j_idx)          # (k, k) Toeplitz
        L = np.linalg.cholesky(Gamma_ar1 + 1e-9 * np.eye(k))
        return L @ rng.standard_normal((k, n))             # correlated Gaussian
    if cell.sigma_structure.startswith("middleton_"):
        # Middleton Class A canonical impulsive noise (Spec §12 PIVOT 19).
        # "middleton_A0p5_G0p1" → impulsive index A=0.5, intensity ratio G'=0.1.
        from mv_patp.data_gen.middleton import sample_middleton_class_a

        parts = cell.sigma_structure.split("_")
        A_idx = float(parts[1][1:].replace("p", "."))        # strip leading "A"
        Gamma_prime = float(parts[2][1:].replace("p", "."))  # strip leading "G"
        return sample_middleton_class_a(
            k, n, A=A_idx, Gamma_prime=Gamma_prime, rng=rng
        )
    # Default: matrix MAL noise via the package sampler.
    from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal
    arr = sample_matrix_mal(
        n_samples=1, k=k, n=n,
        alpha=max(cell.alpha_MAL, (k - 1) / 2.0 + 0.05),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    return arr[0]


def run_mc_cell(
    cell: CellSpec,
    seed: int,
) -> dict[str, Any]:
    """Execute one MC cell: ``replicates`` fits of ``cell.estimator``.

    Returns aggregated MSE / bias / wall-clock statistics in the
    schema consumed by ``verify.py`` and ``plot.py``.
    """
    fit_factory = _ESTIMATOR_REGISTRY.get(cell.estimator)
    if fit_factory is None:
        raise ValueError(
            f"Unknown estimator '{cell.estimator}'. "
            f"Registered: {sorted(_ESTIMATOR_REGISTRY)}"
        )
    try:
        fit = fit_factory()
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "Estimator '%s' not yet importable (%s); skipping cell.",
            cell.estimator, exc,
        )
        return {
            "cell_id": cell.cell_id(),
            "estimator": cell.estimator,
            "k": cell.k, "p": cell.p, "n": cell.n,
            "alpha_MAL": cell.alpha_MAL,
            "sigma_structure": cell.sigma_structure,
            "M_norm": cell.M_norm,
            "data_source": cell.data_source,
            "scenario": cell.scenario,
            "snr_db": cell.snr_db,
            "status": "skipped_not_implemented",
            "mse_mean": float("nan"),
            "mse_std": float("nan"),
            "n_replicates": 0,
            "wall_clock_mean_s": float("nan"),
        }

    sq_errs: list[float] = []
    h_estimates: list[np.ndarray] = []
    wall_clocks: list[float] = []
    n_failures = 0

    for rep in range(cell.replicates):
        rng = np.random.default_rng(seed + rep)
        if cell.data_source == "deepmimo":
            from mv_patp.data_gen.deepmimo_loader import load_deepmimo
            # cell.sigma_structure controls the noise family. "I" -> matrix-MAL
            # at cell.alpha_MAL; "student_t_<df>" -> heavy-tailed misspec noise.
            noise_kind = (
                cell.sigma_structure
                if cell.sigma_structure.startswith("student_t_")
                else "matrix_mal"
            )
            Y, S, H_truth = load_deepmimo(
                scenario=cell.scenario, snr_db=cell.snr_db,
                n_train=cell.n, p=cell.p,
                alpha_mal=cell.alpha_MAL, noise_kind=noise_kind,
                seed=seed + rep,
            )
        else:
            H_truth = rng.standard_normal((cell.k, cell.p))
            S = rng.standard_normal((cell.p, cell.n))
            N = _draw_noise(cell, rng)
            Y = H_truth @ S + N
        t0 = time.perf_counter()
        try:
            res = fit(Y, S)
            H_hat = getattr(res, "H", None)
            if H_hat is None:
                # Some baselines return BaselineResult; em_patp returns
                # EMPATPResult. Both expose .H so this is safe.
                raise AttributeError(".H field missing on result.")
            H_hat = np.asarray(H_hat)
            sq_err = float(np.linalg.norm(H_hat - H_truth, ord="fro") ** 2)
            sq_errs.append(sq_err)
            h_estimates.append(H_hat)
            wall_clocks.append(time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cell %s replicate %d failed: %s",
                           cell.cell_id(), rep, exc)
            n_failures += 1

    base_row: dict[str, Any] = {
        "cell_id": cell.cell_id(),
        "estimator": cell.estimator,
        "k": cell.k, "p": cell.p, "n": cell.n,
        "alpha_MAL": cell.alpha_MAL,
        "sigma_structure": cell.sigma_structure,
        "M_norm": cell.M_norm,
        "data_source": cell.data_source,
        "scenario": cell.scenario,
        "snr_db": cell.snr_db,
    }
    if not sq_errs:
        return {
            **base_row,
            "status": "all_replicates_failed",
            "mse_mean": float("nan"),
            "mse_std": float("nan"),
            "n_replicates": 0,
            "wall_clock_mean_s": float("nan"),
        }

    mse_arr = np.asarray(sq_errs)
    return {
        **base_row,
        "status": "ok" if n_failures == 0 else "partial",
        "mse_mean": float(mse_arr.mean()),
        "mse_std": float(mse_arr.std(ddof=1)) if len(mse_arr) > 1 else 0.0,
        "n_replicates": len(sq_errs),
        "n_failures": n_failures,
        "wall_clock_mean_s": float(np.mean(wall_clocks)),
    }


# --- CLI ---------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mv-patp-mimo Monte Carlo experiments.",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML configuration file.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/<run_id>).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Top-level seed for np.random.default_rng.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip estimator dispatch; only validate config "
                             "and write manifest.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default 1 = "
                             "sequential). Use --workers 12 on M3 Max for ~5x "
                             "speedup. Note: parallel mode uses spawn; per-cell "
                             "seed ordering is non-deterministic.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    cells = expand_grid(config)
    logger.info("Loaded config %s with %d cells.", args.config, len(cells))

    run_id = config.get("run_id") or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "results" / run_id
    )

    t0 = time.perf_counter()
    cell_results: list[dict[str, Any]] = []
    pass_fail: dict[str, str] = {}

    if args.dry_run:
        pass_fail["dry_run"] = "config_valid"
        logger.info("Dry run: skipping cell dispatch.")
    elif args.workers > 1:
        logger.info("Parallel dispatch: %d workers (spawn).", args.workers)
        import os
        from multiprocessing import get_context
        # Limit BLAS/OpenMP threads per worker to 1 so process-level
        # parallelism doesn't compete with numpy's internal threading.
        blas_env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        os.environ.update(blas_env)
        ctx = get_context("spawn")
        try:
            with ctx.Pool(processes=args.workers) as pool:
                results = pool.map(_worker, [(c, args.seed) for c in cells])
            for result in results:
                cell_results.append(result)
                pass_fail[result["cell_id"]] = result["status"]
        except Exception as exc:
            logger.exception("Pool dispatch failed (%s); falling back to sequential.", exc)
            for cell in cells:
                try:
                    result = run_mc_cell(cell, seed=args.seed)
                    cell_results.append(result)
                    pass_fail[cell.cell_id()] = result["status"]
                except Exception as cell_exc:
                    logger.exception("Cell %s failed: %s", cell.cell_id(), cell_exc)
                    pass_fail[cell.cell_id()] = f"error: {type(cell_exc).__name__}"
    else:
        for cell in cells:
            try:
                result = run_mc_cell(cell, seed=args.seed)
                cell_results.append(result)
                pass_fail[cell.cell_id()] = result["status"]
            except Exception as exc:  # log-and-continue per AGENTS Section 5.4
                logger.exception("Cell %s failed: %s", cell.cell_id(), exc)
                pass_fail[cell.cell_id()] = f"error: {type(exc).__name__}"

    wall_clock = time.perf_counter() - t0

    # Write CSV (always) and manifest (always).
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cells.csv"
    if cell_results:
        import pandas as pd  # local import to keep dry-run lightweight
        pd.DataFrame(cell_results).to_csv(csv_path, index=False)
    else:
        csv_path.write_text("cell_id,status,mse_mean,mse_std,n_replicates\n")

    manifest_path = write_manifest(
        output_dir=output_dir,
        config_path=args.config,
        seed=args.seed,
        wall_clock_seconds=wall_clock,
        pass_fail_summary=pass_fail,
    )
    logger.info("Manifest: %s", manifest_path)
    logger.info("Cells CSV: %s", csv_path)

    # Exit code reflects whether any cell errored.
    return 0 if not any(v.startswith("error:") for v in pass_fail.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
