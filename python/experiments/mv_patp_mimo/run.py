"""Monte Carlo orchestrator for mv-patp-mimo experiments.

Loads a YAML configuration, expands the experimental grid
``(k, n, alpha_MAL, M, Sigma, replicate)``, dispatches each cell to the
configured estimators, and writes per-cell CSV plus a reproducibility
``manifest.json`` per AGENTS.md Section 6.1.

Usage
-----
``python -m experiments.mv_patp_mimo.run --config <path-to-yaml>
[--output-dir <dir>] [--seed 42] [--dry-run] [--workers N]
[--per-replicate] [--replicates R]``

R1 revision (DSP-D-26-03654): the MAL shape is never clipped — a MAL cell
with ``alpha_MAL <= (k-1)/2`` is reported with ``status="invalid_alpha"``
and not run; every row records ``alpha_used``. ``grid.alpha_MAL_offset``
gives a per-k valid grid ``alpha = (k-1)/2 + offset``. Non-MAL noise is not
expanded over the inert alpha axis. ``--per-replicate`` writes ``reps.csv``
(paired design: data depend only on ``(data_id, seed, rep)``) with the
fit diagnostics ``converged, n_outer, n_inner, alpha_hat, guard_triggered,
stop_reason, c_hat`` (``c_hat``: Huber constant chosen by
``huber_adaptive_c``, Rv1-C11; NaN for other estimators); ``cells.csv`` adds ``converged_rate`` and ``guard_rate``.
``opmm_alpha`` runs with the R1 return-only guard (the iteration is the
submitted one; only a final iterate whose fixed Huber criterion exceeds
1.05 x its OLS value is replaced by the best recorded iterate);
``opmm_alpha_unguarded`` is the as-submitted fit.

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
    cell, seed, *rest = args_tuple
    per_replicate = bool(rest[0]) if rest else False
    return run_mc_cell(cell, seed=seed, per_replicate=per_replicate)


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


@register("opmm_alpha_unguarded")
def _load_opmm_unguarded():
    """As-submitted oPMM_alpha: no return guard, no per-step guard, no alpha freeze."""
    import functools
    from mv_patp.core import em_patp
    return functools.partial(
        em_patp.em_patp_estimate, return_guard=False, divergence_guard=False,
        alpha_freeze_after=None,
    )


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


# --- R1 fixed-alpha variants (Rv1-C7: fixed vs adaptive alpha) ---------------
# Same estimator as ``opmm_alpha`` (R1 return-only guard on), with the fractional order
# held at a fixed value instead of the in-loop heuristic. Values (theory.md,
# [Q] table): 0.3 = heuristic limit for t3/t4; 0.536 = heuristic limit for t5
# (~ argmin g for t3, 0.50); 0.77 = argmin g for t4; 2.0 = the fixed value
# quoted in the submitted supplement. Values near alpha = 1 are avoided
# (k1*, k2* ~ 1/(alpha-1), see safeguards.md). Additional names only; no
# existing estimator changes.

def _make_opmm_fixed(alpha_val: float):
    from mv_patp.core.em_patp import em_patp_estimate
    return _functools.partial(
        em_patp_estimate, alpha_p_fixed=True, alpha_p_init=alpha_val,
    )


for _fixed_name, _fixed_val in (
    ("opmm_alpha_fixed0p3", 0.3),
    ("opmm_alpha_fixed0p536", 0.536),
    ("opmm_alpha_fixed0p77", 0.77),
    ("opmm_alpha_fixed2p0", 2.0),
):
    _ESTIMATOR_REGISTRY[_fixed_name] = _functools.partial(_make_opmm_fixed, _fixed_val)
del _fixed_name, _fixed_val


# --- R1 Huber tuning variants (Rv1-C11: tuned instead of fixed c = 1.345) -----
# Fixed-c variants are ``huber_irls.fit`` with only ``c_efficiency`` changed
# (same IRLS, MAD scale, OLS start, row-wise fits); ``huber_c1p345`` is
# ``huber_irls`` itself. ``huber_adaptive_c`` chooses c per fit by minimum
# plug-in asymptotic variance on the standardised residuals of the c = 1.345
# fit (rule in ``mv_patp.baseline.huber_adaptive_c``) and records ``c_hat``.

def _make_huber_fixed(c_val: float):
    from mv_patp.baseline import huber_irls
    return _functools.partial(huber_irls.fit, c_efficiency=c_val)


HUBER_FIXED_C: dict[str, float] = {
    "huber_c0p7": 0.7,
    "huber_c1p0": 1.0,
    "huber_c1p345": 1.345,
    "huber_c1p7": 1.7,
    "huber_c2p0": 2.0,
    "huber_c2p5": 2.5,
}
for _hname, _hval in HUBER_FIXED_C.items():
    _ESTIMATOR_REGISTRY[_hname] = _functools.partial(_make_huber_fixed, _hval)
del _hname, _hval


@register("huber_adaptive_c")
def _load_huber_adaptive_c():
    from mv_patp.baseline import huber_adaptive_c
    return huber_adaptive_c.fit


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

# sigma_structure prefixes that select a non-MAL noise family; anything else
# (canonically "I") is matrix MAL noise at alpha_MAL.
_NON_MAL_PREFIXES = ("student_t_", "ar1_gamma_rho", "ar1t_", "middleton_", "gaussian")


def is_mal_structure(sigma_structure: str) -> bool:
    """True when ``sigma_structure`` selects matrix MAL noise."""
    return not str(sigma_structure).startswith(_NON_MAL_PREFIXES)


def parse_ar1t(sigma_structure: str) -> tuple[float, float]:
    """Parse ``"ar1t_rho0p5_df4"`` into ``(rho, df)`` = ``(0.5, 4.0)``."""
    parts = sigma_structure.split("_")
    if len(parts) != 3 or not parts[1].startswith("rho") or not parts[2].startswith("df"):
        raise ValueError(
            f"Bad ar1t sigma_structure '{sigma_structure}'; expected 'ar1t_rho<r>_df<v>'."
        )
    rho = float(parts[1][3:].replace("p", "."))
    df = float(parts[2][2:].replace("p", "."))
    if not (-1.0 < rho < 1.0) or df <= 0:
        raise ValueError(f"ar1t needs |rho| < 1 and df > 0, got rho={rho}, df={df}.")
    return rho, df


def ar1_corr(k: int, rho: float) -> np.ndarray:
    """AR(1) row correlation Gamma_ij = rho^|i-j| (k x k Toeplitz)."""
    i_idx, j_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    return rho ** np.abs(i_idx - j_idx)

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
    # R1: when set, the MAL shape is alpha = (k-1)/2 + alpha_offset (a per-k
    # valid grid); alpha_MAL then holds that computed value.
    alpha_offset: float = float("nan")
    # DeepMIMO only: sorted (name, value) pairs forwarded to load_deepmimo
    # (seed_scheme, noise_per_snr, snr_scaling, alpha_policy); from the
    # config's ``deepmimo_options``. Empty = loader defaults (legacy streams).
    loader_opts: tuple = ()

    @property
    def is_mal(self) -> bool:
        """True when the noise is matrix MAL, i.e. alpha_MAL is not inert."""
        return self.data_source == "synthetic" and is_mal_structure(
            self.sigma_structure
        )

    @property
    def alpha_used(self) -> float:
        """MAL shape actually passed to the sampler (NaN when alpha is inert).

        There is no clipping in the synthetic path: an invalid value is
        reported, never altered. The DeepMIMO loader no longer clips either:
        an alpha <= (k-1)/2 for the real-equivalent k (known only inside the
        loader) raises InvalidAlphaError, so every replicate fails and the
        row says so. Only ``alpha_policy="clip_legacy"`` (reproduction of old
        runs) still clips there; then the sampled shape is unknown here and
        is reported as NaN.
        """
        if self.data_source == "deepmimo":
            if self.sigma_structure.startswith("student_t_"):
                return float("nan")
            if dict(self.loader_opts).get("alpha_policy") == "clip_legacy":
                return float("nan")
            return self.alpha_MAL
        return self.alpha_MAL if self.is_mal else float("nan")

    @property
    def alpha_valid(self) -> bool:
        """Bartlett condition alpha > (k-1)/2 for MG_k(alpha) (MAL cells only)."""
        if not self.is_mal:
            return True
        return bool(np.isfinite(self.alpha_MAL) and self.alpha_MAL > (self.k - 1) / 2.0)

    def data_id(self) -> str:
        """Identifier of the data configuration, shared by all estimators.

        Rows with equal ``data_id`` and equal ``rep`` see identical (Y, S, H).
        """
        if self.data_source == "deepmimo":
            opts = "".join(f"_{k}{v}" for k, v in self.loader_opts)
            return (
                f"deepmimo_{self.scenario}_snr{self.snr_db}_n{self.n}_p{self.p}"
                f"_alpha{self.alpha_MAL}_sigma{self.sigma_structure}{opts}"
            )
        alpha_tag = f"{self.alpha_MAL}" if self.is_mal else "NA"
        return (
            f"k{self.k}_n{self.n}_alpha{alpha_tag}_p{self.p}"
            f"_M{self.M_norm}_sigma{self.sigma_structure}"
        )

    def cell_id(self) -> str:
        return f"{self.data_id()}_est{self.estimator}"


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
        loader_opts = tuple(sorted(
            (str(k), v) for k, v in (config.get("deepmimo_options") or {}).items()
        ))
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
                loader_opts=loader_opts,
            ))
        return cells

    # The alpha axis is either absolute (``alpha_MAL``) or relative to the
    # per-k Bartlett bound (``alpha_MAL_offset``: alpha = (k-1)/2 + offset).
    # Both may be given; their cells are concatenated. The axis only
    # applies to MAL noise: every other noise kind is emitted once per
    # (k, n, p, M, sigma, estimator) with alpha_MAL = NaN, so cells are not
    # replicated over an inert axis.
    offsets = grid.get("alpha_MAL_offset")
    abs_alphas = grid.get("alpha_MAL")
    if abs_alphas is None and offsets is None:
        abs_alphas = [1.0]
    alpha_specs: list[tuple[str, float]] = (
        [("abs", float(a)) for a in (abs_alphas or [])]
        + [("off", float(o)) for o in (offsets or [])]
    )

    seen: set[str] = set()
    for k, n, p, M_norm, sigma_structure, estimator in itertools.product(
        grid.get("k", [2]),
        grid.get("n", [512]),
        grid.get("p", [2]),
        grid.get("M_norm", [0.0]),
        grid.get("sigma_structure", ["I"]),
        estimators,
    ):
        sigma_structure = str(sigma_structure)
        specs = (
            alpha_specs if is_mal_structure(sigma_structure)
            else [("inert", float("nan"))]
        )
        for kind, val in specs:
            if kind == "off":
                alpha_MAL, alpha_offset = (int(k) - 1) / 2.0 + val, val
            else:
                alpha_MAL, alpha_offset = val, float("nan")
            cell = CellSpec(
                k=int(k), n=int(n), alpha_MAL=float(alpha_MAL), p=int(p),
                M_norm=float(M_norm), sigma_structure=sigma_structure,
                replicates=replicates, estimator=str(estimator),
                alpha_offset=float(alpha_offset),
            )
            if cell.cell_id() in seen:  # e.g. an absolute alpha equal to an offset one
                continue
            seen.add(cell.cell_id())
            cells.append(cell)
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
    if cell.sigma_structure == "gaussian":
        # Exactly Gaussian iid noise (the df -> infinity limit of student_t_*).
        return rng.standard_normal((k, n))
    if cell.sigma_structure.startswith("ar1t_"):
        # Correlated heavy-tailed rows: N = L T, L L^T = Gamma (AR(1)),
        # T_ij iid Student-t_df. Each column n_j is a linear map of k iid
        # t variables, so Cov(n_j) = df/(df-2) * Gamma for df > 2 (row
        # correlation Gamma, marginal variance df/(df-2)); columns iid.
        # This is NOT a multivariate-t (no shared mixing scale per column).
        rho, df = parse_ar1t(cell.sigma_structure)
        L = np.linalg.cholesky(ar1_corr(k, rho))
        z = rng.standard_normal((k, n))
        chi2 = rng.chisquare(df=df, size=(k, n))
        return L @ (z / np.sqrt(chi2 / df))
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
    # Default: matrix MAL noise via the package sampler. No clipping (R1):
    # the alpha recorded as alpha_used is exactly the alpha sampled from.
    if not cell.alpha_valid:
        raise InvalidAlphaError(
            f"alpha_MAL={cell.alpha_MAL} violates alpha > (k-1)/2 = {(k - 1) / 2} at k={k}"
        )
    from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal
    arr = sample_matrix_mal(
        n_samples=1, k=k, n=n,
        alpha=cell.alpha_MAL,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    return arr[0]


class InvalidAlphaError(ValueError):
    """MAL shape outside the Bartlett range alpha > (k-1)/2."""


def draw_replicate(
    cell: CellSpec, seed: int, rep: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Y, S, H_truth)`` for replicate ``rep`` of a synthetic cell.

    Depends only on ``(cell.data_id(), seed, rep)`` — never on the
    estimator — so every estimator in a config sees identical data per
    replicate (paired design). The RNG is ``default_rng(seed + rep)``.
    """
    rng = np.random.default_rng(seed + rep)
    H_truth = rng.standard_normal((cell.k, cell.p))
    S = rng.standard_normal((cell.p, cell.n))
    N = _draw_noise(cell, rng)
    return H_truth @ S + N, S, H_truth


PER_REP_COLUMNS = (
    "cell_id", "data_id", "estimator", "k", "n", "p", "sigma_structure",
    "alpha_MAL", "alpha_offset", "alpha_used", "rep", "seed", "status",
    "sq_err", "wall_clock_s", "alpha_hat", "converged",
    "n_outer", "n_inner", "guard_triggered", "stop_reason", "c_hat",
)


def _fit_diagnostics(res: Any) -> dict[str, Any]:
    """Convergence diagnostics exposed by the estimator (R1 safeguards).

    oPMM (``EMPATPResult``): outer/inner iteration counts, divergence-guard
    flag and stop reason. Baselines (``BaselineResult``): ``n_iter`` as
    ``n_outer``; the other fields are NaN / False / "".
    """
    n_outer = getattr(res, "n_iter_outer", getattr(res, "n_iter", float("nan")))
    n_inner = getattr(res, "n_iter_irls_total", float("nan"))
    return {
        "converged": bool(getattr(res, "converged", True)),
        "n_outer": float(n_outer) if n_outer is not None else float("nan"),
        "n_inner": float(n_inner) if n_inner is not None else float("nan"),
        "guard_triggered": bool(getattr(res, "guard_triggered", False)),
        "stop_reason": str(getattr(res, "stop_reason", "") or ""),
    }


def _alpha_hat(res: Any) -> float:
    """Data-adapted alpha exposed by the estimator (oPMM: ``alpha_p``)."""
    val = getattr(res, "alpha_p", None)
    try:
        return float(val) if val is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _c_hat(res: Any) -> float:
    """Data-chosen Huber constant (``huber_adaptive_c``); NaN otherwise."""
    hist = getattr(res, "log_history", None)
    val = hist.get("c_hat") if isinstance(hist, dict) else None
    try:
        return float(val) if val is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def run_mc_cell(
    cell: CellSpec,
    seed: int,
    per_replicate: bool = False,
) -> dict[str, Any]:
    """Execute one MC cell: ``replicates`` fits of ``cell.estimator``.

    Returns aggregated MSE / bias / wall-clock statistics in the
    schema consumed by ``verify.py`` and ``plot.py``. Every row carries
    ``alpha_used`` (the MAL shape actually sampled from; NaN where alpha is
    inert). A MAL cell with alpha <= (k-1)/2 is not run: it returns
    ``status="invalid_alpha"``. With ``per_replicate=True`` the row also
    carries ``"_rep_rows"``: one dict per replicate (``PER_REP_COLUMNS``).
    """
    fit_factory = _ESTIMATOR_REGISTRY.get(cell.estimator)
    if fit_factory is None:
        raise ValueError(
            f"Unknown estimator '{cell.estimator}'. "
            f"Registered: {sorted(_ESTIMATOR_REGISTRY)}"
        )
    base_row: dict[str, Any] = {
        "cell_id": cell.cell_id(),
        "data_id": cell.data_id(),
        "estimator": cell.estimator,
        "k": cell.k, "p": cell.p, "n": cell.n,
        "alpha_MAL": cell.alpha_MAL,
        "alpha_offset": cell.alpha_offset,
        "alpha_used": cell.alpha_used,
        "sigma_structure": cell.sigma_structure,
        "M_norm": cell.M_norm,
        "data_source": cell.data_source,
        "scenario": cell.scenario,
        "snr_db": cell.snr_db,
        "replicates_requested": cell.replicates,
    }
    empty_stats = {
        "mse_mean": float("nan"),
        "mse_std": float("nan"),
        "n_replicates": 0,
        "n_failures": 0,
        "wall_clock_mean_s": float("nan"),
    }

    if not cell.alpha_valid:
        logger.warning("Cell %s: invalid alpha (<= (k-1)/2); not run.", cell.cell_id())
        out = {**base_row, "status": "invalid_alpha", **empty_stats}
        if per_replicate:
            out["_rep_rows"] = []
        return out

    try:
        fit = fit_factory()
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "Estimator '%s' not yet importable (%s); skipping cell.",
            cell.estimator, exc,
        )
        out = {**base_row, "status": "skipped_not_implemented", **empty_stats}
        if per_replicate:
            out["_rep_rows"] = []
        return out

    sq_errs: list[float] = []
    wall_clocks: list[float] = []
    rep_rows: list[dict[str, Any]] = []
    n_failures = 0
    n_converged = 0
    n_guard = 0

    for rep in range(cell.replicates):
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
                seed=seed + rep, **dict(cell.loader_opts),
            )
        else:
            Y, S, H_truth = draw_replicate(cell, seed, rep)
        t0 = time.perf_counter()
        rep_row: dict[str, Any] = {
            "cell_id": cell.cell_id(), "data_id": cell.data_id(),
            "estimator": cell.estimator, "k": cell.k, "n": cell.n, "p": cell.p,
            "sigma_structure": cell.sigma_structure,
            "alpha_MAL": cell.alpha_MAL, "alpha_offset": cell.alpha_offset,
            "alpha_used": cell.alpha_used, "rep": rep, "seed": seed + rep,
        }
        try:
            res = fit(Y, S)
            H_hat = getattr(res, "H", None)
            if H_hat is None:
                # Some baselines return BaselineResult; em_patp returns
                # EMPATPResult. Both expose .H so this is safe.
                raise AttributeError(".H field missing on result.")
            H_hat = np.asarray(H_hat)
            sq_err = float(np.linalg.norm(H_hat - H_truth, ord="fro") ** 2)
            elapsed = time.perf_counter() - t0
            sq_errs.append(sq_err)
            wall_clocks.append(elapsed)
            diag = _fit_diagnostics(res)
            n_converged += int(diag["converged"])
            n_guard += int(diag["guard_triggered"])
            rep_row.update(
                status="ok", sq_err=sq_err, wall_clock_s=elapsed,
                alpha_hat=_alpha_hat(res), c_hat=_c_hat(res), **diag,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cell %s replicate %d failed: %s",
                           cell.cell_id(), rep, exc)
            n_failures += 1
            rep_row.update(
                status=f"failed: {type(exc).__name__}", sq_err=float("nan"),
                wall_clock_s=time.perf_counter() - t0, alpha_hat=float("nan"),
                converged=False, n_outer=float("nan"), n_inner=float("nan"),
                guard_triggered=False, stop_reason="", c_hat=float("nan"),
            )
        if per_replicate:
            rep_rows.append(rep_row)

    if not sq_errs:
        out = {**base_row, "status": "all_replicates_failed", **empty_stats,
               "n_failures": n_failures}
    else:
        mse_arr = np.asarray(sq_errs)
        out = {
            **base_row,
            "status": "ok" if n_failures == 0 else "partial",
            "mse_mean": float(mse_arr.mean()),
            "mse_std": float(mse_arr.std(ddof=1)) if len(mse_arr) > 1 else 0.0,
            "n_replicates": len(sq_errs),
            "n_failures": n_failures,
            "wall_clock_mean_s": float(np.mean(wall_clocks)),
            # R1: share of successful fits that met the stopping rule, and
            # share in which the oPMM divergence guard acted.
            "converged_rate": n_converged / len(sq_errs),
            "guard_rate": n_guard / len(sq_errs),
        }
    if per_replicate:
        out["_rep_rows"] = rep_rows
    return out


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
                             "sequential). Data per replicate depends only on "
                             "(data_id, seed, rep), so results do not depend on "
                             "the number of workers.")
    parser.add_argument("--per-replicate", action="store_true",
                        help="Also write reps.csv: one row per (cell, replicate) "
                             "with sq_err, wall_clock_s, alpha_hat, for paired "
                             "bootstrap CIs (scripts/r1_paired_bootstrap.py).")
    parser.add_argument("--replicates", type=int, default=None,
                        help="Override the config's replicate count (timing "
                             "dry runs).")
    return parser.parse_args(argv)


def _run_cells(cells: list[CellSpec], seed: int, per_replicate: bool,
               workers: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cell_results: list[dict[str, Any]] = []
    pass_fail: dict[str, str] = {}

    def _sequential(todo: list[CellSpec]) -> None:
        for cell in todo:
            try:
                result = run_mc_cell(cell, seed=seed, per_replicate=per_replicate)
                cell_results.append(result)
                pass_fail[cell.cell_id()] = result["status"]
            except Exception as exc:  # log-and-continue per AGENTS Section 5.4
                logger.exception("Cell %s failed: %s", cell.cell_id(), exc)
                pass_fail[cell.cell_id()] = f"error: {type(exc).__name__}"

    if workers <= 1:
        _sequential(cells)
        return cell_results, pass_fail

    logger.info("Parallel dispatch: %d workers (spawn).", workers)
    import os
    from multiprocessing import get_context
    # Limit BLAS/OpenMP threads per worker to 1 so process-level
    # parallelism doesn't compete with numpy's internal threading.
    blas_env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    os.environ.update(blas_env)
    ctx = get_context("spawn")
    try:
        with ctx.Pool(processes=workers) as pool:
            # chunksize=1: cells differ in cost by orders of magnitude (k=8
            # vs k=2), so one cell per task keeps the workers balanced.
            for result in pool.imap(
                _worker, [(c, seed, per_replicate) for c in cells], chunksize=1
            ):
                cell_results.append(result)
                pass_fail[result["cell_id"]] = result["status"]
    except Exception as exc:
        logger.exception("Pool dispatch failed (%s); falling back to sequential.", exc)
        done = set(pass_fail)
        _sequential([c for c in cells if c.cell_id() not in done])
    return cell_results, pass_fail


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.replicates is not None:
        config["replicates"] = int(args.replicates)
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
    else:
        cell_results, pass_fail = _run_cells(
            cells, args.seed, args.per_replicate, args.workers
        )

    wall_clock = time.perf_counter() - t0

    # Write CSV (always) and manifest (always).
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cells.csv"
    rep_rows: list[dict[str, Any]] = []
    for r in cell_results:
        rep_rows.extend(r.pop("_rep_rows", []) or [])
    if cell_results:
        import pandas as pd  # local import to keep dry-run lightweight
        pd.DataFrame(cell_results).to_csv(csv_path, index=False)
    else:
        csv_path.write_text("cell_id,status,mse_mean,mse_std,n_replicates\n")
    if args.per_replicate and not args.dry_run:
        write_per_replicate(rep_rows, output_dir / "reps.csv")
        logger.info("Per-replicate CSV: %s (%d rows)", output_dir / "reps.csv",
                    len(rep_rows))

    input_paths: list[Path] = []
    if config.get("data_source") == "deepmimo":
        from mv_patp.data_gen.deepmimo_loader import channel_npz_path
        input_paths = [
            p for p in (channel_npz_path(s) for s in
                        sorted({c.scenario for c in cells}))
            if p.is_file()
        ]
    manifest_path = write_manifest(
        output_dir=output_dir,
        config_path=args.config,
        seed=args.seed,
        wall_clock_seconds=wall_clock,
        pass_fail_summary=pass_fail,
        input_data_paths=input_paths,
    )
    logger.info("Manifest: %s", manifest_path)
    logger.info("Cells CSV: %s", csv_path)

    # Exit code reflects whether any cell errored.
    return 0 if not any(v.startswith("error:") for v in pass_fail.values()) else 1


def write_per_replicate(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write the long per-replicate table (columns ``PER_REP_COLUMNS``)."""
    import pandas as pd
    df = pd.DataFrame(rows, columns=list(PER_REP_COLUMNS))
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    sys.exit(main())
