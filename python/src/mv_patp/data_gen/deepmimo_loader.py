"""DeepMIMO ray-tracing channel loader for §V-C real-data evaluation.

Wraps the ``deepmimo`` v4 Python package (Alkhateeb 2019, github.com/
DeepMIMO/DeepMIMOv4). Returns ``(Y, S, H_truth)`` triples where ``H_truth``
is a real-equivalent representation of the complex MIMO channel matrix
extracted from DeepMIMO's ray-traced channels, ``S`` is a unit-power
training matrix, and ``Y = H_truth @ S + N`` with ``N`` drawn from the
requested noise family (Student-t or matrix MAL) at the requested SNR.

DeepMIMO v3 scenarios ``O1``/``I1`` referenced in Spec.md §6.1 are
deprecated; this module uses v4 names per Decision Log PIVOT 17:

- ``asu_campus_3p5`` — outdoor campus, 131K RX, 6 reflections, rich multipath.
- ``city_0_newyork_3p5`` — dense urban canyon, 31K RX, 3 reflections,
  MacroDataset with 3 TX positions (heavy NLOS, high correlation).

Complex → real conversion: ``H_real = [Re(H_c); Im(H_c)]`` (vertical stack)
giving shape ``(2 * k_ant, p)`` with phase information preserved.

Data location
-------------
``data_dir`` resolves, in order, to the explicit argument, ``$DEEPMIMO_DATA_DIR``,
``$RESEARCH_DATA_ROOT/store/deepmimo-v4/data`` (the dataset lake) when it
exists, and finally the legacy relative ``data/deepmimo``. Inside it, a
pre-extracted ``<scenario>_ch2d.npz`` (written by
``scripts/data_download/export_deepmimo_channels.py``) is used when present;
otherwise the ``deepmimo`` package builds the channels. The channel table is
cached per process, so the scenario is read once, not once per replicate.

Noise-injection protocol (R1, Rv1-C14)
--------------------------------------
Three keyword options; their defaults reproduce the original runs exactly.

- ``seed_scheme="legacy"``: position RNG ``seed+2``, pilot ``seed``, noise
  ``seed+1``. With ``seed = 42 + rep`` the pilot of replicate ``b+1`` equals
  the first ``p`` rows of the Gaussian numerator of the noise of replicate
  ``b`` (overlapping streams). ``"spawn"`` draws the three streams from
  ``np.random.SeedSequence(seed).spawn(3)``, which removes the overlap.
- ``noise_per_snr=False``: the noise draw does not depend on the SNR, so
  across SNR only its scale changes. ``True`` derives the noise stream from
  (seed, SNR) so every SNR gets an independent draw; channel and pilot stay
  common across SNR.
- ``snr_scaling="realised"``: one scalar makes the *realised* noise power
  equal ``mean((HS)^2) / 10^(SNR/10)`` (self-normalised noise).
  ``"nominal"`` scales so that the *nominal* per-entry noise variance equals
  the nominal per-entry signal power ``||H||_F^2 E[s^2] / k = 1/k`` divided by
  ``10^(SNR/10)``; for Student-t this uses Var = df/(df-2) (df > 2 only).
- ``alpha_policy="invalid"``: a matrix-MAL shape ``alpha <= (k-1)/2`` raises
  :class:`InvalidAlphaError` (no silent change). ``"clip_legacy"`` keeps the
  original behaviour ``alpha = max(alpha, (k-1)/2 + 0.05)`` with a warning,
  only for reproducing old runs.

Satisfies Requirements R10.1, R10.2.
Task: §16.1.

References
----------
Alkhateeb, A. (2019). DeepMIMO: A generic deep learning dataset for
millimeter wave and massive MIMO applications. ITA.
DeepMIMO v4 docs: https://www.deepmimo.net/.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np

from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal

logger = logging.getLogger(__name__)


_SUPPORTED_SCENARIOS: tuple[str, ...] = (
    "asu_campus_3p5",
    "city_0_newyork_3p5",
)

_LAKE_SUBDIR = Path("store") / "deepmimo-v4" / "data"
_LEGACY_DATA_DIR = Path("data") / "deepmimo"

SEED_SCHEMES: tuple[str, ...] = ("legacy", "spawn")
SNR_SCALINGS: tuple[str, ...] = ("realised", "nominal")
ALPHA_POLICIES: tuple[str, ...] = ("invalid", "clip_legacy")


class InvalidAlphaError(ValueError):
    """Matrix-MAL shape outside the Bartlett range alpha > (k-1)/2."""


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Directory holding the DeepMIMO scenario folders (see module docstring)."""
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("DEEPMIMO_DATA_DIR")
    if env:
        return Path(env)
    root = os.environ.get("RESEARCH_DATA_ROOT")
    if root and (Path(root) / _LAKE_SUBDIR).is_dir():
        return Path(root) / _LAKE_SUBDIR
    return _LEGACY_DATA_DIR


def channel_npz_path(scenario: str, data_dir: str | Path | None = None) -> Path:
    """Path of the pre-extracted channel table for ``scenario``."""
    return resolve_data_dir(data_dir) / f"{scenario}_ch2d.npz"


def _ensure_scenarios_folder(data_dir: str | Path) -> None:
    """Configure the ``deepmimo`` package to resolve scenarios from ``data_dir``."""
    import deepmimo  # local import: deepmimo is an optional runtime dep

    deepmimo.config.set("scenarios_folder", str(data_dir))


@functools.lru_cache(maxsize=4)
def _channel_table(scenario: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """``(ch_2d, valid)``: per-RX complex channels and top-quartile RX indices.

    ``ch_2d`` has shape ``(num_rx, k_ant)``: first TX for multi-TX
    scenarios, TX antenna 0 and subcarrier 0. ``valid`` lists RX positions
    whose channel norm is in the top quartile: the ray tracer returns ~36 %
    zero-norm positions and heavy-tailed path losses (median ~1e-7), so each
    sampled channel is required to have at least one strong path, as a real
    system would gate weak users out of the training set.
    """
    npz = Path(data_dir) / f"{scenario}_ch2d.npz"
    if npz.is_file():
        with np.load(npz) as f:
            ch_2d = np.asarray(f["ch2d"])
        logger.info("DeepMIMO %s: channel table from %s", scenario, npz)
    else:
        import deepmimo

        _ensure_scenarios_folder(data_dir)
        ch = deepmimo.load(scenario).channel
        # MacroDataset (multi-TX, e.g. city_0_newyork_3p5) returns one array
        # per TX; the first TX is used.
        if isinstance(ch, (list, tuple)):
            ch = ch[0]
        ch = np.asarray(ch)
        if ch.ndim != 4:
            raise ValueError(
                f"Unexpected DeepMIMO channel ndim={ch.ndim}; expected 4-D."
            )
        _, n_tx_ant, _, n_subc = ch.shape
        if n_tx_ant != 1 or n_subc != 1:
            logger.warning(
                "DeepMIMO scenario %s has n_tx_ant=%d, n_subc=%d; using [0, 0].",
                scenario, n_tx_ant, n_subc,
            )
        ch_2d = ch[:, 0, :, 0]
    norms = np.linalg.norm(ch_2d, axis=1)
    threshold = float(np.percentile(norms, 75.0))
    valid = np.nonzero(norms >= threshold)[0]
    return ch_2d, valid


def _extract_complex_channel(
    scenario: str,
    n_train: int,
    p: int,
    rng: np.random.Generator,
    data_dir: str | Path | None = None,
) -> np.ndarray:
    """Sample ``n_train`` complex channel matrices of shape ``(k_ant, p)``.

    A single RX position gives a length-``k_ant`` complex vector; ``p``
    distinct top-quartile RX positions are stacked as columns.
    """
    ch_2d, valid = _channel_table(scenario, str(resolve_data_dir(data_dir)))
    if valid.size < n_train * p:
        raise ValueError(
            f"Scenario {scenario} has only {valid.size} top-quartile RX "
            f"positions; need {n_train * p} for n_train={n_train} × p={p}."
        )
    selected = rng.choice(valid, size=n_train * p, replace=False)
    chosen = ch_2d[selected]                   # (n_train * p, k_ant)
    k_ant = chosen.shape[1]
    return chosen.reshape(n_train, p, k_ant).transpose(0, 2, 1)


def _real_equiv(H_complex: np.ndarray) -> np.ndarray:
    """Vertical real-equivalent stack: (n, k, p) complex -> (n, 2k, p) real."""
    return np.concatenate(
        [H_complex.real, H_complex.imag], axis=1
    ).astype(np.float64)


def _scale_noise_to_snr(
    HS: np.ndarray, N_unit: np.ndarray, snr_db: float
) -> np.ndarray:
    """Rescale noise so that the *realised* ``mean(HS^2) / mean(N^2)`` is ``snr_db``."""
    sig_power = float(np.mean(HS * HS))
    noise_power = float(np.mean(N_unit * N_unit))
    if noise_power <= 0.0:
        raise ValueError("Noise sample has zero power; cannot rescale.")
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return scale * N_unit


def _snr_key(snr_db: float) -> int:
    """Non-negative integer key of an SNR value (resolution 1e-3 dB)."""
    key = int(round(float(snr_db) * 1000.0)) + 1_000_000
    if key < 0:
        raise ValueError(f"snr_db={snr_db} out of range for the noise stream key.")
    return key


def replicate_streams(
    seed: int, snr_db: float, *, seed_scheme: str = "legacy",
    noise_per_snr: bool = False,
) -> tuple[np.random.Generator, np.random.Generator, np.random.SeedSequence | int]:
    """RNGs for channel positions and pilot, and the noise seed.

    Returns ``(rng_pos, rng_S, noise_seed)``; ``noise_seed`` is an int
    (legacy, SNR-independent) or a ``SeedSequence``. Shared by the DeepMIMO
    and WAIR-D loaders.
    """
    if seed_scheme not in SEED_SCHEMES:
        raise ValueError(f"seed_scheme={seed_scheme!r}; expected one of {SEED_SCHEMES}")
    if seed_scheme == "legacy":
        rng_pos = np.random.default_rng(seed + 2)
        rng_S = np.random.default_rng(seed)
        noise_seed: np.random.SeedSequence | int = (
            np.random.SeedSequence(seed + 1, spawn_key=(_snr_key(snr_db),))
            if noise_per_snr else seed + 1
        )
    else:
        ss_pos, ss_S, ss_noise = np.random.SeedSequence(seed).spawn(3)
        rng_pos = np.random.default_rng(ss_pos)
        rng_S = np.random.default_rng(ss_S)
        noise_seed = (
            np.random.SeedSequence(seed, spawn_key=(2, _snr_key(snr_db)))
            if noise_per_snr else ss_noise
        )
    return rng_pos, rng_S, noise_seed


def draw_unit_noise(
    noise_kind: str, k: int, n_train: int, noise_seed: np.random.SeedSequence | int,
    *, alpha_mal: float = 1.0, alpha_policy: str = "invalid",
) -> tuple[np.ndarray, float]:
    """Unscaled noise ``(k, n_train)`` and its nominal per-entry variance.

    The variance is ``df/(df-2)`` for Student-t (``inf`` for df <= 2) and
    ``nan`` for matrix MAL (nominal scaling is not defined for it here).
    """
    if alpha_policy not in ALPHA_POLICIES:
        raise ValueError(f"alpha_policy={alpha_policy!r}; expected one of {ALPHA_POLICIES}")
    if noise_kind.startswith("student_t_"):
        df = float(noise_kind.split("_")[-1])
        rng_n = np.random.default_rng(noise_seed)
        z = rng_n.standard_normal((k, n_train))
        chi2 = rng_n.chisquare(df=df, size=(k, n_train))
        N_unit = (z / np.sqrt(chi2 / df)).astype(np.float64)
        var = df / (df - 2.0) if df > 2.0 else float("inf")
        return N_unit, var
    if noise_kind == "matrix_mal":
        bound = (k - 1) / 2.0
        alpha_used = alpha_mal
        if not alpha_mal > bound:
            if alpha_policy == "invalid":
                raise InvalidAlphaError(
                    f"alpha_mal={alpha_mal} <= (k-1)/2={bound} for k={k}; "
                    "MG_k(alpha) is undefined. Not run (alpha_policy='invalid')."
                )
            alpha_used = bound + 0.05
            logger.warning(
                "alpha_mal=%.3f <= (k-1)/2=%.3f for k=%d; clipped to %.3f "
                "(alpha_policy='clip_legacy').", alpha_mal, bound, k, alpha_used,
            )
        mal_seed = (
            int(noise_seed) if isinstance(noise_seed, (int, np.integer))
            else int(noise_seed.generate_state(1)[0])
        )
        N_unit = sample_matrix_mal(
            n_samples=1, k=k, n=n_train, alpha=alpha_used, seed=mal_seed,
        )[0]
        return N_unit, float("nan")
    raise ValueError(
        f"Unsupported noise_kind={noise_kind!r}; expected 'matrix_mal' "
        f"or 'student_t_<df>' (e.g. 'student_t_4')."
    )


def scale_noise(
    HS: np.ndarray, H_truth: np.ndarray, N_unit: np.ndarray, unit_var: float,
    snr_db: float, *, snr_scaling: str = "realised",
) -> np.ndarray:
    """Scale ``N_unit`` to ``snr_db`` (realised or nominal powers)."""
    if snr_scaling not in SNR_SCALINGS:
        raise ValueError(f"snr_scaling={snr_scaling!r}; expected one of {SNR_SCALINGS}")
    if snr_scaling == "realised":
        return _scale_noise_to_snr(HS, N_unit, snr_db)
    if not (np.isfinite(unit_var) and unit_var > 0.0):
        raise ValueError(
            "snr_scaling='nominal' needs a finite nominal noise variance "
            "(Student-t with df > 2)."
        )
    k = H_truth.shape[0]
    # Pilot entries are iid N(0,1): E[mean((HS)^2)] = ||H||_F^2 / k.
    sig_power_nominal = float(np.linalg.norm(H_truth) ** 2) / k
    target_var = sig_power_nominal / (10.0 ** (snr_db / 10.0))
    return float(np.sqrt(target_var / unit_var)) * N_unit


def load_deepmimo(
    scenario: str,
    snr_db: float,
    n_train: int,
    *,
    p: int = 2,
    alpha_mal: float = 1.0,
    noise_kind: str = "matrix_mal",
    seed: int = 42,
    data_dir: str | Path | None = None,
    seed_scheme: str = "legacy",
    noise_per_snr: bool = False,
    snr_scaling: str = "realised",
    alpha_policy: str = "invalid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a ``(Y, S, H_truth)`` triple from a DeepMIMO ray-traced channel.

    Pulls a real-equivalent channel matrix ``H_truth`` from the requested
    scenario, draws a unit-power Gaussian training matrix ``S``, samples
    noise ``N`` at the target SNR, and returns ``Y = H_truth @ S + N``.

    Parameters
    ----------
    scenario : str
        DeepMIMO v4 scenario name; one of ``_SUPPORTED_SCENARIOS``.
    snr_db : float
        Target signal-to-noise ratio in dB (``10 log10(P_signal / P_noise)``).
    n_train : int
        Length of the training sequence (number of columns of ``S`` and ``Y``).
    p : int, default 2
        Number of TX antennas; columns of ``H_truth``: ``p`` distinct
        DeepMIMO RX positions.
    alpha_mal : float, default 1.0
        Shape parameter of the matrix-MAL noise (``MG_k(alpha)`` mixing).
        Must exceed ``(k-1)/2`` for the real-equivalent ``k`` (= 16 here)
        unless ``alpha_policy="clip_legacy"``.
    noise_kind : str, default ``"matrix_mal"``
        ``"matrix_mal"`` or ``"student_t_<df>"``.
    seed : int, default 42
        Replicate seed (``run.py`` passes ``42 + rep``).
    data_dir : str, Path or None
        Scenario directory; ``None`` resolves via :func:`resolve_data_dir`.
    seed_scheme, noise_per_snr, snr_scaling, alpha_policy
        Noise-injection protocol; see the module docstring. The defaults
        reproduce the original (pre-R1) runs, except that an invalid
        matrix-MAL alpha now raises instead of being clipped.

    Returns
    -------
    (Y, S, H_truth) : tuple of np.ndarray
        ``Y`` has shape ``(2 * k_ant, n_train)``; ``S`` shape ``(p, n_train)``;
        ``H_truth`` shape ``(2 * k_ant, p)``. All real-valued.

    Raises
    ------
    ValueError
        If the scenario name or an option is unsupported, or the scenario
        has insufficient RX positions.
    InvalidAlphaError
        Matrix-MAL alpha <= (k-1)/2 with ``alpha_policy="invalid"``.
    ImportError
        If no extracted channel table exists and ``deepmimo`` is not installed.
    """
    if scenario not in _SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario {scenario!r}; expected one of "
            f"{_SUPPORTED_SCENARIOS}."
        )

    rng_pos, rng_S, noise_seed = replicate_streams(
        seed, snr_db, seed_scheme=seed_scheme, noise_per_snr=noise_per_snr,
    )
    H_complex = _extract_complex_channel(
        scenario, n_train=1, p=p, rng=rng_pos, data_dir=data_dir,
    )[0]                                        # (k_ant, p)

    H_truth = _real_equiv(H_complex[None])[0]   # (2*k_ant, p)
    # Unit Frobenius norm per call so SNR is well-defined and MSE comparisons
    # are scale-invariant (raw DeepMIMO path loss is ~1e-7 here).
    H_norm = float(np.linalg.norm(H_truth))
    if H_norm <= 0.0:
        raise ValueError(
            f"Selected channel from {scenario} has zero norm; check filter."
        )
    H_truth = H_truth / H_norm
    k = H_truth.shape[0]

    S = rng_S.standard_normal((p, n_train)).astype(np.float64)
    HS = H_truth @ S                            # (k, n_train)

    N_unit, unit_var = draw_unit_noise(
        noise_kind, k, n_train, noise_seed,
        alpha_mal=alpha_mal, alpha_policy=alpha_policy,
    )
    N = scale_noise(HS, H_truth, N_unit, unit_var, snr_db, snr_scaling=snr_scaling)
    return HS + N, S, H_truth
