"""DeepMIMO ray-tracing channel loader for §V-C real-data evaluation.

Wraps the ``deepmimo`` v4 Python package (Alkhateeb 2019, github.com/
DeepMIMO/DeepMIMOv4). Returns ``(Y, S, H_truth)`` triples where ``H_truth``
is a real-equivalent representation of the complex MIMO channel matrix
extracted from DeepMIMO's ray-traced channels, ``S`` is a unit-power
training matrix, and ``Y = H_truth @ S + N`` with ``N`` sampled from the
matrix-variate Asymmetric Laplace distribution at the requested SNR.

DeepMIMO v3 scenarios ``O1``/``I1`` referenced in Spec.md §6.1 are
deprecated; this module uses v4 names per Decision Log PIVOT 17:

- ``asu_campus_3p5`` — outdoor campus, 131K RX, 6 reflections, rich multipath.
- ``city_0_newyork_3p5`` — dense urban canyon, 31K RX, 3 reflections,
  MacroDataset with 3 TX positions (heavy NLOS, high correlation).

Complex → real conversion: ``H_real = [Re(H_c); Im(H_c)]`` (vertical stack)
giving shape ``(2 * k_ant, p)`` with phase information preserved.

Satisfies Requirements R10.1, R10.2.
Task: §16.1.

References
----------
Alkhateeb, A. (2019). DeepMIMO: A generic deep learning dataset for
millimeter wave and massive MIMO applications. ITA.
DeepMIMO v4 docs: https://www.deepmimo.net/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal

logger = logging.getLogger(__name__)


_SUPPORTED_SCENARIOS: tuple[str, ...] = (
    "asu_campus_3p5",
    "city_0_newyork_3p5",
)


def _ensure_scenarios_folder(data_dir: str | Path) -> None:
    """Configure the ``deepmimo`` package to resolve scenarios from ``data_dir``."""
    import deepmimo  # local import: deepmimo is an optional runtime dep

    deepmimo.config.set("scenarios_folder", str(data_dir))


def _extract_complex_channel(
    scenario: str,
    n_train: int,
    p: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n_train`` complex channel matrices of shape ``(k_ant, p)``.

    DeepMIMO returns channels of shape ``(num_rx, n_tx_ant, n_rx_ant,
    n_subcarriers)``. For the chosen sub-6 scenarios ``n_tx_ant=1`` and
    ``n_subcarriers=1``, so a single RX position gives a length-``k_ant``
    complex vector. To build a ``(k_ant, p)`` channel matrix we stack ``p``
    distinct RX positions as columns.
    """
    import deepmimo

    ds = deepmimo.load(scenario)
    ch = ds.channel
    # MacroDataset (multi-TX scenarios like city_0_newyork_3p5) returns a list
    # of DeepMIMOArrays, one per TX. Single-TX scenarios return a 4-D ndarray
    # directly. In the multi-TX case we use the first TX for the §V-C demo;
    # full-MC orchestrator can iterate TXs explicitly via the seed offset.
    if isinstance(ch, (list, tuple)):
        ch = ch[0]
    ch = np.asarray(ch)
    if ch.ndim != 4:
        raise ValueError(
            f"Unexpected DeepMIMO channel ndim={ch.ndim}; expected 4-D."
        )
    num_rx, n_tx_ant, k_ant, n_subc = ch.shape
    if n_tx_ant != 1 or n_subc != 1:
        logger.warning(
            "DeepMIMO scenario %s has n_tx_ant=%d, n_subc=%d; using indices [0, 0].",
            scenario, n_tx_ant, n_subc,
        )
    # Drop singletons -> (num_rx, k_ant) complex.
    ch_2d = ch[:, 0, :, 0]

    # Filter low-power RX positions: ray tracer returns ~36% zero-norm and
    # a heavy-tailed distribution of physical path losses (median ~1e-7).
    # Restrict to the top quartile by Frobenius norm so each sampled channel
    # has at least one strong propagation path; this mirrors how a real
    # MIMO system would gate weak users out of the training set.
    norms = np.linalg.norm(ch_2d, axis=1)
    threshold = float(np.percentile(norms, 75.0))
    valid = np.nonzero(norms >= threshold)[0]
    if valid.size < n_train * p:
        raise ValueError(
            f"Scenario {scenario} has only {valid.size} top-quartile RX "
            f"positions; need {n_train * p} for n_train={n_train} × p={p}."
        )
    # Random RX selection without replacement.
    selected = rng.choice(valid, size=n_train * p, replace=False)
    chosen = ch_2d[selected]                   # (n_train * p, k_ant)
    H_complex = chosen.reshape(n_train, p, k_ant).transpose(0, 2, 1)
    return H_complex                            # (n_train, k_ant, p)


def _real_equiv(H_complex: np.ndarray) -> np.ndarray:
    """Vertical real-equivalent stack: (n, k, p) complex -> (n, 2k, p) real."""
    return np.concatenate(
        [H_complex.real, H_complex.imag], axis=1
    ).astype(np.float64)


def _scale_noise_to_snr(
    HS: np.ndarray, N_unit: np.ndarray, snr_db: float
) -> np.ndarray:
    """Rescale noise sample so that ``E[|HS|^2] / E[|N|^2]`` matches ``snr_db``."""
    sig_power = float(np.mean(HS * HS))
    noise_power = float(np.mean(N_unit * N_unit))
    if noise_power <= 0.0:
        raise ValueError("Noise sample has zero power; cannot rescale.")
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return scale * N_unit


def load_deepmimo(
    scenario: str,
    snr_db: float,
    n_train: int,
    *,
    p: int = 2,
    alpha_mal: float = 1.0,
    noise_kind: str = "matrix_mal",
    seed: int = 42,
    data_dir: str | Path = "data/deepmimo",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a ``(Y, S, H_truth)`` triple from a DeepMIMO ray-traced channel.

    Pulls a real-equivalent channel matrix ``H_truth`` from the requested
    scenario, draws a unit-power Gaussian training matrix ``S``, samples
    matrix-MAL noise ``N`` at the target SNR, and returns ``Y = H_truth @ S + N``.

    Parameters
    ----------
    scenario : str
        DeepMIMO v4 scenario name; one of ``_SUPPORTED_SCENARIOS``.
    snr_db : float
        Target signal-to-noise ratio in dB (``10 log10(P_signal / P_noise)``).
    n_train : int
        Length of the training sequence (number of columns of ``S`` and ``Y``).
    p : int, default 2
        Number of TX antennas; columns of ``H_truth``. For each row index in
        ``H_truth`` we draw ``p`` distinct DeepMIMO RX positions.
    alpha_mal : float, default 1.0
        Shape parameter of the matrix-MAL noise (``MG_k(alpha)`` mixing).
    seed : int, default 42
        Top-level RNG seed (per AGENTS.md §6.2). The training matrix uses
        ``seed``; noise sampling uses ``seed + 1``; RX position sampling
        uses ``seed + 2``.
    data_dir : str or Path, default ``data/deepmimo``
        Directory containing the pre-downloaded DeepMIMO scenario folders.

    Returns
    -------
    (Y, S, H_truth) : tuple of np.ndarray
        ``Y`` has shape ``(2 * k_ant, n_train)``; ``S`` shape ``(p, n_train)``;
        ``H_truth`` shape ``(2 * k_ant, p)``. All real-valued.

    Raises
    ------
    ValueError
        If the scenario name is unsupported or has insufficient RX positions.
    ImportError
        If the optional ``deepmimo`` package is not installed.
    """
    if scenario not in _SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario {scenario!r}; expected one of "
            f"{_SUPPORTED_SCENARIOS}."
        )

    _ensure_scenarios_folder(data_dir)

    rng_pos = np.random.default_rng(seed + 2)
    H_complex = _extract_complex_channel(scenario, n_train=1, p=p, rng=rng_pos)
    # H_complex shape: (1, k_ant, p) — one channel matrix per call.
    H_complex = H_complex[0]                    # (k_ant, p)

    H_truth = _real_equiv(H_complex[None])[0]   # (2*k_ant, p)
    # Normalise to unit Frobenius norm per call so SNR is well-defined and
    # downstream MSE comparisons are scale-invariant. The physical raw path
    # loss in DeepMIMO is ~1e-7 on the chosen scenarios — un-normalised it
    # would saturate float64 noise scaling.
    H_norm = float(np.linalg.norm(H_truth))
    if H_norm <= 0.0:
        raise ValueError(
            f"Selected channel from {scenario} has zero norm; check filter."
        )
    H_truth = H_truth / H_norm
    k = H_truth.shape[0]

    rng_S = np.random.default_rng(seed)
    S = rng_S.standard_normal((p, n_train)).astype(np.float64)
    HS = H_truth @ S                            # (k, n_train)

    # Sample unit-power noise; rescale to target SNR. ``noise_kind`` selects
    # the family: ``matrix_mal`` (Bartlett) or ``student_t_<df>`` (per-element
    # Student-t, matching the synthetic MC convention in run.py::_draw_noise).
    # For k=16 (Re/Im stack of 8 antennas) matrix-MAL clips alpha well above
    # the heavy-tailed regime, so PIVOT 17 §V-C uses Student-t df=4 by default.
    if noise_kind.startswith("student_t_"):
        df = float(noise_kind.split("_")[-1])
        rng_n = np.random.default_rng(seed + 1)
        z = rng_n.standard_normal((k, n_train))
        chi2 = rng_n.chisquare(df=df, size=(k, n_train))
        N_unit = (z / np.sqrt(chi2 / df)).astype(np.float64)
    elif noise_kind == "matrix_mal":
        # Bartlett decomposition requires alpha > (k-1)/2; clip with 0.05 margin.
        alpha_used = max(alpha_mal, (k - 1) / 2.0 + 0.05)
        if alpha_used != alpha_mal:
            logger.info(
                "alpha_mal=%.3f below (k-1)/2=%.3f for k=%d; clipped to %.3f.",
                alpha_mal, (k - 1) / 2.0, k, alpha_used,
            )
        N_unit = sample_matrix_mal(
            n_samples=1, k=k, n=n_train, alpha=alpha_used, seed=seed + 1,
        )[0]
    else:
        raise ValueError(
            f"Unsupported noise_kind={noise_kind!r}; expected 'matrix_mal' "
            f"or 'student_t_<df>' (e.g. 'student_t_4')."
        )
    N = _scale_noise_to_snr(HS, N_unit, snr_db)

    Y = HS + N
    return Y, S, H_truth
