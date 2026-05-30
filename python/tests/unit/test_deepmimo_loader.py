"""Unit tests for data_gen/deepmimo_loader.py::load_deepmimo.

Satisfies Requirement R10.1 (DeepMIMO loader produces a well-formed
``(Y, S, H_truth)`` triple with the requested SNR and shape).

These tests assume the DeepMIMO scenarios ``asu_campus_3p5`` and
``city_0_newyork_3p5`` are pre-downloaded under ``data/deepmimo/`` (per
``scripts/data_download/download_deepmimo.py`` or ``deepmimo.download``).
The tests skip if the optional ``deepmimo`` dependency is not installed
or the scenario folder is missing, so they remain green on CI without
the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

deepmimo = pytest.importorskip("deepmimo")

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "deepmimo"
_ASU = _DATA_DIR / "asu_campus_3p5"
_NYC = _DATA_DIR / "city_0_newyork_3p5"


pytestmark = pytest.mark.skipif(
    not _ASU.is_dir() or not _NYC.is_dir(),
    reason="DeepMIMO scenarios not downloaded; see PIVOT 17 in Spec.md §12.",
)


def _import_loader():
    from mv_patp.data_gen.deepmimo_loader import load_deepmimo
    return load_deepmimo


def test_load_deepmimo_returns_correct_shapes_when_called() -> None:
    """``load_deepmimo`` returns ``(Y, S, H_truth)`` of expected shape.

    Requirement R10.1.
    """
    load_deepmimo = _import_loader()
    Y, S, H = load_deepmimo(
        "asu_campus_3p5", snr_db=10.0, n_train=64, p=2, seed=42,
    )
    assert Y.ndim == 2 and S.ndim == 2 and H.ndim == 2
    k, n = Y.shape
    assert S.shape == (2, n)
    assert H.shape == (k, 2)


def test_load_deepmimo_matches_target_snr_when_called() -> None:
    """Empirical SNR matches the requested ``snr_db`` within 0.5 dB.

    Requirement R10.1.
    """
    load_deepmimo = _import_loader()
    Y, S, H = load_deepmimo(
        "asu_campus_3p5", snr_db=10.0, n_train=256, p=2, seed=42,
    )
    sig_pow = float(np.mean((H @ S) ** 2))
    noise_pow = float(np.mean((Y - H @ S) ** 2))
    snr_emp = 10.0 * np.log10(sig_pow / noise_pow)
    assert abs(snr_emp - 10.0) < 0.5, f"empirical SNR={snr_emp:.2f} dB"


def test_load_deepmimo_normalises_h_to_unit_norm_when_called() -> None:
    """``H_truth`` has unit Frobenius norm to make MSE scale-invariant.

    Requirement R10.1 (paper-supplied design choice; see loader docstring).
    """
    load_deepmimo = _import_loader()
    _, _, H = load_deepmimo(
        "asu_campus_3p5", snr_db=5.0, n_train=64, p=2, seed=42,
    )
    assert abs(float(np.linalg.norm(H)) - 1.0) < 1e-9


def test_load_deepmimo_handles_macrodataset_when_scenario_has_multi_tx() -> None:
    """``city_0_newyork_3p5`` is a 3-TX MacroDataset; loader returns one TX.

    Requirement R10.1, R10.2 (loader robust to multi-TX scenarios).
    """
    load_deepmimo = _import_loader()
    Y, S, H = load_deepmimo(
        "city_0_newyork_3p5", snr_db=10.0, n_train=64, p=2, seed=7,
    )
    assert Y.shape[1] == 64 and S.shape == (2, 64)
    assert H.shape[1] == 2 and H.shape[0] >= 2


def test_load_deepmimo_rejects_unknown_scenario_when_misnamed() -> None:
    """Unsupported scenario names raise ``ValueError``."""
    load_deepmimo = _import_loader()
    with pytest.raises(ValueError, match="Unsupported scenario"):
        load_deepmimo(
            "I1_2p5", snr_db=10.0, n_train=64, p=2, seed=0,
        )
