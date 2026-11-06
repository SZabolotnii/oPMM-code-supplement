"""R1 noise-injection protocol of the ray-traced loaders (Rv1-C14).

Uses a synthetic ``<scenario>_ch2d.npz`` channel table in ``tmp_path``, so
neither the ``deepmimo`` package nor the downloaded scenarios are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.data_gen import deepmimo_loader as dl

SCEN = "asu_campus_3p5"


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    ch = (rng.standard_normal((400, 8)) + 1j * rng.standard_normal((400, 8)))
    np.savez(tmp_path / f"{SCEN}_ch2d.npz", ch2d=ch.astype(np.complex64))
    dl._channel_table.cache_clear()
    yield tmp_path
    dl._channel_table.cache_clear()


def _load(data_dir, **kw):
    base = dict(snr_db=10.0, n_train=64, p=2, noise_kind="student_t_4",
                seed=42, data_dir=data_dir)
    base.update(kw)
    return dl.load_deepmimo(SCEN, **base)


def test_legacy_streams_overlap_and_spawn_removes_it(data_dir) -> None:
    # Legacy: pilot of replicate b+1 = first p rows of the noise numerator of b.
    _, S_next, _ = _load(data_dir, seed=43)
    z = np.random.default_rng(43).standard_normal((16, 64))
    assert np.array_equal(S_next, z[:2])
    # Spawned streams: pilots and channels differ from the legacy ones.
    _, S_sp, _ = _load(data_dir, seed=43, seed_scheme="spawn")
    assert not np.array_equal(S_sp, S_next)


def test_noise_same_across_snr_by_default_and_independent_on_request(data_dir) -> None:
    for scheme in ("legacy", "spawn"):
        Y1, S1, H1 = _load(data_dir, snr_db=0.0, seed_scheme=scheme)
        Y2, S2, H2 = _load(data_dir, snr_db=10.0, seed_scheme=scheme)
        assert np.array_equal(S1, S2) and np.array_equal(H1, H2)
        N1, N2 = Y1 - H1 @ S1, Y2 - H2 @ S2
        r = N1 / N2
        assert np.allclose(r, r.flat[0])          # same shape, rescaled only
        Y1i, S1i, H1i = _load(data_dir, snr_db=0.0, seed_scheme=scheme,
                              noise_per_snr=True)
        Y2i, S2i, H2i = _load(data_dir, snr_db=10.0, seed_scheme=scheme,
                              noise_per_snr=True)
        assert np.array_equal(S1i, S2i) and np.array_equal(H1i, H2i)
        r = (Y1i - H1i @ S1i) / (Y2i - H2i @ S2i)
        assert not np.allclose(r, r.flat[0])      # independent draws


def test_realised_scaling_hits_snr_exactly_nominal_in_expectation(data_dir) -> None:
    Y, S, H = _load(data_dir, snr_db=5.0)
    N = Y - H @ S
    snr = 10 * np.log10(np.mean((H @ S) ** 2) / np.mean(N ** 2))
    assert abs(snr - 5.0) < 1e-9
    # Nominal: mean noise variance over many replicates = (1/k) 10^-0.5.
    v = []
    for s in range(300):
        Y, S, H = _load(data_dir, snr_db=5.0, seed=s, n_train=256,
                        seed_scheme="spawn", snr_scaling="nominal")
        v.append(np.mean((Y - H @ S) ** 2))
    target = (1.0 / 16) * 10 ** -0.5
    assert abs(np.mean(v) / target - 1.0) < 0.05


def test_nominal_scaling_rejects_infinite_variance(data_dir) -> None:
    with pytest.raises(ValueError, match="nominal"):
        _load(data_dir, noise_kind="student_t_2", snr_scaling="nominal")


def test_invalid_mal_alpha_raises_unless_clip_legacy(data_dir) -> None:
    with pytest.raises(dl.InvalidAlphaError):
        _load(data_dir, noise_kind="matrix_mal", alpha_mal=1.0)
    Y, S, H = _load(data_dir, noise_kind="matrix_mal", alpha_mal=1.0,
                    alpha_policy="clip_legacy")
    assert np.all(np.isfinite(Y))


def test_unknown_options_rejected(data_dir) -> None:
    with pytest.raises(ValueError):
        _load(data_dir, seed_scheme="nope")
    with pytest.raises(ValueError):
        _load(data_dir, snr_scaling="nope")


def test_resolve_data_dir_env_order(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEEPMIMO_DATA_DIR", raising=False)
    monkeypatch.delenv("RESEARCH_DATA_ROOT", raising=False)
    assert dl.resolve_data_dir() == dl._LEGACY_DATA_DIR
    lake = tmp_path / "store" / "deepmimo-v4" / "data"
    lake.mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_DATA_ROOT", str(tmp_path))
    assert dl.resolve_data_dir() == lake
    monkeypatch.setenv("DEEPMIMO_DATA_DIR", "/x")
    assert str(dl.resolve_data_dir()) == "/x"
    assert str(dl.resolve_data_dir("/y")) == "/y"


def test_run_py_forwards_deepmimo_options() -> None:
    from experiments.mv_patp_mimo import run

    cfg = {"data_source": "deepmimo", "replicates": 2,
           "deepmimo_options": {"seed_scheme": "spawn", "noise_per_snr": True},
           "grid": {"scenarios": [SCEN], "snr_db": [0.0]}, "estimators": ["ols"]}
    (cell,) = run.expand_grid(cfg)
    assert dict(cell.loader_opts) == {"seed_scheme": "spawn", "noise_per_snr": True}
    assert "seed_schemespawn" in cell.data_id()
    legacy = run.expand_grid({**cfg, "deepmimo_options": {}})[0]
    assert legacy.loader_opts == () and "seed_scheme" not in legacy.data_id()
