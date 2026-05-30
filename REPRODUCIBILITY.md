# Reproducibility Notes

## Python

The canonical Python entry points are preserved under `python/`.

Run the fast test subset:

```bash
cd python
pip install -e ".[dev]"
pytest -q -m "not slow and not gate"
```

Regenerate the synthetic smoke run:

```bash
python -m experiments.mv_patp_mimo.run \
  --config experiments/mv_patp_mimo/config_synthetic_quick.yaml \
  --output-dir ../data/results/synthetic_quick_rebuild
```

Regenerate release figures from stored CSV snapshots:

```bash
python -m experiments.mv_patp_mimo.plot \
  --synthetic-dir ../data/results/synthetic_full_v1 \
  --deepmimo-dir ../data/results/deepmimo_v2 \
  --output-dir ../figures \
  --format pdf --format png
```

External datasets are not bundled. Use `python/scripts/data_download/` and the
manuscript data policy when reproducing DeepMIMO, WAIR-D, or RadioML inputs.

### Revision experiments (Middleton Class A, complex baseband, symbolic recovery)

Middleton Class A canonical impulsive noise (manuscript Table V), three
impulsive indices `A` spanning excess kurtosis `gamma_4 = 3/(A(1+G')^2)`:

```bash
cd python
python -m experiments.mv_patp_mimo.run \
  --config experiments/mv_patp_mimo/config_middleton.yaml \
  --seed 42 --workers 4
# snapshot: ../data/results/middleton_v1/cells.csv
```

Complex-baseband proof-of-concept via the real-equivalent embedding
(manuscript Section V-G):

```bash
python -m experiments.mv_patp_mimo.run_complex_demo \
  --k 4 --p 2 --n 512 --df 4 --reps 200 --seed 42
# snapshot: ../data/results/complex_demo_v1/cells.csv
```

Symbolic verification of the Recovery Theorem (Appendix A), `sympy` only:

```bash
pip install -e ".[dev]"            # provides sympy
python theory/recovery_symbolic_check.py
```

The symbolic script proves `g(3; C) = 1 - gamma4^2/(6 + 9*gamma4 + gamma6)`
under `kappa3 = 0` and confirms the sole `kappa3` contribution is the
`+10*kappa3^2` moment-cumulant term (i.e. no spurious `kappa3^3`).

## R

The R layer is intentionally dependency-light and uses base R only. It reads
the CSV snapshots in `data/results/`, writes summary tables into `R/results/`,
and renders independent PDF checks into `figures/`.

```bash
Rscript R/run_all.R
```

## Lean

The Lean supplement is pinned by `lean-toolchain`.

```bash
lake build
```

The formal scope is structural. It records definitions and small symbolic
facts used by the manuscript; it does not claim to mechanize the matrix MAL
asymptotic theory or Monte Carlo results.
