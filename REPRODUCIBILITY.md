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

### First-round revision study (R1)

Every R1 Monte Carlo study and its paired bootstrap run in sequence from one
script (about 8 h for the studies up to `r1_consolidated` with 8 workers,
see `data/results/r1_logs/progress.txt`):

```bash
cd python
PY=python3 W=8 bash scripts/r1_run_all.sh
```

| Directory under `data/results/` | What it holds |
|---|---|
| `r1_consolidated`, `r1_consolidated_p4` | canonical grid, p = 2 and p = 4 |
| `r1_mal_valid` | matched matrix asymmetric Laplace (OLS = MLE) |
| `r1_nk_sweep` | pilot length n/k |
| `r1_fixed_alpha` | fixed alpha against adaptive alpha |
| `r1_estep` | exact against approximate E-step |
| `r1_gal_k1` | scalar GAL check, k = 1 |
| `r1_correlated` | AR(1)-t noise, whitened Huber |
| `r1_middleton` | Middleton Class A |
| `r1_deepmimo` | DeepMIMO v4 channels (export with `scripts/data_download/export_deepmimo_channels.py`) |
| `r1_huber_tuned`, `r1_middleton_tuned` | Huber with fixed c in {0.7, 1.0, 1.345, 1.7, 2.0, 2.5} and with c chosen from the data (`src/mv_patp/baseline/huber_adaptive_c.py`) |
| `r1_sandwich_coverage` | coverage of the sandwich standard errors |

Each directory has `cells.csv` and `manifest.json`, the per-replicate errors
`reps.csv.gz` (`coverage_reps.csv.gz` for the coverage study), and the paired
bootstrap intervals `paired_bootstrap*.csv` (B = 2000, seed 20261104). In
`paired_bootstrap*.csv`, rows `are_vs_ols` hold MSE(OLS)/MSE(estimator), so a
value above 1 favours the estimator; rows `mse_ratio_<target>_over_<ref>` hold
MSE(target)/MSE(ref), so a value above 1 means the target is worse. Any
bootstrap can be recomputed from the stored replicates without rerunning the
simulation, for example

```bash
python scripts/r1_paired_bootstrap.py ../data/results/r1_middleton/reps.csv.gz \
  -o /tmp/pb.csv --B 2000 --seed 20261104
python scripts/r1_huber_tuned_summary.py ../data/results
python scripts/r1_huber_tuned_middleton_summary.py ../data/results
```

The two summary scripts write the quoted rows (`*_source_rows.csv`) and the
paired comparison of data-chosen c against c = 1.345
(`huber_tuned_adapt_vs_1p345.csv`).

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
