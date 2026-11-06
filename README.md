# MV-PATP-MIMO Code Supplement

This directory is the public code supplement for the manuscript:

**Robust MIMO Channel Identification under Matrix-Variate Heavy-Tailed Noise via
Sign-Preserving Fractional Polynomial Maximization**

It is intentionally separated from the manuscript sources. It contains the
open implementation code, deterministic result snapshots, R reproduction
scripts, rendered figures, and a compact Lean 4 formal supplement.

## Contents

- `python/` - reference Python package, experiment drivers, configs, data
  download helpers, and tests. Includes the Middleton Class A DGP
  (`src/mv_patp/data_gen/middleton.py`), the complex-baseband demo
  (`experiments/mv_patp_mimo/run_complex_demo.py`), and the symbolic Recovery
  Theorem check (`theory/recovery_symbolic_check.py`).
- `data/results/` - CSV and manifest snapshots for the result directories used
  by the manuscript. The `r1_*` directories hold the first-round revision
  studies, with per-replicate errors and paired bootstrap intervals; see
  `REPRODUCIBILITY.md`.
- `R/` - lightweight R scripts that summarize the CSV snapshots and render
  independent PDF checks.
- `figures/` - manuscript PDF figures copied from the release build.
- `Lean/`, `lakefile.lean`, `lean-toolchain` - Lean 4 project with symbolic
  definitions and structural checks for the oPMM score and recovery anchor.
- `REPRODUCIBILITY.md` - exact commands for Python, R, and Lean checks.

## Python Quick Start

```bash
cd code_supplement/python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,real_data]"
pytest -q -m "not slow and not gate"
```

Regenerate manuscript figures from the bundled result snapshots:

```bash
python -m experiments.mv_patp_mimo.plot \
  --synthetic-dir ../data/results/synthetic_full_v1 \
  --deepmimo-dir ../data/results/deepmimo_v2 \
  --output-dir ../figures \
  --format pdf
```

## R Quick Start

```bash
cd code_supplement
Rscript R/run_all.R
```

The R scripts regenerate summary CSVs under `R/results/` and independent PDF
checks under `figures/`.

## Lean Quick Start

```bash
cd code_supplement
lake build
```

The Lean files are not a full mechanization of the statistical proof. They are
a compact formal supplement for symbolic definitions, admissible alpha ranges,
the trace-normalization invariant, and the `alpha = 3` PMM3 recovery anchor.
