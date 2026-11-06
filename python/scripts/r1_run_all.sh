#!/usr/bin/env bash
# R1 (DSP-D-26-03654) Monte Carlo campaign: runs every R1 study in sequence,
# then the paired bootstrap for each. Short studies first, so partial results
# arrive early. Each step logs to $OUT/r1_logs/<name>.log; a failed step does
# not stop the others. Usage: bash scripts/r1_run_all.sh  (from python/)
# Environment: PY (interpreter, default python3), OUT (results root, default
# ../data/results), RESEARCH_DATA_ROOT (DeepMIMO export, see scripts/data_download),
# W (worker processes, default 8).
# The published snapshots store reps.csv gzipped (reps.csv.gz); the bootstrap
# script reads either form, e.g.
#   python scripts/r1_paired_bootstrap.py ../data/results/r1_middleton/reps.csv.gz
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
OUT=${OUT:-$(cd .. && pwd)/data/results}
LOG=$OUT/r1_logs
mkdir -p "$LOG"
export PYTHONPATH=src
export RESEARCH_DATA_ROOT=${RESEARCH_DATA_ROOT:-$HOME/datasets}
W=${W:-8}
BS="--B 2000 --seed 20261104"

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/progress.txt"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE  $name" >> "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAIL  $name (exit $?)" >> "$LOG/progress.txt"
  fi
}
run() {  # run <name> <config>
  step "$1" $PY -m experiments.mv_patp_mimo.run --config "experiments/mv_patp_mimo/$2" \
    --output-dir "$OUT/$1" --seed 42 --workers $W --per-replicate --log-level WARNING
}
boot() {  # boot <name> [extra args]
  local name=$1; shift
  step "${name}_bootstrap" $PY scripts/r1_paired_bootstrap.py "$OUT/$name/reps.csv" \
    -o "$OUT/$name/paired_bootstrap.csv" $BS "$@"
}

step r1_gal_k1 $PY scripts/r1_gal_k1.py -o "$OUT/r1_gal_k1" --reps 500 --workers $W
boot r1_gal_k1 --target opmm_alpha --refs scalar_opmm opmm_alpha_unguarded
run  r1_nk_sweep config_r1_nk_sweep.yaml
boot r1_nk_sweep --target opmm_alpha --refs opmm_alpha_unguarded
step r1_estep $PY scripts/r1_estep_ablation.py -o "$OUT/r1_estep" --reps 500 --workers $W
boot r1_estep --target opmm_alpha_exact --refs opmm_alpha_unguarded opmm_alpha
run  r1_middleton config_r1_middleton.yaml
boot r1_middleton
run  r1_deepmimo config_r1_deepmimo.yaml
boot r1_deepmimo
step r1_sandwich_coverage $PY scripts/r1_sandwich_coverage.py -o "$OUT/r1_sandwich_coverage" --workers $W
run  r1_correlated config_r1_correlated.yaml
boot r1_correlated
run  r1_fixed_alpha config_r1_fixed_alpha.yaml
boot r1_fixed_alpha --target opmm_alpha --refs opmm_alpha_fixed0p3 opmm_alpha_fixed0p536 opmm_alpha_fixed0p77 opmm_alpha_fixed2p0 opmm_alpha_unguarded
run  r1_consolidated_p4 config_r1_consolidated_p4.yaml
boot r1_consolidated_p4
run  r1_mal_valid config_r1_mal_valid.yaml
boot r1_mal_valid
run  r1_consolidated config_r1_consolidated.yaml
boot r1_consolidated
# Paired ratios quoted in the text: matrix PMM2 vs oPMM_alpha (canonical grid)
# and whitened Huber vs Huber (AR(1)-t).
step r1_consolidated_pmm2_bootstrap $PY scripts/r1_paired_bootstrap.py \
  "$OUT/r1_consolidated/reps.csv" -o "$OUT/r1_consolidated/paired_bootstrap_pmm2_over_opmm.csv" \
  $BS --target matrix_pmm2 --refs opmm_alpha
step r1_correlated_whuber_bootstrap $PY scripts/r1_paired_bootstrap.py \
  "$OUT/r1_correlated/reps.csv" -o "$OUT/r1_correlated/paired_bootstrap_whuber_over_huber.csv" \
  $BS --target whitened_huber --refs huber_irls
# Rv1-C11: Huber with tuned constants (fixed c grid + data-driven c) on the
# r1_consolidated data (same grid, seed scheme and replicates).
run  r1_huber_tuned config_r1_huber_tuned.yaml
boot r1_huber_tuned --target opmm_alpha --refs huber_irls huber_adaptive_c \
  huber_c0p7 huber_c1p0 huber_c1p345 huber_c1p7 huber_c2p0 huber_c2p5
# Rv1-C11 on Middleton Class A: same estimators on the r1_middleton data
# (same grid, seed scheme and replicates), plus adaptive c vs c = 1.345.
run  r1_middleton_tuned config_r1_middleton_tuned.yaml
boot r1_middleton_tuned --target opmm_alpha --refs huber_irls huber_adaptive_c \
  huber_c0p7 huber_c1p0 huber_c1p345 huber_c1p7 huber_c2p0 huber_c2p5
step r1_middleton_tuned_adapt_bootstrap $PY scripts/r1_paired_bootstrap.py \
  "$OUT/r1_middleton_tuned/reps.csv" -o "$OUT/r1_middleton_tuned/paired_bootstrap_adapt_vs_irls.csv" \
  $BS --target huber_adaptive_c --refs huber_irls
echo "$(date '+%F %T') ALL FINISHED" >> "$LOG/progress.txt"
