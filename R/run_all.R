#!/usr/bin/env Rscript

message("MV-PATP-MIMO R supplement")

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "R/run_all.R"
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)

source(file.path(root, "R", "01_summarize_results.R"), local = TRUE)
source(file.path(root, "R", "02_make_figures.R"), local = TRUE)

message("Done. Outputs:")
message("  R/results/all_cells.csv")
message("  R/results/are_summary.csv")
message("  R/results/deepmimo_summary.csv")
message("  figures/r_fig_synthetic_are.pdf")
message("  figures/r_fig_deepmimo_nmse.pdf")
