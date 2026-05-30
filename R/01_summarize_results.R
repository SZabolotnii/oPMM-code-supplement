#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "R/01_summarize_results.R"
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)

results_root <- file.path(root, "data", "results")
out_dir <- file.path(root, "R", "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_cells <- function(path) {
  csv <- file.path(path, "cells.csv")
  if (!file.exists(csv)) {
    return(NULL)
  }
  dat <- read.csv(csv, stringsAsFactors = FALSE)
  dat$run_id <- basename(path)
  for (col in c("data_source", "scenario", "snr_db")) {
    if (!col %in% names(dat)) {
      dat[[col]] <- NA
    }
  }
  dat
}

run_dirs <- list.dirs(results_root, recursive = FALSE, full.names = TRUE)
cells <- do.call(rbind, Filter(Negate(is.null), lapply(run_dirs, read_cells)))
if (is.null(cells) || nrow(cells) == 0) {
  stop("No cells.csv files found under data/results")
}

write.csv(cells, file.path(out_dir, "all_cells.csv"), row.names = FALSE)

group_cols <- c("run_id", "k", "p", "n", "alpha_MAL", "sigma_structure",
                "M_norm", "data_source", "scenario", "snr_db")
cells$group_id <- do.call(paste, c(cells[group_cols], sep = "|"))

ols <- cells[cells$estimator == "ols", c("group_id", "mse_mean")]
names(ols)[2] <- "mse_ols"
are <- merge(cells, ols, by = "group_id", all.x = TRUE)
are$ARE <- are$mse_ols / are$mse_mean
are <- are[order(are$run_id, are$group_id, are$estimator), ]
write.csv(are, file.path(out_dir, "are_summary.csv"), row.names = FALSE)

headline <- are[
  are$run_id == "synthetic_full_v1" &
    are$k == 2 & are$p == 2 & are$n == 512 &
    are$alpha_MAL == 1.0 &
    are$estimator %in% c("ols", "matrix_pmm2", "matrix_pmm3", "opmm_alpha"),
  c("sigma_structure", "estimator", "mse_mean", "ARE", "n_replicates")
]
write.csv(headline, file.path(out_dir, "synthetic_headline_are.csv"), row.names = FALSE)

deepmimo <- cells[
  cells$run_id == "deepmimo_v2" &
    cells$estimator %in% c("ols", "matrix_pmm2", "matrix_pmm3", "opmm_alpha"),
  c("scenario", "snr_db", "estimator", "mse_mean", "mse_std", "n_replicates")
]
deepmimo <- deepmimo[order(deepmimo$scenario, deepmimo$snr_db, deepmimo$estimator), ]
write.csv(deepmimo, file.path(out_dir, "deepmimo_summary.csv"), row.names = FALSE)

runtime <- aggregate(
  wall_clock_mean_s ~ run_id + estimator,
  data = cells,
  FUN = function(x) mean(x, na.rm = TRUE)
)
write.csv(runtime, file.path(out_dir, "runtime_summary.csv"), row.names = FALSE)

message("Wrote R summary CSV files to ", out_dir)
