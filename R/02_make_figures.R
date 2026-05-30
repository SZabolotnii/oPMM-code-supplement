#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "R/02_make_figures.R"
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)

out_dir <- file.path(root, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

are_path <- file.path(root, "R", "results", "are_summary.csv")
deep_path <- file.path(root, "R", "results", "deepmimo_summary.csv")
if (!file.exists(are_path) || !file.exists(deep_path)) {
  stop("Run R/01_summarize_results.R first")
}

are <- read.csv(are_path, stringsAsFactors = FALSE)
deep <- read.csv(deep_path, stringsAsFactors = FALSE)

selected_estimators <- c("matrix_pmm2", "matrix_pmm3", "opmm_alpha")
selected_regimes <- c("student_t_3", "student_t_4", "student_t_5", "student_t_1000", "I")

synthetic <- are[
  are$run_id == "synthetic_full_v1" &
    are$k == 2 & are$p == 2 & are$n == 512 &
    are$alpha_MAL == 1.0 &
    are$estimator %in% selected_estimators &
    are$sigma_structure %in% selected_regimes,
]

pdf(file.path(out_dir, "r_fig_synthetic_are.pdf"), width = 7, height = 4)
if (nrow(synthetic) > 0) {
  mat <- matrix(NA_real_, nrow = length(selected_estimators), ncol = length(selected_regimes))
  rownames(mat) <- selected_estimators
  colnames(mat) <- c("t3", "t4", "t5", "t1000", "Gauss")
  for (i in seq_along(selected_estimators)) {
    for (j in seq_along(selected_regimes)) {
      row <- synthetic[
        synthetic$estimator == selected_estimators[[i]] &
          synthetic$sigma_structure == selected_regimes[[j]],
      ]
      if (nrow(row) > 0) {
        mat[i, j] <- row$ARE[[1]]
      }
    }
  }
  barplot(
    mat,
    beside = TRUE,
    col = c("#4472C4", "#ED7D31", "#70AD47"),
    ylim = c(0, max(2, mat, na.rm = TRUE)),
    ylab = "ARE = MSE_OLS / MSE_method",
    xlab = "Tail regime",
    main = "Synthetic headline ARE"
  )
  abline(h = 1, lty = 2, col = "gray40")
  legend("topright", legend = rownames(mat), fill = c("#4472C4", "#ED7D31", "#70AD47"),
         bty = "n", cex = 0.8)
} else {
  plot.new()
  text(0.5, 0.5, "No synthetic headline rows found")
}
dev.off()

pdf(file.path(out_dir, "r_fig_deepmimo_nmse.pdf"), width = 7, height = 4)
if (nrow(deep) > 0) {
  scenarios <- unique(deep$scenario)
  old_par <- par(mfrow = c(1, length(scenarios)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  colors <- c(ols = "gray40", matrix_pmm2 = "#4472C4", matrix_pmm3 = "#ED7D31", opmm_alpha = "#70AD47")
  for (scen in scenarios) {
    sub <- deep[deep$scenario == scen, ]
    yrange <- range(sub$mse_mean, na.rm = TRUE)
    plot(
      NA,
      xlim = range(sub$snr_db, na.rm = TRUE),
      ylim = yrange,
      log = "y",
      xlab = "SNR (dB)",
      ylab = "Mean MSE per H element",
      main = scen
    )
    grid()
    for (est in names(colors)) {
      rows <- sub[sub$estimator == est, ]
      rows <- rows[order(rows$snr_db), ]
      if (nrow(rows) > 0) {
        lines(rows$snr_db, rows$mse_mean, type = "b", col = colors[[est]], pch = 19)
      }
    }
    legend("topright", legend = names(colors), col = colors, pch = 19, bty = "n", cex = 0.7)
  }
} else {
  plot.new()
  text(0.5, 0.5, "No DeepMIMO rows found")
}
dev.off()

message("Wrote R PDF figures to ", out_dir)
