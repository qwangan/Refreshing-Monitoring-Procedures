## Section 5.1: independent normal simulations.

args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grepl("^--file=", args)]
root <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
out_dir <- file.path(root, "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

T_horizon <- 1500L
lambda <- 0.5
gamma <- 11
n_rep <- 1000L
t_index <- seq_len(T_horizon)

localized_blocks <- function(refreshed) {
  rejection_times <- which(refreshed >= gamma)
  if (!length(rejection_times)) {
    return(data.frame(start = integer(), end = integer()))
  }
  block_start <- 1L
  answer <- vector("list", length(rejection_times))
  for (j in seq_along(rejection_times)) {
    tau <- rejection_times[j]
    block_path <- c(1, refreshed[block_start:tau])
    last_minimum <- max(which(block_path == min(block_path)))
    sigma <- if (last_minimum == 1L) block_start else block_start + last_minimum - 1L
    answer[[j]] <- data.frame(start = sigma, end = tau)
    block_start <- tau + 1L
  }
  do.call(rbind, answer)
}

run_path <- function(mu) {
  x <- rnorm(length(mu), mu, 1)
  e_value <- exp(x - 0.5)
  factor <- 1 - lambda + lambda * e_value
  original <- cumprod(factor)
  refreshed <- numeric(length(mu))
  for (t in seq_along(mu)) {
    refreshed[t] <- if (t == 1L || refreshed[t - 1L] >= gamma) {
      factor[t]
    } else {
      refreshed[t - 1L] * factor[t]
    }
  }
  list(x = x, e_value = e_value, original = original,
       refreshed = refreshed, blocks = localized_blocks(refreshed))
}

path_metrics <- function(path, alternative) {
  covered <- rep(FALSE, length(alternative))
  interval_true <- logical(nrow(path$blocks))
  if (nrow(path$blocks)) {
    for (j in seq_len(nrow(path$blocks))) {
      covered[path$blocks$start[j]:path$blocks$end[j]] <- TRUE
      interval_true[j] <- any(alternative[path$blocks$start[j]:path$blocks$end[j]])
    }
  }
  true_covered <- sum(alternative & covered)
  false_covered <- sum(!alternative & covered)
  c(
    power = true_covered / max(sum(alternative), 1L),
    coverage_fdp = false_covered / max(sum(covered), 1L),
    iou = true_covered / max(sum(alternative | covered), 1L),
    localized_fdp = if (length(interval_true)) mean(!interval_true) else 0
  )
}

signal_intervals <- function(alternative) {
  runs <- rle(alternative)
  ends <- cumsum(runs$lengths)
  starts <- ends - runs$lengths + 1L
  data.frame(start = starts[runs$values], end = ends[runs$values])
}

plot_refreshing_path <- function(path, alternative, file, shocks = FALSE) {
  y <- log(path$refreshed)
  ylim <- range(c(y, log(gamma)), finite = TRUE)
  ylim[1] <- max(ylim[1], -2)
  pdf(file, width = 8, height = 4.8)
  par(mar = c(4.4, 4.8, 0.8, 0.8))
  plot(t_index, y, type = "n", xlab = "t", ylab = "log refreshing process",
       ylim = ylim)
  signals <- signal_intervals(alternative)
  if (shocks) {
    abline(v = signals$start, col = adjustcolor("#009E73", 0.75), lty = 2)
  } else {
    for (j in seq_len(nrow(signals))) {
      rect(signals$start[j], ylim[1], signals$end[j], ylim[2],
           col = adjustcolor("#009E73", 0.14), border = "#006B4F")
    }
  }
  if (nrow(path$blocks)) {
    for (j in seq_len(nrow(path$blocks))) {
      rect(path$blocks$start[j], ylim[1], path$blocks$end[j], ylim[2],
           col = adjustcolor("#E64B35", 0.24), border = "#B83220")
    }
  }
  abline(h = log(gamma), col = "gray45", lty = 2)
  lines(t_index, y, col = "#0072B2", lwd = 1.2)
  rejection_times <- which(path$refreshed >= gamma)
  points(rejection_times, y[rejection_times], pch = 16, cex = 0.55,
         col = "#9E2A16")
  dev.off()
}

alternative_recurring <- ((t_index - 1L) %% 300L) >= 200L
alternative_shocks <- t_index %% 300L == 0L
mu_recurring <- as.numeric(alternative_recurring)
mu_shocks <- 50 * as.numeric(alternative_shocks)

pdf(file.path(out_dir, "Figure5.pdf"), width = 8, height = 2.6)
plot(t_index, mu_recurring, type = "s", xlab = "t", ylab = expression(mu[t]),
     col = "#009E73", lwd = 1.2)
dev.off()

set.seed(20260811L)
representative_recurring <- run_path(mu_recurring)
plot_refreshing_path(representative_recurring, alternative_recurring,
                     file.path(out_dir, "Figure6.pdf"))

set.seed(20260806L)
representative_shocks <- run_path(mu_shocks)
plot_refreshing_path(representative_shocks, alternative_shocks,
                     file.path(out_dir, "Figure7.pdf"), shocks = TRUE)

monte_carlo <- function(mu, alternative, seed) {
  set.seed(seed)
  values <- matrix(NA_real_, nrow = n_rep, ncol = 4L)
  colnames(values) <- c("power", "coverage_fdp", "iou", "localized_fdp")
  for (i in seq_len(n_rep)) {
    values[i, ] <- path_metrics(run_path(mu), alternative)
  }
  list(mean = colMeans(values), sd = apply(values, 2, sd))
}

recurring_mc <- monte_carlo(mu_recurring, alternative_recurring, 20260901L)
shocks_mc <- monte_carlo(mu_shocks, alternative_shocks, 20260902L)

table1 <- data.frame(
  case = c("Alternating periods", "Single-point shocks"),
  power = c(recurring_mc$mean["power"], shocks_mc$mean["power"]),
  coverage_fdp = c(recurring_mc$mean["coverage_fdp"], shocks_mc$mean["coverage_fdp"]),
  iou = c(recurring_mc$mean["iou"], shocks_mc$mean["iou"]),
  localized_fdp = c(recurring_mc$mean["localized_fdp"], shocks_mc$mean["localized_fdp"]),
  n_rep = n_rep
)
write.csv(table1, file.path(out_dir, "Table1.csv"), row.names = FALSE)
print(table1)
