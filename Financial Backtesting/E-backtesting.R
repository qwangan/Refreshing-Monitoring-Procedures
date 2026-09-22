## Section 5.3: resetting e-backtesting of NASDAQ Composite forecasts.

args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grepl("^--file=", args)]
script_path <- sub("^--file=", "", script_arg[[1]])
script_path <- gsub("~\\+~", " ", script_path)
root <- dirname(normalizePath(script_path))
out_dir <- file.path(root, "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

user_args <- commandArgs(trailingOnly = TRUE)
input_argument <- user_args[startsWith(user_args, "--input=")]
input_file <- if (length(input_argument)) {
  sub("--input=", "", input_argument[[length(input_argument)]], fixed = TRUE)
} else {
  file.path(root, "NASDAQ.rds")
}
input <- readRDS(normalizePath(input_file, mustWork = TRUE))
backtest <- input$backtest
figure14 <- if (!is.null(input$figure14)) input$figure14 else input$figure10

## Figure 14: negated percentage log-returns and ES forecasts, 2000-2025.
plot_dates <- as.Date(figure14$date)
forecast_colors <- c("orange", "green3", "dodgerblue2", "#E84A5F")
event_dates <- as.Date(c("2008-09-15", "2020-03-11"))

pdf(file.path(out_dir, "Figure14.pdf"), width = 11, height = 5.2)
layout(matrix(1:2, nrow = 1))
par(mar = c(4.5, 4.7, 1, 0.8))
plot(plot_dates, figure14$negated_percentage_log_return, type = "l",
     xlab = "dates", ylab = "negated percentage log returns", lwd = 0.55)
abline(v = event_dates, lty = 3, col = "gray35")
es_forecasts <- rbind(
  figure14$ES_0975_normal,
  figure14$ES_0975_t,
  figure14$ES_0975_skewed_t,
  figure14$ES_0975_empirical
)
plot(plot_dates, es_forecasts[1, ], type = "n", xlab = "dates",
     ylab = expression(ES[0.975] ~ forecast), ylim = range(es_forecasts))
for (i in seq_len(nrow(es_forecasts))) {
  lines(plot_dates, es_forecasts[i, ], col = forecast_colors[i], lwd = 1)
}
abline(v = event_dates, lty = 3, col = "gray35")
legend("topleft", c("normal", "t", "skewed-t", "empirical"),
       col = forecast_colors, lwd = 1, bty = "n")
dev.off()

## GREE, GREL, and GREM test supermartingales.
rolling_window <- 500L
lambda_max <- 0.5
fdr_target <- 0.2
alpha <- uniroot(
  function(a) a * (1 + log(1 / a)) - fdr_target,
  c(1e-8, fdr_target), tol = 1e-14
)$root
gamma <- 1 / alpha

y <- backtest$y
dates <- as.Date(backtest$dates)
es <- backtest$ESout2
var <- backtest$VaRout2b
p <- backtest$nvec[2]
horizon <- length(dates)

e_statistic <- function(x, r, z) {
  pmax(x - z, 0) / ((1 - p) * (r - z))
}

empirical_kelly <- function(e) {
  e_bar <- mean(e)
  denominator <- mean((e - e_bar)^2) + (e_bar - 1)^2
  raw <- if (denominator > 0) (e_bar - 1) / denominator else 0
  max(min(raw, lambda_max), 0)
}

compute_path <- function(r, z) {
  log_gree <- log_grel <- log_grem <- numeric(horizon)
  lambda_gree <- lambda_grel <- numeric(horizon)
  for (i in seq_len(horizon)) {
    history <- i:(i + rolling_window - 1L)
    current <- i + rolling_window
    lambda_gree[i] <- empirical_kelly(e_statistic(y[history], r[history], z[history]))
    lambda_grel[i] <- empirical_kelly(e_statistic(y[history], r[current], z[current]))
    current_e <- e_statistic(y[current], r[current], z[current])
    old_gree <- if (i == 1L) 0 else log_gree[i - 1L]
    old_grel <- if (i == 1L) 0 else log_grel[i - 1L]
    log_gree[i] <- old_gree + log1p(lambda_gree[i] * (current_e - 1))
    log_grel[i] <- old_grel + log1p(lambda_grel[i] * (current_e - 1))
    largest <- max(log_gree[i], log_grel[i])
    log_grem[i] <- largest +
      log(exp(log_gree[i] - largest) + exp(log_grel[i] - largest)) - log(2)
  }
  list(log_grem = log_grem)
}

forecast_pairs <- list(
  list(r = es[1, ], z = var[1, ]),
  list(r = es[2, ], z = var[2, ]),
  list(r = es[3, ], z = var[3, ]),
  list(r = es[4, ], z = var[4, ]),
  list(r = 1.1 * es[3, ], z = var[3, ])
)
labels <- c("Normal", "t", "Skewed-t", "Empirical", "Skewed-t +10% ES")
colors <- c("orange", "green", "blue", "red", "black")
one_shot <- do.call(rbind, lapply(forecast_pairs, function(x) {
  compute_path(x$r, x$z)$log_grem
}))

reset <- function(log_path) {
  increments <- c(log_path[1], diff(log_path))
  answer <- numeric(length(log_path))
  answer[1] <- increments[1]
  for (i in 2:length(log_path)) {
    answer[i] <- if (answer[i - 1] >= log(gamma)) {
      increments[i]
    } else {
      answer[i - 1] + increments[i]
    }
  }
  answer
}
reset_process <- t(apply(one_shot, 1, reset))

localized_blocks <- function(path) {
  rejection_times <- which(path >= log(gamma))
  if (!length(rejection_times)) {
    return(data.frame(start = integer(), end = integer()))
  }
  block_start <- 1L
  answer <- vector("list", length(rejection_times))
  for (j in seq_along(rejection_times)) {
    tau <- rejection_times[j]
    block <- c(0, path[block_start:tau])
    last_minimum <- max(which(block == min(block)))
    sigma <- if (last_minimum == 1L) block_start else block_start + last_minimum - 1L
    answer[[j]] <- data.frame(start = sigma, end = tau)
    block_start <- tau + 1L
  }
  do.call(rbind, answer)
}

localized_rejection_blocks <- do.call(rbind, lapply(seq_along(labels), function(i) {
  blocks <- localized_blocks(reset_process[i, ])
  if (!nrow(blocks)) return(NULL)
  data.frame(
    forecast = labels[i],
    rejection = seq_len(nrow(blocks)),
    localized_start = dates[blocks$start],
    rejection_date = dates[blocks$end]
  )
}))
write.csv(
  localized_rejection_blocks,
  file.path(out_dir, "localized_rejection_blocks.csv"),
  row.names = FALSE
)

## Figure 15: original GREM process and four resetting processes.
ylim <- backtest$e_lim
block_colors <- c("#FDB462", "#80B1D3", "#B3DE69", "#FCCDE5",
                  "#BC80BD", "#CCEBC5")
pdf(file.path(out_dir, "Figure15.pdf"), width = 12, height = 6.2)
layout(matrix(c(1, 2, 3, 1, 4, 5), nrow = 2, byrow = TRUE),
       widths = c(1.05, 1.5, 1.5))
par(mar = c(4.1, 4.1, 2, 0.8))
plot(dates, one_shot[1, ], type = "n", xlab = "dates",
     ylab = "log test supermartingale", ylim = ylim)
for (i in seq_along(labels)) lines(dates, one_shot[i, ], col = colors[i])
abline(h = log(gamma), lty = 2, col = "gray45")
abline(v = event_dates, lty = 3, col = "gray35")
legend("topleft", labels, col = colors, lwd = 1, bty = "n", cex = 0.68)
title("One-shot GREM")

for (i in 1:4) {
  blocks <- localized_blocks(reset_process[i, ])
  plot(dates, reset_process[i, ], type = "n", xlab = "dates",
       ylab = "log resetting process", ylim = ylim)
  if (nrow(blocks)) {
    for (j in seq_len(nrow(blocks))) {
      rect(dates[blocks$start[j]], ylim[1], dates[blocks$end[j]], ylim[2],
           col = adjustcolor(block_colors[(j - 1) %% length(block_colors) + 1], 0.28),
           border = NA)
    }
  }
  abline(h = log(gamma), lty = 2, col = "gray45")
  abline(v = event_dates, lty = 3, col = "gray35")
  lines(dates, reset_process[i, ], col = colors[i])
  rejection_times <- which(reset_process[i, ] >= log(gamma))
  points(dates[rejection_times], reset_process[i, rejection_times], pch = 16,
         cex = 0.55, col = colors[i])
  title(labels[i])
}
dev.off()

print(localized_rejection_blocks)
