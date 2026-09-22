## Reconstruct the NASDAQ VaR and ES forecasts used in Section 5.3.
##
## The script is self-contained apart from the CRAN packages rugarch and sgt.
## It reads daily NASDAQ Composite closing prices, constructs negated percentage
## log-returns, estimates rolling AR(1)-GARCH(1,1) forecasts under normal, t,
## and skewed-t innovations, and also computes rolling empirical forecasts.

args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- args_all[grepl("^--file=", args_all)]
script_path <- sub("^--file=", "", script_arg[[1]])
script_path <- gsub("~\\+~", " ", script_path)
root <- dirname(normalizePath(script_path))
args <- commandArgs(trailingOnly = TRUE)
detected_cores <- parallel::detectCores()
default_cores <- if (is.na(detected_cores)) 1L else max(1L, detected_cores - 1L)

argument <- function(name, default) {
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (!length(value)) default else sub(prefix, "", value[[length(value)]], fixed = TRUE)
}

price_file <- normalizePath(
  argument("prices", file.path(root, "NASDAQ.csv")),
  mustWork = TRUE
)
output_file <- argument("output", file.path(root, "NASDAQ-regenerated.rds"))
reference_file <- argument("reference", file.path(root, "NASDAQ.rds"))
checkpoint_dir <- argument(
  "checkpoint-dir", file.path(root, "NASDAQ-forecast-checkpoints")
)
cores <- as.integer(argument("cores", default_cores))
chunk_size <- as.integer(argument("chunk-size", 100L))
validate_date <- argument("validate-date", "")

if (!is.finite(cores) || cores < 1L) stop("--cores must be a positive integer")
if (!is.finite(chunk_size) || chunk_size < 1L) {
  stop("--chunk-size must be a positive integer")
}
for (package in c("rugarch", "sgt")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("Install the CRAN package '%s' before running this script", package))
  }
}

rolling_window <- 500L
var_levels <- c(0.95, 0.99, 0.875, 0.975)
es_levels <- c(0.875, 0.975)
figure_start <- as.Date("2000-01-03")
monitoring_start <- as.Date("2005-01-04")
model_names <- c("normal", "t", "skewed_t", "empirical")

prices <- read.csv(price_file, stringsAsFactors = FALSE)
if (!identical(names(prices), c("Date", "Close"))) {
  stop("NASDAQ.csv must have exactly two columns named Date and Close")
}
prices$Date <- as.Date(prices$Date)
prices$Close <- as.numeric(prices$Close)
if (anyNA(prices) || any(prices$Close <= 0) || any(diff(prices$Date) <= 0)) {
  stop("Price dates must be unique and increasing, and closing prices must be positive")
}
if (min(prices$Date) != as.Date("1996-01-16") ||
    max(prices$Date) != as.Date("2025-12-31")) {
  stop("The supplied price series must run from 1996-01-16 through 2025-12-31")
}

losses <- -diff(log(prices$Close)) * 100
return_dates <- prices$Date[-1L]
target_indices <- seq.int(rolling_window + 1L, length(losses))
forecast_dates <- return_dates[target_indices]

specifications <- list(
  normal = rugarch::ugarchspec(
    mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
    distribution.model = "norm"
  ),
  t = rugarch::ugarchspec(
    mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
    distribution.model = "std"
  ),
  skewed_t = rugarch::ugarchspec(
    mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
    distribution.model = "sstd"
  )
)

fit_model <- function(specification, observations) {
  fit <- rugarch::ugarchfit(
    specification,
    observations,
    solver = "hybrid",
    solver.control = list(trace = 0)
  )
  if (!is.null(fit@fit$convergence) && fit@fit$convergence != 0) {
    stop("The GARCH optimizer did not converge")
  }
  fit
}

skewed_t_standardized_risk <- function(fit) {
  nu <- as.numeric(rugarch::coef(fit)["shape"])
  skew <- as.numeric(rugarch::coef(fit)["skew"])
  qmodel <- sgt::qsgt(
    var_levels,
    mu = 0,
    sigma = 1,
    lambda = (skew^2 - 1) / (skew^2 + 1),
    p = 2,
    q = nu / 2
  )

  asymmetry <- -(skew^2 - 1) / (skew^2 + 1)
  c_value <- gamma((nu + 1) / 2) /
    (gamma(nu / 2) * sqrt(pi * (nu - 2)))
  a_value <- 4 * asymmetry * c_value * (nu - 2) / (nu - 1)
  b_value <- sqrt(1 + 3 * asymmetry^2 - a_value^2)

  esmodel <- vapply(3:4, function(j) {
    if (qmodel[j] >= a_value / b_value) {
      alpha_tilde <- sgt::psgt(
        b_value / (1 - asymmetry) * (-qmodel[j] + a_value / b_value),
        mu = 0, sigma = 1, lambda = 0, p = 2, q = nu / 2
      )
      es_t <- sqrt((nu - 2) / nu) * nu^(nu / 2) /
        (2 * alpha_tilde * sqrt(pi)) *
        gamma((nu - 1) / 2) / gamma(nu / 2) *
        (stats::qt(1 - alpha_tilde, df = nu)^2 + nu)^((1 - nu) / 2)
      -alpha_tilde / (1 - var_levels[j]) * (1 - asymmetry) *
        (-a_value / b_value - (1 - asymmetry) / b_value * es_t)
    } else {
      reflected <- -asymmetry
      reflected_a <- 4 * reflected * c_value * (nu - 2) / (nu - 1)
      alpha_tilde <- sgt::psgt(
        b_value / (1 - reflected) *
          (sgt::qsgt(
            var_levels[j], mu = 0, sigma = 1, lambda = reflected,
            p = 2, q = nu / 2
          ) + reflected_a / b_value),
        mu = 0, sigma = 1, lambda = 0, p = 2, q = nu / 2
      )
      es_t <- sqrt((nu - 2) / nu) * nu^(nu / 2) /
        (2 * alpha_tilde * sqrt(pi)) *
        gamma((nu - 1) / 2) / gamma(nu / 2) *
        (stats::qt(1 - alpha_tilde, df = nu)^2 + nu)^((1 - nu) / 2)
      -alpha_tilde / (1 - var_levels[j]) * (1 - reflected) *
        (-reflected_a / b_value - (1 - reflected) / b_value * es_t)
    }
  }, numeric(1))
  list(var = qmodel, es = esmodel)
}

forecast_parametric <- function(model, target_index) {
  observations <- losses[(target_index - rolling_window):(target_index - 1L)]
  fit <- fit_model(specifications[[model]], observations)

  if (model == "normal") {
    standardized_var <- rugarch::qdist(
      "norm", p = var_levels, mu = 0, sigma = 1
    )
    standardized_es <- rugarch::ddist(
      "norm",
      rugarch::qdist("norm", p = es_levels, mu = 0, sigma = 1),
      mu = 0,
      sigma = 1
    ) / (1 - es_levels)
  } else if (model == "t") {
    nu <- as.numeric(rugarch::coef(fit)["shape"])
    standardized_var <- rugarch::qdist(
      "std", p = var_levels, mu = 0, sigma = 1, shape = nu
    )
    t_quantile <- stats::qt(es_levels, df = nu)
    standardized_es <- stats::dt(t_quantile, df = nu) /
      (1 - es_levels) * (nu + t_quantile^2) / (nu - 1) *
      sqrt((nu - 2) / nu)
  } else {
    standardized <- skewed_t_standardized_risk(fit)
    standardized_var <- standardized$var
    standardized_es <- standardized$es
  }

  one_step <- rugarch::ugarchforecast(fit, n.ahead = 1)
  mu_next <- as.numeric(rugarch::fitted(one_step))
  sigma_next <- as.numeric(rugarch::sigma(one_step))
  c(
    VaR = mu_next + sigma_next * standardized_var[4],
    ES = mu_next + sigma_next * standardized_es[2]
  )
}

forecast_empirical <- function(target_index) {
  observations <- losses[(target_index - rolling_window):(target_index - 1L)]
  var <- unname(stats::quantile(observations, probs = 0.975))
  c(VaR = var, ES = mean(observations[observations >= var]))
}

forecast_one <- function(model, target_index) {
  if (model == "empirical") {
    forecast_empirical(target_index)
  } else {
    forecast_parametric(model, target_index)
  }
}

if (nzchar(validate_date)) {
  validation_date <- as.Date(validate_date)
  validation_index <- which(return_dates == validation_date)
  if (length(validation_index) != 1L || validation_index <= rolling_window) {
    stop("--validate-date must identify one forecastable trading date")
  }
  validation <- t(vapply(
    model_names,
    forecast_one,
    target_index = validation_index,
    FUN.VALUE = c(VaR = 0, ES = 0)
  ))
  print(validation)
  if (file.exists(reference_file)) {
    reference <- readRDS(reference_file)
    reference_date_index <- which(as.Date(reference$backtest$dates) == validation_date)
    if (length(reference_date_index) == 1L) {
      reference_forecast_index <- reference_date_index + rolling_window
      frozen <- cbind(
        VaR = reference$backtest$VaRout2b[, reference_forecast_index],
        ES = reference$backtest$ESout2[, reference_forecast_index]
      )
      rownames(frozen) <- model_names
      cat("\nDifference from the frozen NASDAQ.rds forecasts:\n")
      print(validation - frozen)
    }
  }
  quit(save = "no", status = 0)
}

save_rds_atomic <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  saveRDS(object, temporary)
  if (!file.rename(temporary, path)) stop("Could not atomically save ", path)
}

parallel_map <- function(indices, function_to_run) {
  if (.Platform$OS.type != "windows" && cores > 1L) {
    parallel::mclapply(
      indices,
      function_to_run,
      mc.cores = cores,
      mc.preschedule = FALSE
    )
  } else {
    lapply(indices, function_to_run)
  }
}

run_model <- function(model) {
  dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
  checkpoint_file <- file.path(checkpoint_dir, paste0(model, ".rds"))
  result <- matrix(
    NA_real_, nrow = length(target_indices), ncol = 2,
    dimnames = list(NULL, c("VaR", "ES"))
  )
  if (file.exists(checkpoint_file)) {
    saved <- readRDS(checkpoint_file)
    if (!identical(dim(saved), dim(result)) ||
        !identical(colnames(saved), colnames(result))) {
      stop("Invalid checkpoint: ", checkpoint_file)
    }
    result <- saved
  }

  pending <- which(!stats::complete.cases(result))
  if (!length(pending)) {
    message("Using complete checkpoint for ", model)
    return(result)
  }
  chunks <- split(pending, ceiling(seq_along(pending) / chunk_size))
  for (chunk_number in seq_along(chunks)) {
    rows <- chunks[[chunk_number]]
    values <- parallel_map(rows, function(row) {
      forecast_one(model, target_indices[row])
    })
    failures <- vapply(values, inherits, logical(1), what = "try-error")
    if (any(failures)) stop("At least one ", model, " forecast failed")
    result[rows, ] <- do.call(rbind, values)
    save_rds_atomic(result, checkpoint_file)
    message(
      model, ": ", sum(stats::complete.cases(result)), "/", nrow(result),
      " forecasts completed through ", max(forecast_dates[rows])
    )
  }
  result
}

forecasts <- setNames(lapply(model_names, run_model), model_names)
var_matrix <- do.call(rbind, lapply(forecasts, function(value) value[, "VaR"]))
es_matrix <- do.call(rbind, lapply(forecasts, function(value) value[, "ES"]))
rownames(var_matrix) <- rownames(es_matrix) <- model_names

figure_return_index <- which(return_dates == figure_start)
monitoring_return_index <- which(return_dates == monitoring_start)
if (length(figure_return_index) != 1L || length(monitoring_return_index) != 1L) {
  stop("Required Figure 14 or monitoring start date is absent")
}
figure_forecast_index <- figure_return_index - rolling_window
history_return_index <- monitoring_return_index - rolling_window
history_forecast_index <- history_return_index - rolling_window

figure_columns <- figure_forecast_index:ncol(es_matrix)
backtest_columns <- history_forecast_index:ncol(es_matrix)
backtest_loss_indices <- history_return_index:length(losses)
monitoring_loss_indices <- monitoring_return_index:length(losses)

bundle <- list(
  backtest = list(
    y = losses[backtest_loss_indices],
    dates = return_dates[monitoring_loss_indices],
    ESout2 = es_matrix[, backtest_columns, drop = FALSE],
    VaRout2b = var_matrix[, backtest_columns, drop = FALSE],
    nvec = es_levels,
    e_lim = c(-1, 5),
    metadata = list(
      source = "Yahoo Finance, NASDAQ Composite (^IXIC), Close",
      price_start = min(prices$Date),
      price_end = max(prices$Date),
      rolling_forecast_window = rolling_window,
      rolling_betting_window = rolling_window,
      monitoring_start = monitoring_start,
      calculation = "rolling AR(1)-GARCH(1,1) and empirical forecasts",
      price_file_md5 = unname(tools::md5sum(price_file)),
      R_version = R.version.string,
      rugarch_version = as.character(utils::packageVersion("rugarch")),
      sgt_version = as.character(utils::packageVersion("sgt"))
    )
  ),
  figure14 = data.frame(
    date = format(return_dates[figure_return_index:length(losses)]),
    negated_percentage_log_return = losses[figure_return_index:length(losses)],
    ES_0975_normal = es_matrix["normal", figure_columns],
    ES_0975_t = es_matrix["t", figure_columns],
    ES_0975_skewed_t = es_matrix["skewed_t", figure_columns],
    ES_0975_empirical = es_matrix["empirical", figure_columns],
    check.names = FALSE
  ),
  description = paste(
    "NASDAQ Composite inputs used in Section 5.3, reconstructed from",
    "daily closing prices by NASDAQ-forecasts.R"
  )
)

if (length(bundle$figure14$date) != 6539L ||
    length(bundle$backtest$dates) != 5282L ||
    length(bundle$backtest$y) != 5782L ||
    ncol(bundle$backtest$ESout2) != 5782L ||
    ncol(bundle$backtest$VaRout2b) != 5782L) {
  stop("Reconstructed arrays do not have the prespecified dimensions")
}

save_rds_atomic(bundle, output_file)
message("Saved regenerated forecast bundle to ", normalizePath(output_file))
message(
  "Figure 14 dates: ", min(as.Date(bundle$figure14$date)), " through ",
  max(as.Date(bundle$figure14$date)), "; resetting backtest dates: ",
  min(bundle$backtest$dates), " through ", max(bundle$backtest$dates)
)
