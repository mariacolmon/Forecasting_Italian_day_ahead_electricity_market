# ==============================================================================
# SCRIPT 07 - LEAR MODEL
# ===============================================================================


# ==============================================================================
# 0. LIBRARIES
# ===============================================================================


suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(arrow)
  library(glmnet)
  library(Matrix)
  library(zoo)
})


# ==============================================================================
# 1. CONFIGURATION
# ===============================================================================

# Empty string means: use the standard project folders under data/.
# Non-empty string means: write outputs under experiments/<EXPERIMENT_ID>/.
EXPERIMENT_ID <- NULL

RUN_FINAL_STAGE <- TRUE

RUN_FAST_VALIDATION <- FALSE
VALIDATION_ZONES_FAST <- c("NORD", "CSUD", "SICI")
VALIDATION_TARGETS_FAST <- c("price", "purchases", "sales")
MAX_VALIDATION_DAYS_PER_SERIES <- NA_integer_

RUN_FAST_FINAL <- FALSE
FINAL_ZONES_FAST <- c("NORD", "CSUD", "SICI")
FINAL_TARGETS_FAST <- c("price", "purchases", "sales")
MAX_FINAL_DAYS_PER_SERIES <- NA_integer_

RUN_PARALLEL <- FALSE
N_WORKERS <- max(1L, parallel::detectCores() - 1L)

ENABLE_CHECKPOINTS <- TRUE
RESUME_FROM_CHECKPOINT <- TRUE
FORCE_RERUN_EXISTING_CHECKPOINTS <- FALSE

INNER_VALIDATION_DAYS <- 30L
LEAR_ALPHA <- 1

MIN_TOTAL_OBS <- 60L
MIN_INNER_TRAIN_OBS <- 30L
MIN_INNER_VALIDATION_OBS <- 7L

MAIN_RMAE_BENCHMARK <- "naive_week_before"
MODEL_NAME <- "lear"

LEAR_STRATEGIES <- tibble::tribble(
  ~strategy_id,               ~window_months, ~window_type, ~recalibration_frequency, ~lambda_method,
  "lear_6m_monthly_holdout",    6L,            "rolling",    "monthly",               "holdout",
  "lear_12m_monthly_holdout",  12L,            "rolling",    "monthly",               "holdout",
  "lear_24m_monthly_holdout",  24L,            "rolling",    "monthly",               "holdout",
  "lear_exp_monthly_holdout",  NA_integer_,    "expanding",  "monthly",               "holdout"
)

TARGETS <- c("price", "purchases", "sales")

PHYSICAL_ZONES <- c("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")

# Cross-zone regressors. These are lagged variables from all physical zones,
# attached to every zone-specific LEAR equation. They capture regional
# interactions without using contemporaneous information from the delivery day.
INCLUDE_CROSS_ZONE_LAGS <- TRUE
CROSS_ZONE_VARIABLES <- c("price", "purchases", "sales")
CROSS_ZONE_LAGS_HOURS <- c(24L, 168L)

LAG_VARIABLES <- c(
  "price", "purchases", "sales", "hhi", "rsi", "mti", "pun",
  "purchases_italy", "sales_italy", "unsold_italy",
  "purchases_external_total", "sales_external_total",
  "purchases_external_n_active_areas", "sales_external_n_active_areas"
)

LAGS_HOURS <- c(24L, 48L, 168L)

REQUIRED_PREDICTION_COLUMNS <- c(
  "model", "target", "zone", "split", "forecast_date",
  "delivery_datetime_model", "delivery_date", "hour", "horizon",
  "y_true", "y_pred", "strategy_id", "window_months", "window_type",
  "recalibration_frequency", "lambda_method", "recalibration_date"
)


# ==============================================================================
# 2. PATHS
# ===============================================================================

USE_EXPERIMENT <- !is.null(EXPERIMENT_ID) && nzchar(EXPERIMENT_ID)
RUN_ROOT <- if (USE_EXPERIMENT) file.path("experiments", EXPERIMENT_ID) else "."

INPUT_PANEL_REL <- "data/processed/gme_model_panel_weather_hourly.rds"
INPUT_EVAL_REL  <- "data/evaluation/eval_index_hourly.rds"
INPUT_NAIVE_WEEK_REL <- "data/predictions/pred_naive_week_before.parquet"

PRED_DIR <- file.path(RUN_ROOT, "data", "predictions")
LOG_DIR  <- file.path(RUN_ROOT, "data", "model_logs", "lear")

CHECKPOINT_SCOPE <- paste(
  ifelse(RUN_FAST_VALIDATION,
         paste0("validation_fast_", MAX_VALIDATION_DAYS_PER_SERIES, "d"),
         "validation_full"),
  ifelse(RUN_FINAL_STAGE,
         ifelse(RUN_FAST_FINAL,
                paste0("final_fast_", MAX_FINAL_DAYS_PER_SERIES, "d"),
                "final_full"),
         "no_final"),
  sep = "__"
)

CHECKPOINT_DIR <- file.path(LOG_DIR, "checkpoints", CHECKPOINT_SCOPE)

if (!dir.exists(PRED_DIR)) dir.create(PRED_DIR, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(LOG_DIR)) dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(CHECKPOINT_DIR)) dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

resolve_input_path <- function(relative_path) {
  experiment_path <- file.path(RUN_ROOT, relative_path)
  if (USE_EXPERIMENT && file.exists(experiment_path)) {
    return(experiment_path)
  }
  relative_path
}

INPUT_PANEL <- resolve_input_path(INPUT_PANEL_REL)
INPUT_EVAL  <- resolve_input_path(INPUT_EVAL_REL)
INPUT_NAIVE_WEEK <- resolve_input_path(INPUT_NAIVE_WEEK_REL)


# ==============================================================================
# 3. GENERAL HELPERS
# ===============================================================================

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || all(is.na(x))) y else x
}

assert_file_exists <- function(path) {
  if (!file.exists(path)) {
    stop("Required file not found: ", path, call. = FALSE)
  }
  invisible(TRUE)
}

assert_required_columns <- function(df, required_cols, object_name = "data") {
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      object_name, " is missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

safe_as_date <- function(x) {
  if (inherits(x, "Date")) return(x)
  as.Date(x)
}

safe_as_posixct_utc <- function(x) {
  as.POSIXct(x, tz = "UTC")
}

numeric_median_safe <- function(x) {
  x <- as.numeric(x)
  x[!is.finite(x)] <- NA_real_
  med <- suppressWarnings(stats::median(x, na.rm = TRUE))
  if (!is.finite(med)) med <- 0
  med
}

limit_eval_days_per_series <- function(eval_df, max_days_per_series) {
  if (is.null(max_days_per_series) || is.na(max_days_per_series)) {
    return(eval_df)
  }

  eval_df %>%
    arrange(target, zone, split, delivery_date, hour) %>%
    group_by(target, zone, split) %>%
    mutate(.day_rank = dense_rank(delivery_date)) %>%
    ungroup() %>%
    filter(.day_rank <= max_days_per_series) %>%
    select(-.day_rank)
}

period_id_from_delivery_date <- function(delivery_date, frequency) {
  frequency <- tolower(frequency)

  if (frequency == "monthly") {
    return(as.character(floor_date(delivery_date, unit = "month")))
  }

  if (frequency == "weekly") {
    return(as.character(floor_date(delivery_date, unit = "week", week_start = 1)))
  }

  if (frequency == "daily") {
    return(as.character(delivery_date))
  }

  stop("Unsupported recalibration_frequency: ", frequency, call. = FALSE)
}

map_tasks <- function(task_list, fun) {
  if (RUN_PARALLEL) {
    if (requireNamespace("future", quietly = TRUE) && requireNamespace("furrr", quietly = TRUE)) {
      old_plan <- future::plan()
      on.exit(future::plan(old_plan), add = TRUE)
      future::plan(future::multisession, workers = N_WORKERS)
      return(furrr::future_map(task_list, fun, .options = furrr::furrr_options(seed = TRUE)))
    }

    warning(
      "RUN_PARALLEL = TRUE, but packages 'future' and/or 'furrr' are not installed. ",
      "Falling back to sequential execution."
    )
  }

  purrr::map(task_list, fun)
}

# Atomic writes: write to a temporary file first, then rename. This reduces the
# chance of leaving a corrupted final file if the machine shuts down during write.
safe_write_parquet <- function(x, path) {
  tmp <- paste0(path, ".tmp_", Sys.getpid(), "_", format(Sys.time(), "%Y%m%d%H%M%S"))
  arrow::write_parquet(x, tmp)
  if (file.exists(path)) file.remove(path)
  ok <- file.rename(tmp, path)
  if (!ok) stop("Could not move temporary parquet file to: ", path, call. = FALSE)
  invisible(path)
}

safe_write_csv <- function(x, path) {
  tmp <- paste0(path, ".tmp_", Sys.getpid(), "_", format(Sys.time(), "%Y%m%d%H%M%S"))
  readr::write_csv(x, tmp)
  if (file.exists(path)) file.remove(path)
  ok <- file.rename(tmp, path)
  if (!ok) stop("Could not move temporary csv file to: ", path, call. = FALSE)
  invisible(path)
}

safe_write_rds <- function(x, path) {
  tmp <- paste0(path, ".tmp_", Sys.getpid(), "_", format(Sys.time(), "%Y%m%d%H%M%S"))
  saveRDS(x, tmp)
  if (file.exists(path)) file.remove(path)
  ok <- file.rename(tmp, path)
  if (!ok) stop("Could not move temporary rds file to: ", path, call. = FALSE)
  invisible(path)
}

sanitize_file_id <- function(x) {
  x <- as.character(x)
  x[is.na(x) | !nzchar(x)] <- "NA"
  x %>%
    str_replace_all("[^A-Za-z0-9_-]+", "_") %>%
    str_replace_all("_+", "_")
}

strategy_checkpoint_id <- function(stage_name, strategy_row) {
  target_part <- if ("target" %in% names(strategy_row) && !is.na(strategy_row$target[[1]])) {
    as.character(strategy_row$target[[1]])
  } else {
    "all_targets"
  }

  paste(
    sanitize_file_id(stage_name),
    sanitize_file_id(target_part),
    sanitize_file_id(strategy_row$strategy_id[[1]]),
    sep = "__"
  )
}

strategy_checkpoint_paths <- function(stage_name, strategy_row) {
  id <- strategy_checkpoint_id(stage_name, strategy_row)
  list(
    predictions = file.path(CHECKPOINT_DIR, paste0(id, "__predictions.parquet")),
    fit_log = file.path(CHECKPOINT_DIR, paste0(id, "__fit_log.parquet")),
    done = file.path(CHECKPOINT_DIR, paste0(id, "__DONE.txt"))
  )
}

strategy_checkpoint_exists <- function(stage_name, strategy_row) {
  paths <- strategy_checkpoint_paths(stage_name, strategy_row)
  file.exists(paths$predictions) && file.exists(paths$fit_log) && file.exists(paths$done)
}

load_strategy_checkpoint <- function(stage_name, strategy_row) {
  paths <- strategy_checkpoint_paths(stage_name, strategy_row)
  list(
    predictions = arrow::read_parquet(paths$predictions),
    fit_log = arrow::read_parquet(paths$fit_log)
  )
}

save_strategy_checkpoint <- function(result, stage_name, strategy_row) {
  paths <- strategy_checkpoint_paths(stage_name, strategy_row)
  safe_write_parquet(result$predictions, paths$predictions)
  safe_write_parquet(result$fit_log, paths$fit_log)

  target_for_manifest <- if ("target" %in% names(strategy_row) && !is.na(strategy_row$target[[1]])) {
    as.character(strategy_row$target[[1]])
  } else {
    "all_targets"
  }

  writeLines(
    c(
      paste0("stage=", stage_name),
      paste0("strategy_id=", strategy_row$strategy_id[[1]]),
      paste0("target=", target_for_manifest),
      paste0("saved_at=", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
      paste0("n_predictions=", nrow(result$predictions)),
      paste0("n_fits=", nrow(result$fit_log)),
      paste0("n_failed=", sum(result$fit_log$status != "ok" | is.na(result$fit_log$status)))
    ),
    con = paths$done
  )

  invisible(paths)
}


# ==============================================================================
# 4. FEATURE ENGINEERING
# ===============================================================================

make_cross_zone_lag_features <- function(panel, zones = PHYSICAL_ZONES) {
  # Create variables such as price_NORD_lag24 or sales_SICI_lag168.
  # These variables are common across the target zone at a given timestamp:
  # for example, the NORD equation can use lagged prices and sales from
  # CNOR, CSUD, SUD, CALA, SICI and SARD.
  #
  # Methodological note: only lagged values are created. No contemporaneous
  # cross-zone market variables are used, preserving the day-ahead information
  # set and avoiding leakage.

  cross_vars <- intersect(CROSS_ZONE_VARIABLES, names(panel))

  if (!isTRUE(INCLUDE_CROSS_ZONE_LAGS) || length(cross_vars) == 0) {
    return(tibble(datetime_model = unique(panel$datetime_model)))
  }

  cross_base <- panel %>%
    mutate(
      datetime_model = safe_as_posixct_utc(datetime_model),
      zone = as.character(zone)
    ) %>%
    filter(zone %in% zones) %>%
    select(datetime_model, zone, all_of(cross_vars)) %>%
    pivot_longer(
      cols = all_of(cross_vars),
      names_to = "variable",
      values_to = "value"
    ) %>%
    group_by(datetime_model, variable, zone) %>%
    summarise(value = mean(as.numeric(value), na.rm = TRUE), .groups = "drop") %>%
    mutate(value = if_else(is.nan(value), NA_real_, value)) %>%
    pivot_wider(
      names_from = c(variable, zone),
      values_from = value,
      names_glue = "{variable}_{zone}"
    ) %>%
    arrange(datetime_model)

  raw_cross_cols <- setdiff(names(cross_base), "datetime_model")

  for (col in raw_cross_cols) {
    for (lag_h in CROSS_ZONE_LAGS_HOURS) {
      new_col <- paste0(col, "_lag", lag_h)
      cross_base <- cross_base %>%
        mutate(!!new_col := dplyr::lag(.data[[col]], n = lag_h))
    }
  }

  cross_lag_cols <- grep(
    paste0(
      "^(", paste(CROSS_ZONE_VARIABLES, collapse = "|"), ")_.*_lag(",
      paste(CROSS_ZONE_LAGS_HOURS, collapse = "|"), ")$"
    ),
    names(cross_base),
    value = TRUE
  )

  cross_base %>%
    select(datetime_model, all_of(cross_lag_cols))
}

prepare_lear_panel <- function(panel) {
  required <- c("datetime_model", "date", "hour", "zone")
  assert_required_columns(panel, required, "panel")

  lag_vars <- intersect(LAG_VARIABLES, names(panel))

  panel_base <- panel %>%
    mutate(
      datetime_model = safe_as_posixct_utc(datetime_model),
      date = safe_as_date(date),
      zone = as.character(zone),
      hour = as.integer(hour)
    )

  cross_zone_lags <- make_cross_zone_lag_features(panel_base)

  out <- panel_base %>%
    mutate(
      datetime_model = safe_as_posixct_utc(datetime_model),
      date = safe_as_date(date),
      zone = as.character(zone),
      hour = as.integer(hour),
      weekday_num = lubridate::wday(date, week_start = 1),
      weekday_f = factor(
        weekday_num,
        levels = 1:7,
        labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
      ),
      month_f = factor(lubridate::month(date), levels = 1:12),
      is_weekend_f = factor(
        if_else(weekday_num %in% c(6L, 7L), "weekend", "weekday"),
        levels = c("weekday", "weekend")
      )
    ) %>%
    arrange(zone, datetime_model) %>%
    group_by(zone)

  for (lag_var in lag_vars) {
    for (lag_h in LAGS_HOURS) {
      new_col <- paste0(lag_var, "_lag", lag_h)
      out <- out %>%
        mutate(!!new_col := dplyr::lag(.data[[lag_var]], n = lag_h))
    }
  }

  out <- out %>%
    ungroup() %>%
    select(-weekday_num) %>%
    left_join(cross_zone_lags, by = "datetime_model")

  if ("holiday" %in% names(out)) {
    out <- out %>%
      mutate(holiday_f = factor(if_else(as.logical(holiday), "holiday", "non_holiday")))
  }

  out
}

make_x_cols_key <- function(target, zone) {
  paste(as.character(target), as.character(zone), sep = "__")
}

get_lear_x_cols <- function(df, target, zone = NULL) {
  own_target_lags <- paste0(target, "_lag", c(24L, 48L, 168L))

  other_market_lags <- c(
    "price_lag24", "purchases_lag24", "sales_lag24",
    "hhi_lag24", "rsi_lag24", "mti_lag24"
  )

  national_common_lags <- c(
    "pun_lag24",
    "purchases_italy_lag24",
    "sales_italy_lag24",
    "unsold_italy_lag24",
    "purchases_external_total_lag24",
    "sales_external_total_lag24"
  )

  weather_cols <- c(
    "temperature_2m", "wind_speed_100m", "shortwave_radiation"
  )

  cross_zone_pattern <- paste0(
    "^(", paste(CROSS_ZONE_VARIABLES, collapse = "|"), ")_.*_lag(",
    paste(CROSS_ZONE_LAGS_HOURS, collapse = "|"), ")$"
  )

  cross_zone_lags <- grep(
    cross_zone_pattern,
    names(df),
    value = TRUE
  )

  # Cross-zone features should represent other physical bidding zones only.
  # The local zone is already represented by local lag variables such as
  # price_lag24, purchases_lag24 and sales_lag24. For example, in the NORD
  # equation we exclude price_NORD_lag24, purchases_NORD_lag168, etc.
  if (!is.null(zone) && !is.na(zone)) {
    own_zone_pattern <- paste0("_", as.character(zone), "_lag")
    cross_zone_lags <- cross_zone_lags[!str_detect(cross_zone_lags, fixed(own_zone_pattern))]
  }

  calendar_cols <- c("weekday_f", "month_f", "is_weekend_f")

  if ("holiday_f" %in% names(df)) {
    calendar_cols <- c(calendar_cols, "holiday_f")
  }

  candidate_cols <- unique(c(
    own_target_lags,
    other_market_lags,
    national_common_lags,
    cross_zone_lags,
    weather_cols,
    calendar_cols
  ))

  intersect(candidate_cols, names(df))
}

create_feature_audit <- function(panel_lear, targets, zones = PHYSICAL_ZONES) {
  tidyr::crossing(target = targets, zone = zones) %>%
    mutate(x_cols = map2(target, zone, ~ get_lear_x_cols(panel_lear, .x, .y))) %>%
    unnest_longer(x_cols, values_to = "x_col") %>%
    mutate(
      available = TRUE,
      class = map_chr(x_col, ~ paste(class(panel_lear[[.x]]), collapse = "/"))
    )
}



# ==============================================================================
# 5. PREPROCESSING FOR GLMNET
# ===============================================================================

identify_column_types <- function(train_df, x_cols) {
  x_cols <- intersect(x_cols, names(train_df))

  factor_cols <- x_cols[map_lgl(train_df[x_cols], ~ is.factor(.x) || is.character(.x) || is.logical(.x))]
  numeric_cols <- setdiff(x_cols, factor_cols)

  list(
    numeric_cols = numeric_cols,
    factor_cols = factor_cols
  )
}

format_predictor_frame <- function(df, preprocessor, fitting = FALSE) {
  x_cols <- preprocessor$x_cols

  for (col in setdiff(x_cols, names(df))) {
    df[[col]] <- NA
  }

  x_df <- df %>% select(all_of(x_cols))

  for (col in preprocessor$numeric_cols) {
    x_df[[col]] <- as.numeric(x_df[[col]])
    x_df[[col]][!is.finite(x_df[[col]])] <- NA_real_
    x_df[[col]][is.na(x_df[[col]])] <- preprocessor$numeric_medians[[col]]
  }

  for (col in preprocessor$factor_cols) {
    allowed_levels <- preprocessor$factor_levels[[col]]
    raw_values <- as.character(x_df[[col]])
    raw_values[is.na(raw_values) | !(raw_values %in% allowed_levels)] <- "MISSING"
    x_df[[col]] <- factor(raw_values, levels = allowed_levels)
  }

  x_df
}

make_sparse_model_matrix <- function(x_df) {
  if (ncol(x_df) == 0) {
    return(Matrix::Matrix(0, nrow = nrow(x_df), ncol = 0, sparse = TRUE))
  }

  mm <- Matrix::sparse.model.matrix(~ ., data = x_df)

  intercept_col <- which(colnames(mm) == "(Intercept)")
  if (length(intercept_col) > 0) {
    mm <- mm[, -intercept_col, drop = FALSE]
  }

  mm
}

create_lear_preprocessor <- function(train_df, x_cols) {
  x_cols <- intersect(x_cols, names(train_df))

  if (length(x_cols) == 0) {
    stop("No predictor columns are available for LEAR.", call. = FALSE)
  }

  col_types <- identify_column_types(train_df, x_cols)

  numeric_medians <- map(
    col_types$numeric_cols,
    ~ numeric_median_safe(train_df[[.x]])
  )
  names(numeric_medians) <- col_types$numeric_cols

  factor_levels <- map(
    col_types$factor_cols,
    function(col) {
      vals <- as.character(train_df[[col]])
      vals <- vals[!is.na(vals)]
      lvls <- unique(vals)
      lvls <- sort(lvls)
      lvls <- unique(c(lvls, "MISSING"))
      if (length(lvls) < 2) lvls <- unique(c(lvls, "OTHER"))
      lvls
    }
  )
  names(factor_levels) <- col_types$factor_cols

  preprocessor <- list(
    x_cols = x_cols,
    numeric_cols = col_types$numeric_cols,
    factor_cols = col_types$factor_cols,
    numeric_medians = numeric_medians,
    factor_levels = factor_levels,
    model_matrix_cols = character(0)
  )

  x_train <- format_predictor_frame(train_df, preprocessor, fitting = TRUE)
  mm_train <- make_sparse_model_matrix(x_train)

  if (ncol(mm_train) == 0) {
    stop("Model matrix has zero columns after preprocessing.", call. = FALSE)
  }

  preprocessor$model_matrix_cols <- colnames(mm_train)
  preprocessor
}

apply_lear_preprocessor <- function(df, preprocessor) {
  x_df <- format_predictor_frame(df, preprocessor, fitting = FALSE)
  mm <- make_sparse_model_matrix(x_df)

  expected_cols <- preprocessor$model_matrix_cols
  current_cols <- colnames(mm)

  missing_cols <- setdiff(expected_cols, current_cols)
  if (length(missing_cols) > 0) {
    zero_mat <- Matrix::sparseMatrix(
      i = integer(0),
      j = integer(0),
      dims = c(nrow(mm), length(missing_cols)),
      dimnames = list(NULL, missing_cols)
    )
    mm <- cbind(mm, zero_mat)
  }

  extra_cols <- setdiff(colnames(mm), expected_cols)
  if (length(extra_cols) > 0) {
    mm <- mm[, setdiff(colnames(mm), extra_cols), drop = FALSE]
  }

  mm <- mm[, expected_cols, drop = FALSE]
  mm
}


# ==============================================================================
# 6. LEAR FITTING WITH TEMPORAL HOLDOUT LAMBDA SELECTION
# ===============================================================================

fit_lear_with_inner_lambda <- function(
    train_df,
    x_cols,
    inner_validation_days = INNER_VALIDATION_DAYS,
    alpha = LEAR_ALPHA
) {
  t_total_start <- Sys.time()

  empty_result <- list(
    model = NULL,
    selected_lambda = NA_real_,
    inner_mae = NA_real_,
    n_nonzero = NA_integer_,
    n_train = nrow(train_df),
    n_inner_train = NA_integer_,
    n_inner_validation = NA_integer_,
    n_features = NA_integer_,
    model_matrix_cols = character(0),
    preprocessor = NULL,
    fit_seconds = NA_real_,
    lambda_selection_seconds = NA_real_,
    status = "failed",
    error_message = NA_character_
  )

  tryCatch({
    if (!"y" %in% names(train_df)) {
      stop("train_df must contain a column named 'y'.")
    }

    train_df <- train_df %>%
      mutate(
        date = safe_as_date(date),
        y = as.numeric(y)
      ) %>%
      filter(is.finite(y)) %>%
      arrange(date)

    if (nrow(train_df) < MIN_TOTAL_OBS) {
      stop("Not enough total observations: ", nrow(train_df), ".")
    }

    last_train_date <- max(train_df$date, na.rm = TRUE)
    inner_start_date <- last_train_date - lubridate::days(inner_validation_days) + lubridate::days(1)

    inner_train_df <- train_df %>% filter(date < inner_start_date)
    inner_valid_df <- train_df %>% filter(date >= inner_start_date)

    if (nrow(inner_train_df) < MIN_INNER_TRAIN_OBS) {
      stop("Not enough inner-training observations: ", nrow(inner_train_df), ".")
    }

    if (nrow(inner_valid_df) < MIN_INNER_VALIDATION_OBS) {
      stop("Not enough inner-validation observations: ", nrow(inner_valid_df), ".")
    }

    t_lambda_start <- Sys.time()

    inner_preprocessor <- create_lear_preprocessor(inner_train_df, x_cols)
    x_inner_train <- apply_lear_preprocessor(inner_train_df, inner_preprocessor)
    x_inner_valid <- apply_lear_preprocessor(inner_valid_df, inner_preprocessor)

    y_inner_train <- inner_train_df$y
    y_inner_valid <- inner_valid_df$y

    fit_path <- glmnet::glmnet(
      x = x_inner_train,
      y = y_inner_train,
      alpha = alpha,
      family = "gaussian",
      standardize = TRUE,
      intercept = TRUE
    )

    lambda_grid <- fit_path$lambda
    pred_inner <- predict(fit_path, newx = x_inner_valid, s = lambda_grid)

    inner_mae_by_lambda <- colMeans(abs(sweep(pred_inner, 1, y_inner_valid, FUN = "-")), na.rm = TRUE)

    if (all(!is.finite(inner_mae_by_lambda))) {
      stop("All inner validation MAEs are non-finite.")
    }

    best_idx <- which.min(inner_mae_by_lambda)
    selected_lambda <- as.numeric(lambda_grid[best_idx])
    selected_inner_mae <- as.numeric(inner_mae_by_lambda[best_idx])

    lambda_selection_seconds <- as.numeric(difftime(Sys.time(), t_lambda_start, units = "secs"))

    t_fit_start <- Sys.time()

    final_preprocessor <- create_lear_preprocessor(train_df, x_cols)
    x_train_full <- apply_lear_preprocessor(train_df, final_preprocessor)
    y_train_full <- train_df$y

    final_model <- glmnet::glmnet(
      x = x_train_full,
      y = y_train_full,
      alpha = alpha,
      family = "gaussian",
      lambda = selected_lambda,
      standardize = TRUE,
      intercept = TRUE
    )

    fit_seconds <- as.numeric(difftime(Sys.time(), t_fit_start, units = "secs"))

    coefs <- as.matrix(stats::coef(final_model, s = selected_lambda))
    n_nonzero <- sum(abs(coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]) > 0)

    list(
      model = final_model,
      selected_lambda = selected_lambda,
      inner_mae = selected_inner_mae,
      n_nonzero = as.integer(n_nonzero),
      n_train = nrow(train_df),
      n_inner_train = nrow(inner_train_df),
      n_inner_validation = nrow(inner_valid_df),
      n_features = ncol(x_train_full),
      model_matrix_cols = colnames(x_train_full),
      preprocessor = final_preprocessor,
      fit_seconds = fit_seconds,
      lambda_selection_seconds = lambda_selection_seconds,
      status = "ok",
      error_message = NA_character_,
      total_fit_function_seconds = as.numeric(difftime(Sys.time(), t_total_start, units = "secs"))
    )
  }, error = function(e) {
    empty_result$error_message <- conditionMessage(e)
    empty_result$total_fit_function_seconds <- as.numeric(difftime(Sys.time(), t_total_start, units = "secs"))
    empty_result
  })
}


# ==============================================================================
# 7. TRAINING WINDOW AND FORECASTING TASKS
# ===============================================================================

build_training_window <- function(
    panel_lear,
    target,
    zone,
    hour,
    recalibration_date,
    window_months,
    window_type
) {
  recalibration_date <- safe_as_date(recalibration_date)
  window_type <- tolower(window_type)

  df <- panel_lear %>%
    filter(
      .data$zone == !!zone,
      .data$hour == !!hour,
      .data$date <= !!recalibration_date
    )

  if (window_type == "rolling") {
    if (is.na(window_months)) {
      stop("window_months cannot be NA when window_type = 'rolling'.")
    }
    start_date <- recalibration_date %m-% months(as.integer(window_months)) + days(1)
    df <- df %>% filter(.data$date >= !!start_date)
  } else if (window_type == "expanding") {
    # Keep all observations up to the recalibration date.
  } else {
    stop("Unsupported window_type: ", window_type)
  }

  df %>%
    mutate(y = as.numeric(.data[[target]])) %>%
    arrange(date)
}

build_future_frame <- function(panel_lear, eval_block, x_cols) {
  feature_cols <- unique(c("date", "zone", "hour", x_cols))

  future_features <- panel_lear %>%
    select(any_of(feature_cols)) %>%
    distinct(date, zone, hour, .keep_all = TRUE)

  eval_block %>%
    left_join(
      future_features,
      by = c(
        "delivery_date" = "date",
        "zone" = "zone",
        "hour" = "hour"
      )
    )
}

make_failure_predictions <- function(eval_block, task, error_message) {
  eval_block %>%
    transmute(
      model = MODEL_NAME,
      target,
      zone,
      split,
      forecast_date,
      delivery_datetime_model,
      delivery_date,
      hour,
      horizon,
      y_true,
      y_pred = NA_real_,
      strategy_id = task$strategy_id,
      window_months = as.integer(task$window_months),
      window_type = task$window_type,
      recalibration_frequency = task$recalibration_frequency,
      lambda_method = task$lambda_method,
      recalibration_date = safe_as_date(task$recalibration_date)
    )
}

make_fit_log_row <- function(
    task,
    eval_block,
    fit_result,
    prediction_seconds,
    total_seconds,
    status = NULL,
    error_message = NULL
) {
  status <- status %||% fit_result$status %||% "failed"
  error_message <- error_message %||% fit_result$error_message %||% NA_character_

  tibble(
    target = task$target,
    zone = task$zone,
    hour = as.integer(task$hour),
    split = task$split,
    strategy_id = task$strategy_id,
    forecast_date = min(eval_block$forecast_date, na.rm = TRUE),
    delivery_date = min(eval_block$delivery_date, na.rm = TRUE),
    block_delivery_end = max(eval_block$delivery_date, na.rm = TRUE),
    recalibration_date = safe_as_date(task$recalibration_date),
    window_months = as.integer(task$window_months),
    window_type = task$window_type,
    recalibration_frequency = task$recalibration_frequency,
    lambda_method = task$lambda_method,
    selected_lambda = as.numeric(fit_result$selected_lambda %||% NA_real_),
    inner_mae = as.numeric(fit_result$inner_mae %||% NA_real_),
    n_train = as.integer(fit_result$n_train %||% NA_integer_),
    n_inner_train = as.integer(fit_result$n_inner_train %||% NA_integer_),
    n_inner_validation = as.integer(fit_result$n_inner_validation %||% NA_integer_),
    n_features = as.integer(fit_result$n_features %||% NA_integer_),
    n_nonzero = as.integer(fit_result$n_nonzero %||% NA_integer_),
    fit_seconds = as.numeric(fit_result$fit_seconds %||% NA_real_),
    lambda_selection_seconds = as.numeric(fit_result$lambda_selection_seconds %||% NA_real_),
    prediction_seconds = as.numeric(prediction_seconds %||% NA_real_),
    total_seconds = as.numeric(total_seconds %||% NA_real_),
    status = as.character(status),
    error_message = as.character(error_message)
  )
}

forecast_lear_task <- function(task, eval_with_blocks, panel_lear, x_cols_by_target_zone) {
  task <- as.list(task)
  task_start <- Sys.time()

  eval_block <- eval_with_blocks %>%
    filter(
      .data$target == !!task$target,
      .data$zone == !!task$zone,
      .data$hour == !!as.integer(task$hour),
      .data$split == !!task$split,
      .data$period_id == !!task$period_id
    ) %>%
    arrange(delivery_datetime_model)

  if (nrow(eval_block) == 0) {
    fit_result <- list(status = "failed", error_message = "Empty evaluation block.")
    return(list(
      predictions = tibble(),
      fit_log = make_fit_log_row(task, tibble(forecast_date = as.Date(NA), delivery_date = as.Date(NA)), fit_result, NA_real_, 0)
    ))
  }

  tryCatch({
    x_cols <- x_cols_by_target_zone[[make_x_cols_key(task$target, task$zone)]]

    train_df <- build_training_window(
      panel_lear = panel_lear,
      target = task$target,
      zone = task$zone,
      hour = as.integer(task$hour),
      recalibration_date = task$recalibration_date,
      window_months = as.integer(task$window_months),
      window_type = task$window_type
    )

    fit_result <- fit_lear_with_inner_lambda(
      train_df = train_df,
      x_cols = x_cols,
      inner_validation_days = INNER_VALIDATION_DAYS,
      alpha = LEAR_ALPHA
    )

    prediction_seconds <- NA_real_

    if (identical(fit_result$status, "ok")) {
      t_pred_start <- Sys.time()
      future_df <- build_future_frame(panel_lear, eval_block, x_cols)
      x_future <- apply_lear_preprocessor(future_df, fit_result$preprocessor)
      y_pred <- as.numeric(predict(fit_result$model, newx = x_future, s = fit_result$selected_lambda))
      prediction_seconds <- as.numeric(difftime(Sys.time(), t_pred_start, units = "secs"))

      predictions <- eval_block %>%
        mutate(y_pred = y_pred) %>%
        transmute(
          model = MODEL_NAME,
          target,
          zone,
          split,
          forecast_date,
          delivery_datetime_model,
          delivery_date,
          hour,
          horizon,
          y_true,
          y_pred,
          strategy_id = task$strategy_id,
          window_months = as.integer(task$window_months),
          window_type = task$window_type,
          recalibration_frequency = task$recalibration_frequency,
          lambda_method = task$lambda_method,
          recalibration_date = safe_as_date(task$recalibration_date)
        )
    } else {
      predictions <- make_failure_predictions(eval_block, task, fit_result$error_message)
    }

    total_seconds <- as.numeric(difftime(Sys.time(), task_start, units = "secs"))

    list(
      predictions = predictions,
      fit_log = make_fit_log_row(task, eval_block, fit_result, prediction_seconds, total_seconds)
    )
  }, error = function(e) {
    fit_result <- list(
      status = "failed",
      error_message = conditionMessage(e),
      selected_lambda = NA_real_,
      inner_mae = NA_real_,
      n_train = NA_integer_,
      n_inner_train = NA_integer_,
      n_inner_validation = NA_integer_,
      n_features = NA_integer_,
      n_nonzero = NA_integer_,
      fit_seconds = NA_real_,
      lambda_selection_seconds = NA_real_
    )

    total_seconds <- as.numeric(difftime(Sys.time(), task_start, units = "secs"))

    list(
      predictions = make_failure_predictions(eval_block, task, conditionMessage(e)),
      fit_log = make_fit_log_row(
        task,
        eval_block,
        fit_result,
        prediction_seconds = NA_real_,
        total_seconds = total_seconds,
        status = "failed",
        error_message = conditionMessage(e)
      )
    )
  })
}

prepare_eval_for_strategy <- function(eval_df, strategy_row) {
  freq <- strategy_row$recalibration_frequency[[1]]

  eval_df %>%
    mutate(
      period_id = period_id_from_delivery_date(delivery_date, freq)
    ) %>%
    group_by(split, period_id) %>%
    mutate(recalibration_date = min(forecast_date, na.rm = TRUE)) %>%
    ungroup()
}

run_lear_for_one_strategy <- function(eval_df, strategy_row, panel_lear, x_cols_by_target_zone) {
  strategy_row <- strategy_row %>% slice(1)
  strategy_id <- strategy_row$strategy_id[[1]]

  message("\n--- Running LEAR strategy: ", strategy_id, " ---")

  eval_with_blocks <- prepare_eval_for_strategy(eval_df, strategy_row)

  tasks <- eval_with_blocks %>%
    distinct(target, zone, hour, split, period_id, recalibration_date) %>%
    mutate(
      strategy_id = strategy_id,
      window_months = as.integer(strategy_row$window_months[[1]]),
      window_type = strategy_row$window_type[[1]],
      recalibration_frequency = strategy_row$recalibration_frequency[[1]],
      lambda_method = strategy_row$lambda_method[[1]]
    ) %>%
    arrange(split, target, zone, hour, recalibration_date)

  message("Number of recalibration tasks: ", nrow(tasks))

  task_list <- split(tasks, seq_len(nrow(tasks)))
  results <- map_tasks(
    task_list,
    ~ forecast_lear_task(.x, eval_with_blocks, panel_lear, x_cols_by_target_zone)
  )

  predictions <- map_dfr(results, "predictions")
  fit_log <- map_dfr(results, "fit_log")

  list(
    predictions = predictions,
    fit_log = fit_log
  )
}

run_lear_strategies <- function(
    eval_df,
    strategies_tbl,
    panel_lear,
    x_cols_by_target_zone,
    stage_name = "unknown"
) {
  all_predictions <- list()
  all_logs <- list()

  for (i in seq_len(nrow(strategies_tbl))) {
    strategy_row <- strategies_tbl %>% slice(i)
    checkpoint_id <- strategy_checkpoint_id(stage_name, strategy_row)

    if ("target" %in% names(strategy_row) && !is.na(strategy_row$target[[1]])) {
      eval_strategy <- eval_df %>% filter(target == strategy_row$target[[1]])
    } else {
      eval_strategy <- eval_df
    }

    if (nrow(eval_strategy) == 0) {
      warning("No evaluation rows for strategy row ", i, ". Skipping.")
      next
    }

    if (
      isTRUE(ENABLE_CHECKPOINTS) &&
      isTRUE(RESUME_FROM_CHECKPOINT) &&
      !isTRUE(FORCE_RERUN_EXISTING_CHECKPOINTS) &&
      strategy_checkpoint_exists(stage_name, strategy_row)
    ) {
      message("\n>>> Loading existing LEAR checkpoint: ", checkpoint_id)
      result <- load_strategy_checkpoint(stage_name, strategy_row)
    } else {
      result <- run_lear_for_one_strategy(
        eval_df = eval_strategy,
        strategy_row = strategy_row,
        panel_lear = panel_lear,
        x_cols_by_target_zone = x_cols_by_target_zone
      )

      if (isTRUE(ENABLE_CHECKPOINTS)) {
        save_strategy_checkpoint(result, stage_name, strategy_row)
        message(">>> Checkpoint saved: ", checkpoint_id)
      }
    }

    all_predictions[[i]] <- result$predictions
    all_logs[[i]] <- result$fit_log

    # Free memory between strategies/targets in long runs.
    rm(result)
    invisible(gc())
  }

  list(
    predictions = bind_rows(all_predictions),
    fit_log = bind_rows(all_logs)
  )
}


# ==============================================================================
# 8. METRICS AND STRATEGY SELECTION
# ===============================================================================

compute_validation_metrics <- function(predictions, naive_week_path) {
  assert_file_exists(naive_week_path)

  naive_week <- read_parquet(naive_week_path) %>%
    mutate(
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      delivery_datetime_model = safe_as_posixct_utc(delivery_datetime_model),
      y_pred_naive_week = as.numeric(y_pred)
    ) %>%
    select(target, zone, split, delivery_datetime_model, y_pred_naive_week)

  pred_eval <- predictions %>%
    mutate(
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      delivery_datetime_model = safe_as_posixct_utc(delivery_datetime_model),
      y_true = as.numeric(y_true),
      y_pred = as.numeric(y_pred)
    ) %>%
    left_join(
      naive_week,
      by = c("target", "zone", "split", "delivery_datetime_model")
    ) %>%
    mutate(
      abs_error = abs(y_true - y_pred),
      sq_error = (y_true - y_pred)^2,
      abs_error_naive_week = abs(y_true - y_pred_naive_week)
    )

  metrics_by_zone <- pred_eval %>%
    filter(split == "validation") %>%
    group_by(target, zone, strategy_id, window_months, window_type, recalibration_frequency, lambda_method) %>%
    summarise(
      n = n(),
      n_nonmissing = sum(!is.na(y_pred)),
      MAE = mean(abs_error, na.rm = TRUE),
      RMSE = sqrt(mean(sq_error, na.rm = TRUE)),
      mean_abs_y_true = mean(abs(y_true), na.rm = TRUE),
      NMAE = if_else(mean_abs_y_true > 0, MAE / mean_abs_y_true, NA_real_),
      NRMSE = if_else(mean_abs_y_true > 0, RMSE / mean_abs_y_true, NA_real_),
      rMAE = sum(abs_error, na.rm = TRUE) / sum(abs_error_naive_week, na.rm = TRUE),
      pct_missing_y_pred = 100 * mean(is.na(y_pred)),
      .groups = "drop"
    ) %>%
    arrange(target, zone, rMAE)

  metrics_by_target <- pred_eval %>%
    filter(split == "validation") %>%
    group_by(target, strategy_id, window_months, window_type, recalibration_frequency, lambda_method) %>%
    summarise(
      n = n(),
      n_nonmissing = sum(!is.na(y_pred)),
      validation_MAE = mean(abs_error, na.rm = TRUE),
      validation_RMSE = sqrt(mean(sq_error, na.rm = TRUE)),
      mean_abs_y_true = mean(abs(y_true), na.rm = TRUE),
      validation_NMAE = if_else(mean_abs_y_true > 0, validation_MAE / mean_abs_y_true, NA_real_),
      validation_NRMSE = if_else(mean_abs_y_true > 0, validation_RMSE / mean_abs_y_true, NA_real_),
      validation_rMAE = sum(abs_error, na.rm = TRUE) / sum(abs_error_naive_week, na.rm = TRUE),
      pct_missing_y_pred = 100 * mean(is.na(y_pred)),
      .groups = "drop"
    ) %>%
    arrange(target, validation_rMAE)

  list(
    by_zone = metrics_by_zone,
    by_target = metrics_by_target
  )
}

select_best_strategy_by_target <- function(validation_metrics_by_target) {
  validation_metrics_by_target %>%
    group_by(target) %>%
    arrange(validation_rMAE, validation_MAE, .by_group = TRUE) %>%
    slice(1) %>%
    ungroup() %>%
    transmute(
      target,
      selected_strategy_id = strategy_id,
      strategy_id = strategy_id,
      window_months,
      window_type,
      recalibration_frequency,
      lambda_method,
      validation_MAE,
      validation_RMSE,
      validation_NMAE,
      validation_NRMSE,
      validation_rMAE
    )
}

compute_time_summary <- function(fit_log) {
  if (nrow(fit_log) == 0) {
    return(tibble())
  }

  fit_log %>%
    group_by(target, zone, split, strategy_id) %>%
    summarise(
      number_of_fits = n(),
      total_minutes = sum(total_seconds, na.rm = TRUE) / 60,
      total_hours = sum(total_seconds, na.rm = TRUE) / 3600,
      mean_seconds = mean(total_seconds, na.rm = TRUE),
      median_seconds = median(total_seconds, na.rm = TRUE),
      p90_seconds = as.numeric(stats::quantile(total_seconds, 0.90, na.rm = TRUE)),
      max_seconds = max(total_seconds, na.rm = TRUE),
      number_of_failed_fits = sum(status != "ok" | is.na(status)),
      average_selected_lambda = mean(selected_lambda, na.rm = TRUE),
      average_number_nonzero_coefficients = mean(n_nonzero, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(split, target, zone, strategy_id)
}


# Runtime summary compatible with SCRIPT 05 - EVALUATION METRICS.
# evaluation_metrics.R reads files matching data/model_logs/runtime_*.parquet
# and expects one row per model-target-zone-split.
make_runtime_summary_for_evaluation <- function(predictions, fit_log) {
  if (nrow(predictions) == 0 || nrow(fit_log) == 0) {
    return(tibble())
  }

  pred_counts <- predictions %>%
    mutate(
      model = as.character(model),
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      delivery_date = safe_as_date(delivery_date)
    ) %>%
    group_by(model, target, zone, split) %>%
    summarise(
      n_predictions = n(),
      n_forecast_days = n_distinct(delivery_date),
      .groups = "drop"
    )

  runtime <- fit_log %>%
    mutate(
      model = MODEL_NAME,
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      recalibration_date = safe_as_date(recalibration_date),
      total_seconds = as.numeric(total_seconds),
      failed_fit = status != "ok" | is.na(status)
    ) %>%
    group_by(model, target, zone, split) %>%
    summarise(
      n_recalibrations = n_distinct(recalibration_date),
      total_seconds = sum(total_seconds, na.rm = TRUE),
      total_minutes = total_seconds / 60,
      n_failed = sum(failed_fit, na.rm = TRUE),
      status = if_else(n_failed == 0L, "OK", "FAILED"),
      .groups = "drop"
    ) %>%
    left_join(
      pred_counts,
      by = c("model", "target", "zone", "split")
    ) %>%
    mutate(
      n_predictions = as.integer(coalesce(n_predictions, 0L)),
      n_forecast_days = as.integer(coalesce(n_forecast_days, 0L)),
      n_recalibrations = as.integer(n_recalibrations),
      mean_seconds_per_forecast_day = if_else(
        n_forecast_days > 0L,
        total_seconds / n_forecast_days,
        NA_real_
      )
    ) %>%
    select(
      model,
      target,
      zone,
      split,
      n_predictions,
      n_recalibrations,
      total_seconds,
      total_minutes,
      mean_seconds_per_forecast_day,
      status
    ) %>%
    arrange(split, target, zone)

  runtime
}


# ==============================================================================
# 9. QUALITY CHECKS
# ===============================================================================

quality_check_predictions <- function(predictions, allow_incomplete_days = FALSE) {
  message("\n================ LEAR PREDICTION QUALITY CHECKS ================\n")

  assert_required_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "LEAR predictions")

  predictions <- predictions %>%
    mutate(
      model = as.character(model),
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      forecast_date = safe_as_date(forecast_date),
      delivery_date = safe_as_date(delivery_date),
      delivery_datetime_model = safe_as_posixct_utc(delivery_datetime_model),
      hour = as.integer(hour),
      horizon = as.integer(horizon),
      y_true = as.numeric(y_true),
      y_pred = as.numeric(y_pred),
      strategy_id = as.character(strategy_id)
    )

  duplicates <- predictions %>%
    count(model, target, zone, split, delivery_datetime_model, strategy_id) %>%
    filter(n > 1)

  if (nrow(duplicates) > 0) {
    print(duplicates, n = 100)
    stop("Duplicated LEAR predictions found by model-target-zone-split-datetime-strategy.", call. = FALSE)
  }

  unexpected_horizons <- predictions %>%
    distinct(horizon) %>%
    filter(!(horizon %in% 1:24))

  if (nrow(unexpected_horizons) > 0) {
    print(unexpected_horizons)
    stop("Unexpected horizons found. Expected horizons 1:24.", call. = FALSE)
  }

  day_counts <- predictions %>%
    group_by(model, target, zone, split, strategy_id, delivery_date) %>%
    summarise(n_hours = n_distinct(hour), .groups = "drop")

  incomplete_days <- day_counts %>% filter(n_hours != 24)

  if (nrow(incomplete_days) > 0) {
    msg <- paste0("Found ", nrow(incomplete_days), " model-target-zone-date groups without 24 hourly predictions.")
    if (allow_incomplete_days) {
      warning(msg, " This is allowed because fast/debug mode is active.")
    } else {
      print(incomplete_days, n = 100)
      stop(msg, call. = FALSE)
    }
  }

  missing_summary <- predictions %>%
    group_by(model, target, split, strategy_id) %>%
    summarise(
      n = n(),
      missing_y_true = sum(is.na(y_true)),
      missing_y_pred = sum(is.na(y_pred)),
      pct_missing_y_pred = 100 * mean(is.na(y_pred)),
      .groups = "drop"
    ) %>%
    arrange(split, target, strategy_id)

  print(missing_summary, n = 100)

  invisible(predictions)
}

extract_failures <- function(fit_log) {
  fit_log %>%
    filter(status != "ok" | !is.na(error_message)) %>%
    arrange(split, target, zone, hour, recalibration_date, strategy_id)
}

save_lear_outputs <- function(predictions, fit_log, prefix = "lear") {
  safe_write_parquet(fit_log, file.path(LOG_DIR, paste0(prefix, "_fit_log.parquet")))

  time_summary <- compute_time_summary(fit_log)
  safe_write_csv(time_summary, file.path(LOG_DIR, paste0(prefix, "_compute_time_summary.csv")))

  # Only the final LEAR run writes the runtime file consumed by SCRIPT 05.
  # Validation-only runs remain inside the LEAR log folder to avoid being mixed
  # with final model-comparison outputs.
  if (identical(prefix, "lear")) {
    runtime_summary <- make_runtime_summary_for_evaluation(predictions, fit_log)
    runtime_dir <- file.path(RUN_ROOT, "data", "model_logs")
    dir.create(runtime_dir, recursive = TRUE, showWarnings = FALSE)
    safe_write_parquet(runtime_summary, file.path(runtime_dir, "runtime_lear.parquet"))
  } else {
    runtime_summary <- tibble()
  }

  failures <- extract_failures(fit_log)
  if (nrow(failures) > 0) {
    safe_write_csv(failures, file.path(LOG_DIR, paste0(prefix, "_failures.csv")))
  }

  invisible(list(
    time_summary = time_summary,
    runtime_summary = runtime_summary,
    failures = failures
  ))
}


# ==============================================================================
# 10. READ INPUTS
# ===============================================================================

assert_file_exists(INPUT_PANEL)
assert_file_exists(INPUT_EVAL)
assert_file_exists(INPUT_NAIVE_WEEK)

message("Reading panel: ", INPUT_PANEL)
panel <- readRDS(INPUT_PANEL)

message("Reading evaluation index: ", INPUT_EVAL)
eval_index <- readRDS(INPUT_EVAL)

assert_required_columns(
  eval_index,
  c(
    "target", "zone", "split", "forecast_date", "delivery_datetime_model",
    "delivery_date", "hour", "horizon", "y_true"
  ),
  "eval_index"
)

panel_lear <- prepare_lear_panel(panel)

feature_audit <- create_feature_audit(panel_lear, TARGETS, PHYSICAL_ZONES)
safe_write_csv(feature_audit, file.path(LOG_DIR, "lear_feature_audit.csv"))

x_cols_tbl <- tidyr::crossing(target = TARGETS, zone = PHYSICAL_ZONES) %>%
  mutate(
    key = map2_chr(target, zone, make_x_cols_key),
    x_cols = map2(target, zone, ~ get_lear_x_cols(panel_lear, .x, .y))
  )

x_cols_by_target_zone <- setNames(x_cols_tbl$x_cols, x_cols_tbl$key)

message("
================ LEAR FEATURE SETS ================
")
for (i in seq_len(nrow(x_cols_tbl))) {
  message(
    x_cols_tbl$target[[i]], " / ", x_cols_tbl$zone[[i]], ": ",
    paste(x_cols_tbl$x_cols[[i]], collapse = ", ")
  )
}

eval_index <- eval_index %>%
  mutate(
    target = as.character(target),
    zone = as.character(zone),
    split = as.character(split),
    forecast_date = safe_as_date(forecast_date),
    delivery_date = safe_as_date(delivery_date),
    delivery_datetime_model = safe_as_posixct_utc(delivery_datetime_model),
    hour = as.integer(hour),
    horizon = as.integer(horizon),
    y_true = as.numeric(y_true)
  ) %>%
  filter(target %in% TARGETS)


# ==============================================================================
# 11. VALIDATION STAGE: STRATEGY SEARCH ON 2024
# ===============================================================================

validation_eval <- eval_index %>%
  filter(split == "validation")

if (RUN_FAST_VALIDATION) {
  message("\nFAST VALIDATION MODE is active.")
  validation_eval <- validation_eval %>%
    filter(
      zone %in% VALIDATION_ZONES_FAST,
      target %in% VALIDATION_TARGETS_FAST
    ) %>%
    limit_eval_days_per_series(MAX_VALIDATION_DAYS_PER_SERIES)
}

if (nrow(validation_eval) == 0) {
  stop("No validation rows selected. Check validation filters.", call. = FALSE)
}

validation_result <- run_lear_strategies(
  eval_df = validation_eval,
  strategies_tbl = LEAR_STRATEGIES,
  panel_lear = panel_lear,
  x_cols_by_target_zone = x_cols_by_target_zone,
  stage_name = "validation"
)

pred_validation_all <- validation_result$predictions
fit_log_validation <- validation_result$fit_log

pred_validation_all <- quality_check_predictions(
  pred_validation_all,
  allow_incomplete_days = RUN_FAST_VALIDATION
)

safe_write_parquet(
  pred_validation_all,
  file.path(LOG_DIR, "lear_validation_predictions.parquet")
)
safe_write_rds(
  pred_validation_all,
  file.path(LOG_DIR, "lear_validation_predictions.rds")
)

validation_metrics <- compute_validation_metrics(
  predictions = pred_validation_all,
  naive_week_path = INPUT_NAIVE_WEEK
)

safe_write_csv(
  validation_metrics$by_target,
  file.path(LOG_DIR, "lear_validation_strategy_results.csv")
)

safe_write_csv(
  validation_metrics$by_zone,
  file.path(LOG_DIR, "lear_validation_strategy_results_by_zone.csv")
)

selected_strategy_by_target <- select_best_strategy_by_target(validation_metrics$by_target)

safe_write_csv(
  selected_strategy_by_target,
  file.path(LOG_DIR, "lear_selected_strategy_by_target.csv")
)

save_lear_outputs(
  predictions = pred_validation_all,
  fit_log = fit_log_validation,
  prefix = "lear_validation"
)

message("\n================ SELECTED LEAR STRATEGY BY TARGET ================\n")
print(selected_strategy_by_target)

if (!RUN_FINAL_STAGE) {
  message("\nRUN_FINAL_STAGE = FALSE. Validation strategy search completed. Final test predictions were not produced.")
  message("Logs saved to: ", LOG_DIR)
}


# ==============================================================================
# 12. FINAL STAGE: SELECTED STRATEGIES ONLY, VALIDATION + TEST
# ===============================================================================

if (RUN_FINAL_STAGE) {

  message("\nRUN_FINAL_STAGE = TRUE. Running selected LEAR strategies for validation and test.")

  selected_strategies_for_final <- selected_strategy_by_target %>%
    select(target, strategy_id, window_months, window_type, recalibration_frequency, lambda_method)

  final_eval <- eval_index %>%
    filter(split %in% c("validation", "test"))

  if (RUN_FAST_FINAL) {
    message("\nFAST FINAL MODE is active.")
    final_eval <- final_eval %>%
      filter(
        zone %in% FINAL_ZONES_FAST,
        target %in% FINAL_TARGETS_FAST
      ) %>%
      limit_eval_days_per_series(MAX_FINAL_DAYS_PER_SERIES)
  }

  if (nrow(final_eval) == 0) {
    stop("No final-stage rows selected. Check final-stage filters.", call. = FALSE)
  }

  final_result <- run_lear_strategies(
    eval_df = final_eval,
    strategies_tbl = selected_strategies_for_final,
    panel_lear = panel_lear,
    x_cols_by_target_zone = x_cols_by_target_zone,
    stage_name = "final"
  )

  pred_lear <- final_result$predictions
  fit_log_final <- final_result$fit_log

  pred_lear <- quality_check_predictions(
    pred_lear,
    allow_incomplete_days = RUN_FAST_FINAL
  )

  # In the final prediction file, there must be only one LEAR prediction per
  # target-zone-delivery_datetime_model. Different targets may use different
  # selected strategies, but there should be no competing strategies for the same
  # target-zone-hour.
  final_duplicates <- pred_lear %>%
    count(model, target, zone, split, delivery_datetime_model) %>%
    filter(n > 1)

  if (nrow(final_duplicates) > 0) {
    print(final_duplicates, n = 100)
    stop("Final pred_lear contains duplicated predictions. Check selected strategies.", call. = FALSE)
  }

  safe_write_rds(pred_lear, file.path(PRED_DIR, "pred_lear.rds"))
  safe_write_parquet(pred_lear, file.path(PRED_DIR, "pred_lear.parquet"))

  save_lear_outputs(
    predictions = pred_lear,
    fit_log = fit_log_final,
    prefix = "lear"
  )

  message("\nLEAR final predictions saved successfully:")
  message("  ", file.path(PRED_DIR, "pred_lear.rds"))
  message("  ", file.path(PRED_DIR, "pred_lear.parquet"))
  message("LEAR logs saved to: ", LOG_DIR)

}

# ==============================================================================
# END OF SCRIPT 07
# ==============================================================================
