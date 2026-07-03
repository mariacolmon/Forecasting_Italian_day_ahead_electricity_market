# ==============================================================================
# SCRIPT 06 - SARIMAX MODEL 
# ==============================================================================

# ==============================================================================
# 0. LIBRARIES
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(arrow)
  library(forecast)
  library(zoo)
})

RUN_FINAL_STAGE <- TRUE
RESUME_FROM_VALIDATION_CANDIDATES <- TRUE

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================

INPUT_PANEL <- "data/processed/gme_model_panel_weather_hourly.rds"
INPUT_EVAL  <- "data/evaluation/eval_index_hourly.rds"
INPUT_NAIVE_WEEK <- "data/predictions/pred_naive_week_before.parquet"

PRED_DIR <- "data/predictions"
LOG_DIR <- "data/model_logs/sarimax"
CANDIDATE_PRED_DIR <- file.path(LOG_DIR, "validation_candidate_predictions")
DIAG_DIR <- "figures/sarimax_diagnostics"

OUTPUT_PRED_RDS <- file.path(PRED_DIR, "pred_sarimax.rds")
OUTPUT_PRED_PARQUET <- file.path(PRED_DIR, "pred_sarimax.parquet")

dir.create(PRED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CANDIDATE_PRED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG_DIR, recursive = TRUE, showWarnings = FALSE)


PHYSICAL_ZONES <- c("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")
TARGETS <- c("price", "purchases", "sales")

# Cost-control switches ---------------------------------------------------------
RUN_FAST_VALIDATION <- FALSE
VALIDATION_ZONES_FAST <- c("NORD", "CSUD", "SICI")
VALIDATION_TARGETS_FAST <- c("price", "purchases", "sales")

RUN_FAST_FINAL <- FALSE
FINAL_ZONES_FAST <- c("NORD", "CSUD", "SICI")
FINAL_TARGETS_FAST <- c("price", "purchases", "sales")

RUN_PARALLEL <- FALSE
N_WORKERS <- max(1L, parallel::detectCores() - 1L)
RUN_DIAGNOSTICS <- TRUE
SAVE_XREG_AUDIT <- TRUE

# Debugging switches. Keep Inf for final experiment.
MAX_VALIDATION_DAYS_PER_SERIES <- Inf
MAX_FINAL_DAYS_PER_SERIES <- Inf

# Recalibration frequency -------------------------------------------------------
# "weekly" is the first option to try. If still expensive, use "monthly".
RECALIBRATION_FREQUENCY <- "monthly"  # allowed: "weekly", "monthly"

# SARIMA order selection --------------------------------------------------------
# "recalibration": select order on Total Italy at each recalibration date.
# "initial_fixed": select order once on Total Italy before validation and reuse it.
# "fixed": use the manual default orders below without auto.arima.
ORDER_SELECTION_MODE <- "fixed"  # allowed: "recalibration", "initial_fixed", "fixed"
ORDER_SELECTION_USE_XREG <- FALSE        # keep FALSE for speed and stability
ORDER_SELECTION_IC <- "aic"
ORDER_SELECTION_TRACE <- FALSE
ORDER_SELECTION_APPROXIMATION <- TRUE
ORDER_SELECTION_STEPWISE <- TRUE
ORDER_SELECTION_METHOD <- "CSS"          # faster than exact ML for order screening

# Restricted search space for representative Total Italy order selection.
ORDER_MAX_P <- 1
ORDER_MAX_Q <- 1
ORDER_MAX_P_SEASONAL <- 1
ORDER_MAX_Q_SEASONAL <- 1
ORDER_MAX_D <- 0
ORDER_MAX_D_SEASONAL <- 1
ORDER_MAX_ORDER <- 3

# Manual fallback/default orders by target.
# Used when ORDER_SELECTION_MODE == "fixed" or when representative auto.arima fails.
DEFAULT_ORDERS <- tibble::tribble(
  ~target,      ~p, ~d, ~q, ~P, ~D, ~Q, ~s,
  "price",      1L, 0L, 1L, 1L, 1L, 0L, 24L,
  "purchases", 1L, 0L, 1L, 1L, 0L, 0L, 24L,
  "sales",     1L, 0L, 1L, 1L, 0L, 0L, 24L
)

SEASONAL_PERIOD <- 24
MIN_TRAIN_OBS <- 24 * 30

# Strategies compared in validation.
# The order selection criterion is controlled by ORDER_SELECTION_IC; validation
# selects only the calibration window length.
SARIMAX_STRATEGIES <- tibble::tribble(
  ~strategy_id, ~window_months, ~ic,
  "sarimax_3m", 3L, ORDER_SELECTION_IC,
  "sarimax_6m", 6L, ORDER_SELECTION_IC
)
USE_TARGET_SPECIFIC_REGIONAL_XREG <- TRUE
REGIONAL_LAG_HOURS <- 24

USE_CROSS_REGION_LAGS_AS_XREG <- FALSE
USE_NATIONAL_LAGS_AS_XREG <- FALSE
USE_OWN_ZONE_STRUCTURE_LAGS_AS_XREG <- FALSE
USE_TARGET_LAGS_AS_XREG <- FALSE
NATIONAL_XREG_VARS <- c(
  "pun",
  "purchases_italy",
  "sales_italy",
  "unsold_italy",
  "purchases_external_total",
  "purchases_external_n_active_areas",
  "sales_external_total",
  "sales_external_n_active_areas"
)

INCLUDE_HOUR_FACTOR <- FALSE     # daily pattern is handled by SARIMA period 24
INCLUDE_WEEKDAY_FACTOR <- TRUE
INCLUDE_MTI_XREG <- FALSE

# Final diagnostic cases.
DIAGNOSTIC_CASES <- tibble::tribble(
  ~target,      ~zone,
  "price",     "NORD",
  "price",     "SICI",
  "purchases", "NORD",
  "sales",     "SUD"
)
CROSS_REGION_LAG_HOURS <- integer(0)
NATIONAL_LAG_HOURS <- integer(0)
CROSS_REGION_MARKET_VARS <- character(0)

# Create output folders.
dir.create(PRED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CANDIDATE_PRED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG_DIR, recursive = TRUE, showWarnings = FALSE)

# Optional parallel backend -----------------------------------------------------
if (RUN_PARALLEL) {
  if (!requireNamespace("furrr", quietly = TRUE) || !requireNamespace("future", quietly = TRUE)) {
    warning("RUN_PARALLEL = TRUE but furrr/future are not installed. Falling back to sequential execution.")
    RUN_PARALLEL <- FALSE
  } else if (ORDER_SELECTION_MODE == "recalibration") {
    warning("RUN_PARALLEL is not recommended with ORDER_SELECTION_MODE = 'recalibration' because order caching is sequential. Falling back to sequential execution.")
    RUN_PARALLEL <- FALSE
  } else {
    future::plan(future::multisession, workers = N_WORKERS)
  }
}

map_safe <- function(.x, .f, ...) {
  if (isTRUE(RUN_PARALLEL)) {
    furrr::future_map(.x, .f, ..., .options = furrr::furrr_options(seed = TRUE))
  } else {
    purrr::map(.x, .f, ...)
  }
}

# Environments used for sequential caching/logging.
ORDER_CACHE <- new.env(parent = emptyenv())
ORDER_LOG_ENV <- new.env(parent = emptyenv())
ORDER_LOG_ENV$rows <- list()

# ==============================================================================
# 2. GENERAL HELPERS
# ==============================================================================

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

safe_as_posix_utc <- function(x) {
  as.POSIXct(x, tz = "UTC")
}

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

mode_or_na <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

clean_numeric_series <- function(y) {
  y <- safe_numeric(y)
  if (any(!is.finite(y))) {
    y <- zoo::na.approx(y, na.rm = FALSE, rule = 2)
    med_y <- suppressWarnings(median(y, na.rm = TRUE))
    if (is.na(med_y) || is.nan(med_y)) med_y <- 0
    y[!is.finite(y)] <- med_y
  }
  y
}

order_row_to_list <- function(row) {
  list(
    order = c(as.integer(row$p), as.integer(row$d), as.integer(row$q)),
    seasonal = c(as.integer(row$P), as.integer(row$D), as.integer(row$Q)),
    period = as.integer(row$s)
  )
}

get_default_order <- function(target) {
  row <- DEFAULT_ORDERS %>% filter(target == !!target)
  if (nrow(row) != 1) stop("No default SARIMAX order defined for target: ", target)
  order_row_to_list(row)
}

extract_order_from_fit <- function(fit, target, fallback_source = "auto.arima") {
  ord <- tryCatch(forecast::arimaorder(fit), error = function(e) NULL)

  if (!is.null(ord)) {
    tibble(
      target = target,
      p = as.integer(ord[["p"]] %||% NA_integer_),
      d = as.integer(ord[["d"]] %||% NA_integer_),
      q = as.integer(ord[["q"]] %||% NA_integer_),
      P = as.integer(ord[["P"]] %||% NA_integer_),
      D = as.integer(ord[["D"]] %||% NA_integer_),
      Q = as.integer(ord[["Q"]] %||% NA_integer_),
      s = as.integer(ord[["Frequency"]] %||% ord[["frequency"]] %||% SEASONAL_PERIOD),
      order_source = fallback_source
    )
  } else {
    arma_vec <- fit$arma %||% rep(NA_integer_, 7)
    tibble(
      target = target,
      p = as.integer(arma_vec[1]),
      d = as.integer(arma_vec[6]),
      q = as.integer(arma_vec[2]),
      P = as.integer(arma_vec[3]),
      D = as.integer(arma_vec[7]),
      Q = as.integer(arma_vec[4]),
      s = as.integer(arma_vec[5] %||% SEASONAL_PERIOD),
      order_source = fallback_source
    )
  }
}

get_fit_order <- function(fit) {
  out <- tibble(
    selected_p = NA_integer_, selected_d = NA_integer_, selected_q = NA_integer_,
    selected_P = NA_integer_, selected_D = NA_integer_, selected_Q = NA_integer_,
    selected_s = NA_integer_
  )

  ord <- tryCatch(forecast::arimaorder(fit), error = function(e) NULL)

  if (!is.null(ord)) {
    out$selected_p <- as.integer(ord[["p"]] %||% NA_integer_)
    out$selected_d <- as.integer(ord[["d"]] %||% NA_integer_)
    out$selected_q <- as.integer(ord[["q"]] %||% NA_integer_)
    out$selected_P <- as.integer(ord[["P"]] %||% NA_integer_)
    out$selected_D <- as.integer(ord[["D"]] %||% NA_integer_)
    out$selected_Q <- as.integer(ord[["Q"]] %||% NA_integer_)
    out$selected_s <- as.integer(ord[["Frequency"]] %||% ord[["frequency"]] %||% SEASONAL_PERIOD)
    return(out)
  }

  arma_vec <- fit$arma %||% rep(NA_integer_, 7)
  if (length(arma_vec) >= 7) {
    out$selected_p <- as.integer(arma_vec[1])
    out$selected_q <- as.integer(arma_vec[2])
    out$selected_P <- as.integer(arma_vec[3])
    out$selected_Q <- as.integer(arma_vec[4])
    out$selected_s <- as.integer(arma_vec[5])
    out$selected_d <- as.integer(arma_vec[6])
    out$selected_D <- as.integer(arma_vec[7])
  }
  out
}

make_na_predictions <- function(eval_day, target, zone, window_months, ic, strategy_id) {
  eval_day %>%
    transmute(
      model = "sarimax",
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      forecast_date = as.Date(forecast_date),
      delivery_datetime_model = safe_as_posix_utc(delivery_datetime_model),
      delivery_date = as.Date(delivery_date),
      hour = as.integer(hour),
      horizon = as.integer(horizon),
      y_true = as.numeric(y_true),
      y_pred = NA_real_,
      strategy_id = as.character(strategy_id),
      window_months = as.integer(window_months),
      ic = as.character(ic)
    )
}

# ==============================================================================
# 3. LOAD INPUTS AND PREPARE PANEL
# ==============================================================================

load_inputs <- function() {
  if (!file.exists(INPUT_PANEL)) stop("Missing input panel: ", INPUT_PANEL)
  if (!file.exists(INPUT_EVAL)) stop("Missing evaluation index: ", INPUT_EVAL)

  list(
    panel = readRDS(INPUT_PANEL),
    eval_index = readRDS(INPUT_EVAL)
  )
}

prepare_sarimax_panel <- function(panel) {
  required <- c("datetime_model", "date", "hour", "zone", TARGETS)
  missing_required <- setdiff(required, names(panel))
  if (length(missing_required) > 0) {
    stop("Missing required columns in panel: ", paste(missing_required, collapse = ", "))
  }
  if (!all(panel$hour %in% 1:24)) {
    stop("Panel hour column must be in 1:24 after DST normalization.")
  }

  panel %>%
    mutate(
      datetime_model = safe_as_posix_utc(datetime_model),
      date = as.Date(date),
      hour = as.integer(hour),
      zone = as.character(zone),
      year = lubridate::year(date),
      month = lubridate::month(date),
      weekday_num = lubridate::wday(date, week_start = 1),
      is_weekend = weekday_num %in% c(6, 7)
    ) %>%
    arrange(zone, datetime_model)
}

make_same_target_cross_zone_lags <- function(panel, lag_hours = 24L) {

  panel %>%
    filter(zone %in% PHYSICAL_ZONES) %>%
    select(datetime_model, zone, price, purchases, sales) %>%
    pivot_longer(
      cols = c(price, purchases, sales),
      names_to = "variable",
      values_to = "value"
    ) %>%
    arrange(zone, variable, datetime_model) %>%
    group_by(zone, variable) %>%
    mutate(
      value_lag = dplyr::lag(safe_numeric(value), lag_hours),
      lag_name = paste0("lag", lag_hours)
    ) %>%
    ungroup() %>%
    select(datetime_model, zone, variable, lag_name, value_lag) %>%
    pivot_wider(
      names_from = c(variable, zone, lag_name),
      values_from = value_lag,
      names_glue = "{variable}_{zone}_{lag_name}"
    )
}


make_other_zone_aggregate_lags <- function(panel, lag_hours = 24L) {

  base <- panel %>%
    filter(zone %in% PHYSICAL_ZONES) %>%
    group_by(datetime_model) %>%
    mutate(
      n_zones = n(),
      price_sum = sum(safe_numeric(price), na.rm = TRUE),
      purchases_sum = sum(safe_numeric(purchases), na.rm = TRUE),
      sales_sum = sum(safe_numeric(sales), na.rm = TRUE),

      other_zones_price_mean = if_else(
        n_zones > 1,
        (price_sum - safe_numeric(price)) / (n_zones - 1),
        NA_real_
      ),
      other_zones_purchases_total = purchases_sum - safe_numeric(purchases),
      other_zones_sales_total = sales_sum - safe_numeric(sales)
    ) %>%
    ungroup() %>%
    select(
      datetime_model,
      zone,
      other_zones_price_mean,
      other_zones_purchases_total,
      other_zones_sales_total
    ) %>%
    arrange(zone, datetime_model) %>%
    group_by(zone) %>%
    mutate(
      other_zones_price_mean_lag24 = dplyr::lag(other_zones_price_mean, lag_hours),
      other_zones_purchases_total_lag24 = dplyr::lag(other_zones_purchases_total, lag_hours),
      other_zones_sales_total_lag24 = dplyr::lag(other_zones_sales_total, lag_hours)
    ) %>%
    ungroup() %>%
    select(
      datetime_model,
      zone,
      other_zones_price_mean_lag24,
      other_zones_purchases_total_lag24,
      other_zones_sales_total_lag24
    )
}

validate_eval_index <- function(eval_index) {
  required <- c(
    "target", "zone", "split", "forecast_date", "delivery_datetime_model",
    "delivery_date", "hour", "horizon", "y_true"
  )
  missing_required <- setdiff(required, names(eval_index))
  if (length(missing_required) > 0) {
    stop("Missing required columns in eval_index: ", paste(missing_required, collapse = ", "))
  }
  if (!all(c("validation", "test") %in% unique(eval_index$split))) {
    stop("eval_index must contain both validation and test splits.")
  }

  eval_index %>%
    mutate(
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      forecast_date = as.Date(forecast_date),
      delivery_datetime_model = safe_as_posix_utc(delivery_datetime_model),
      delivery_date = as.Date(delivery_date),
      hour = as.integer(hour),
      horizon = as.integer(horizon),
      y_true = as.numeric(y_true)
    ) %>%
    arrange(target, zone, delivery_datetime_model)
}

# ==============================================================================
# 3B. CROSS-REGIONAL AND NATIONAL LAG FEATURES
# ==============================================================================

make_market_lags_wide <- function(panel,
                                  lag_hours = CROSS_REGION_LAG_HOURS,
                                  market_vars = CROSS_REGION_MARKET_VARS) {
  market_vars <- intersect(market_vars, names(panel))
  if (!USE_CROSS_REGION_LAGS_AS_XREG || length(market_vars) == 0 || length(lag_hours) == 0) {
    return(tibble(datetime_model = unique(panel$datetime_model)))
  }

  panel_long <- panel %>%
    filter(zone %in% PHYSICAL_ZONES) %>%
    select(datetime_model, zone, all_of(market_vars)) %>%
    mutate(
      datetime_model = safe_as_posix_utc(datetime_model),
      zone = as.character(zone)
    ) %>%
    pivot_longer(
      cols = all_of(market_vars),
      names_to = "variable",
      values_to = "value"
    ) %>%
    arrange(zone, variable, datetime_model)

  lagged_long <- purrr::map_dfr(as.integer(lag_hours), function(L) {
    panel_long %>%
      group_by(zone, variable) %>%
      arrange(datetime_model, .by_group = TRUE) %>%
      mutate(
        value_lag = dplyr::lag(safe_numeric(value), L),
        lag_name = paste0("lag", L)
      ) %>%
      ungroup()
  })

  lagged_long %>%
    select(datetime_model, zone, variable, lag_name, value_lag) %>%
    pivot_wider(
      names_from = c(variable, zone, lag_name),
      values_from = value_lag,
      names_glue = "{variable}_{zone}_{lag_name}"
    ) %>%
    arrange(datetime_model)
}

make_national_lags_wide <- function(panel,
                                    lag_hours = NATIONAL_LAG_HOURS,
                                    national_vars = NATIONAL_XREG_VARS) {
  national_vars <- intersect(national_vars, names(panel))
  if (!USE_NATIONAL_LAGS_AS_XREG || length(national_vars) == 0 || length(lag_hours) == 0) {
    return(tibble(datetime_model = unique(panel$datetime_model)))
  }

  # National/common variables are repeated by zone in the panel. Keep one row
  # per modelling timestamp before computing lags.
  panel_nat <- panel %>%
    arrange(datetime_model, zone) %>%
    distinct(datetime_model, .keep_all = TRUE) %>%
    select(datetime_model, all_of(national_vars)) %>%
    mutate(datetime_model = safe_as_posix_utc(datetime_model)) %>%
    arrange(datetime_model)

  for (L in as.integer(lag_hours)) {
    for (v in national_vars) {
      new_col <- paste0(v, "_lag", L)
      panel_nat[[new_col]] <- dplyr::lag(safe_numeric(panel_nat[[v]]), L)
    }
  }

  panel_nat %>%
    select(datetime_model, matches("_lag[0-9]+$")) %>%
    arrange(datetime_model)
}

add_cross_regional_features <- function(panel) {
  market_lags_wide <- make_market_lags_wide(panel)
  national_lags_wide <- make_national_lags_wide(panel)

  out <- panel %>%
    left_join(market_lags_wide, by = "datetime_model") %>%
    left_join(national_lags_wide, by = "datetime_model")

  message("Cross-regional lag columns added: ", max(0L, ncol(market_lags_wide) - 1L))
  message("National/common lag columns added: ", max(0L, ncol(national_lags_wide) - 1L))

  out
}

get_cross_region_lag_cols <- function(df, target_zone) {
  if (!USE_CROSS_REGION_LAGS_AS_XREG || is.null(target_zone) || is.na(target_zone)) {
    return(character(0))
  }

  zone_pattern <- paste(PHYSICAL_ZONES, collapse = "|")
  var_pattern <- paste(intersect(CROSS_REGION_MARKET_VARS, c("price", "purchases", "sales", "hhi", "rsi")), collapse = "|")
  lag_pattern <- paste(as.integer(CROSS_REGION_LAG_HOURS), collapse = "|")

  if (var_pattern == "" || lag_pattern == "") return(character(0))

  candidate_pattern <- paste0("^(", var_pattern, ")_(", zone_pattern, ")_lag(", lag_pattern, ")$")

  cols <- names(df)[stringr::str_detect(names(df), candidate_pattern)]

  # Exclude all lagged market predictors belonging to the target zone itself.
  cols <- cols[!stringr::str_detect(cols, paste0("_", target_zone, "_lag"))]

  cols
}

get_national_lag_cols <- function(df) {
  if (!USE_NATIONAL_LAGS_AS_XREG) return(character(0))

  vars_available <- intersect(NATIONAL_XREG_VARS, names(df))
  # The lagged columns are the ones added by make_national_lags_wide().
  var_pattern <- paste(NATIONAL_XREG_VARS, collapse = "|")
  lag_pattern <- paste(as.integer(NATIONAL_LAG_HOURS), collapse = "|")

  if (var_pattern == "" || lag_pattern == "") return(character(0))

  candidate_pattern <- paste0("^(", var_pattern, ")_lag(", lag_pattern, ")$")
  names(df)[stringr::str_detect(names(df), candidate_pattern)]
}

make_target_zone_data <- function(panel, target, zone) {
  if (!(target %in% TARGETS)) stop("Unknown target: ", target)
  if (!(zone %in% PHYSICAL_ZONES)) stop("Unknown zone: ", zone)

  panel %>%
    filter(zone == !!zone) %>%
    arrange(datetime_model) %>%
    mutate(
      y = safe_numeric(.data[[target]]),
      hhi_lag24 = if ("hhi" %in% names(.)) dplyr::lag(safe_numeric(hhi), 24) else NA_real_,
      rsi_lag24 = if ("rsi" %in% names(.)) dplyr::lag(safe_numeric(rsi), 24) else NA_real_,
      mti_lag24 = if ("mti" %in% names(.)) dplyr::lag(as.character(mti), 24) else NA_character_
    )
}

make_representative_data <- function(panel, target) {
  value_col <- switch(
    target,
    price = "pun",
    purchases = "purchases_italy",
    sales = "sales_italy",
    stop("Unknown target: ", target)
  )

  if (value_col %in% names(panel)) {
    rep_df <- panel %>%
      select(datetime_model, date, hour, all_of(value_col), any_of(c("temperature_2m", "wind_speed_100m", "shortwave_radiation"))) %>%
      group_by(datetime_model, date, hour) %>%
      summarise(
        y = mean(safe_numeric(.data[[value_col]]), na.rm = TRUE),
        temperature_2m = if ("temperature_2m" %in% names(.)) mean(safe_numeric(temperature_2m), na.rm = TRUE) else NA_real_,
        wind_speed_100m = if ("wind_speed_100m" %in% names(.)) mean(safe_numeric(wind_speed_100m), na.rm = TRUE) else NA_real_,
        shortwave_radiation = if ("shortwave_radiation" %in% names(.)) mean(safe_numeric(shortwave_radiation), na.rm = TRUE) else NA_real_,
        .groups = "drop"
      )
  } else {
    warning("Representative column ", value_col, " not found. Falling back to aggregation across physical zones.")
    rep_df <- panel %>%
      filter(zone %in% PHYSICAL_ZONES) %>%
      group_by(datetime_model, date, hour) %>%
      summarise(
        y = if (target == "price") mean(safe_numeric(.data[[target]]), na.rm = TRUE) else sum(safe_numeric(.data[[target]]), na.rm = TRUE),
        temperature_2m = if ("temperature_2m" %in% names(.)) mean(safe_numeric(temperature_2m), na.rm = TRUE) else NA_real_,
        wind_speed_100m = if ("wind_speed_100m" %in% names(.)) mean(safe_numeric(wind_speed_100m), na.rm = TRUE) else NA_real_,
        shortwave_radiation = if ("shortwave_radiation" %in% names(.)) mean(safe_numeric(shortwave_radiation), na.rm = TRUE) else NA_real_,
        .groups = "drop"
      )
  }

  rep_df %>%
    mutate(
      datetime_model = safe_as_posix_utc(datetime_model),
      date = as.Date(date),
      hour = as.integer(hour),
      weekday_num = lubridate::wday(date, week_start = 1)
    ) %>%
    arrange(datetime_model)
}

# ==============================================================================
# 4. XREG CONSTRUCTION
# ==============================================================================

make_xreg_base <- function(df) {
  out <- df %>%
    mutate(
      weekday_f = factor(lubridate::wday(date, week_start = 1), levels = 1:7),
      hour_f = factor(hour, levels = 1:24),
      mti_lag24_f = if ("mti_lag24" %in% names(df)) as.character(mti_lag24) else NA_character_
    )
  out
}

create_xreg_preprocessor <- function(train_df, target, target_zone) {

  numeric_candidates <- c(
    "temperature_2m",
    "wind_speed_100m",
    "shortwave_radiation"
  )

  if (USE_OWN_ZONE_STRUCTURE_LAGS_AS_XREG) {
    numeric_candidates <- c(
      numeric_candidates,
      "hhi_lag24",
      "rsi_lag24"
    )
  }

  if (USE_TARGET_SPECIFIC_REGIONAL_XREG) {

    same_target_cross_zone_cols <- names(train_df)[
      stringr::str_detect(
        names(train_df),
        paste0("^", target, "_(", paste(PHYSICAL_ZONES, collapse = "|"), ")_lag", REGIONAL_LAG_HOURS, "$")
      ) &
        !stringr::str_detect(
          names(train_df),
          paste0("_", target_zone, "_lag", REGIONAL_LAG_HOURS, "$")
        )
    ]

    aggregate_cols <- switch(
      target,
      price = c(
        "other_zones_purchases_total_lag24",
        "other_zones_sales_total_lag24"
      ),
      purchases = c(
        "other_zones_price_mean_lag24",
        "other_zones_sales_total_lag24"
      ),
      sales = c(
        "other_zones_price_mean_lag24",
        "other_zones_purchases_total_lag24"
      ),
      character(0)
    )

    numeric_candidates <- c(
      numeric_candidates,
      same_target_cross_zone_cols,
      aggregate_cols
    )
  }

  if (USE_TARGET_LAGS_AS_XREG) {
    numeric_candidates <- c(numeric_candidates, "target_lag24", "target_lag168")
  }

  numeric_cols <- intersect(numeric_candidates, names(train_df))
  train_base <- make_xreg_base(train_df)

  numeric_info <- list()
  kept_numeric_cols <- character(0)

  for (col in numeric_cols) {
    train_values <- safe_numeric(train_base[[col]])
    med <- suppressWarnings(stats::median(train_values, na.rm = TRUE))
    if (is.na(med) || is.nan(med)) med <- 0

    train_imp <- ifelse(is.na(train_values), med, train_values)
    mu <- mean(train_imp, na.rm = TRUE)
    sig <- stats::sd(train_imp, na.rm = TRUE)

    if (is.na(sig) || sig <= 0) next

    train_base[[col]] <- (train_imp - mu) / sig
    numeric_info[[col]] <- list(median = med, mean = mu, sd = sig)
    kept_numeric_cols <- c(kept_numeric_cols, col)
  }

  factor_cols <- character(0)
  if (INCLUDE_WEEKDAY_FACTOR) factor_cols <- c(factor_cols, "weekday_f")
  if (INCLUDE_HOUR_FACTOR) factor_cols <- c(factor_cols, "hour_f")

  if (INCLUDE_MTI_XREG && "mti_lag24" %in% names(train_df)) {
    train_levels <- sort(unique(train_base$mti_lag24_f[!is.na(train_base$mti_lag24_f)]))
    if (length(train_levels) > 1) {
      factor_cols <- c(factor_cols, "mti_lag24_f")
    }
  }

  xreg_cols <- c(kept_numeric_cols, factor_cols)

  if (length(xreg_cols) == 0) {
    stop("No valid xreg columns available for SARIMAX.")
  }

  x_df <- train_base %>% select(all_of(xreg_cols))

  mm <- stats::model.matrix(~ ., data = x_df)
  if ("(Intercept)" %in% colnames(mm)) {
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  }
  mm[!is.finite(mm)] <- 0

  constant_cols <- colnames(mm)[apply(mm, 2, function(x) stats::sd(x, na.rm = TRUE) == 0)]
  rank_raw <- qr(mm)$rank
  rank_deficient <- rank_raw < ncol(mm)

  list(
    numeric_info = numeric_info,
    xreg_cols = xreg_cols,
    columns = colnames(mm),
    constant_cols = constant_cols,
    rank_raw = rank_raw,
    n_raw = ncol(mm),
    rank_deficient = rank_deficient
  )
}
apply_xreg_preprocessor <- function(df, pp) {
  base <- make_xreg_base(df)

  for (col in names(pp$numeric_info)) {
    info <- pp$numeric_info[[col]]
    values <- safe_numeric(base[[col]])
    imp <- ifelse(is.na(values), info$median, values)
    base[[col]] <- (imp - info$mean) / info$sd
  }

  # MTI levels, if ever used, should be handled manually. It is disabled by default.
  x_df <- base %>% select(all_of(pp$xreg_cols))
  mm <- stats::model.matrix(~ ., data = x_df)
  if ("(Intercept)" %in% colnames(mm)) {
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  }
  mm[!is.finite(mm)] <- 0

  missing_cols <- setdiff(pp$columns, colnames(mm))
  if (length(missing_cols) > 0) {
    add <- matrix(0, nrow = nrow(mm), ncol = length(missing_cols))
    colnames(add) <- missing_cols
    mm <- cbind(mm, add)
  }

  extra_cols <- setdiff(colnames(mm), pp$columns)
  if (length(extra_cols) > 0) {
    mm <- mm[, setdiff(colnames(mm), extra_cols), drop = FALSE]
  }

  mm <- mm[, pp$columns, drop = FALSE]
  mm[!is.finite(mm)] <- 0
  mm
}

make_order_selection_xreg <- function(rep_df) {
  # Optional simple xreg for representative order selection. Disabled by default.
  base <- rep_df %>%
    mutate(weekday_f = factor(lubridate::wday(date, week_start = 1), levels = 1:7)) %>%
    select(weekday_f, any_of(c("temperature_2m", "wind_speed_100m", "shortwave_radiation")))

  for (col in intersect(c("temperature_2m", "wind_speed_100m", "shortwave_radiation"), names(base))) {
    values <- safe_numeric(base[[col]])
    med <- suppressWarnings(median(values, na.rm = TRUE))
    if (is.na(med) || is.nan(med)) med <- 0
    values <- ifelse(is.na(values), med, values)
    sig <- sd(values, na.rm = TRUE)
    if (is.na(sig) || sig <= 0) {
      base[[col]] <- NULL
    } else {
      base[[col]] <- (values - mean(values, na.rm = TRUE)) / sig
    }
  }

  mm <- stats::model.matrix(~ ., data = base)
  if ("(Intercept)" %in% colnames(mm)) {
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  }
  mm[!is.finite(mm)] <- 0
  mm
}

# ==============================================================================
# 5. REPRESENTATIVE ORDER SELECTION
# ==============================================================================

get_initial_order_selection_date <- function(eval_index) {
  min(eval_index$forecast_date[eval_index$split == "validation"], na.rm = TRUE)
}

make_order_cache_key <- function(target, window_months, ic, selection_date) {
  paste(target, window_months, ic, as.character(selection_date), ORDER_SELECTION_MODE, sep = "__")
}

append_order_log <- function(row) {
  ORDER_LOG_ENV$rows[[length(ORDER_LOG_ENV$rows) + 1L]] <- row
}

select_representative_order <- function(panel, target, selection_date, window_months, ic) {
  if (ORDER_SELECTION_MODE == "fixed") {
    default <- DEFAULT_ORDERS %>% filter(target == !!target)
    row <- default %>%
      mutate(
        selection_date = as.Date(selection_date),
        window_months = as.integer(window_months),
        ic = as.character(ic),
        order_selection_seconds = 0,
        order_status = "fixed_default",
        order_error_message = NA_character_
      )
    return(row)
  }

  rep_df <- make_representative_data(panel, target)
  train_start <- as.Date(selection_date) %m-% months(window_months)

  train_df <- rep_df %>%
    filter(date >= train_start, date <= as.Date(selection_date)) %>%
    arrange(datetime_model)

  if (nrow(train_df) < MIN_TRAIN_OBS) {
    default <- DEFAULT_ORDERS %>% filter(target == !!target)
    row <- default %>%
      mutate(
        selection_date = as.Date(selection_date),
        window_months = as.integer(window_months),
        ic = as.character(ic),
        order_selection_seconds = 0,
        order_status = "fallback_too_few_observations",
        order_error_message = paste0("Too few representative observations: ", nrow(train_df))
      )
    return(row)
  }

  y <- clean_numeric_series(train_df$y)
  y_ts <- stats::ts(y, frequency = SEASONAL_PERIOD)
  xreg_order <- NULL
  if (ORDER_SELECTION_USE_XREG) {
    xreg_order <- make_order_selection_xreg(train_df)
  }

  start <- Sys.time()
  fit <- tryCatch(
    forecast::auto.arima(
      y = y_ts,
      xreg = xreg_order,
      seasonal = TRUE,
      stationary = FALSE,
      max.p = ORDER_MAX_P,
      max.q = ORDER_MAX_Q,
      max.P = ORDER_MAX_P_SEASONAL,
      max.Q = ORDER_MAX_Q_SEASONAL,
      max.d = ORDER_MAX_D,
      max.D = ORDER_MAX_D_SEASONAL,
      max.order = ORDER_MAX_ORDER,
      ic = ic,
      stepwise = ORDER_SELECTION_STEPWISE,
      approximation = ORDER_SELECTION_APPROXIMATION,
      method = ORDER_SELECTION_METHOD,
      allowdrift = FALSE,
      allowmean = TRUE,
      trace = ORDER_SELECTION_TRACE
    ),
    error = function(e) e
  )
  end <- Sys.time()
  seconds <- as.numeric(difftime(end, start, units = "secs"))

  if (inherits(fit, "error")) {
    default <- DEFAULT_ORDERS %>% filter(target == !!target)
    row <- default %>%
      mutate(
        selection_date = as.Date(selection_date),
        window_months = as.integer(window_months),
        ic = as.character(ic),
        order_selection_seconds = seconds,
        order_status = "fallback_auto_arima_failed",
        order_error_message = fit$message
      )
    return(row)
  }

  extract_order_from_fit(fit, target, fallback_source = "representative_total_italy") %>%
    mutate(
      selection_date = as.Date(selection_date),
      window_months = as.integer(window_months),
      ic = as.character(ic),
      order_selection_seconds = seconds,
      order_status = "ok",
      order_error_message = NA_character_
    )
}

get_representative_order_cached <- function(panel, target, selection_date, window_months, ic) {
  key <- make_order_cache_key(target, window_months, ic, selection_date)

  if (exists(key, envir = ORDER_CACHE, inherits = FALSE)) {
    row <- get(key, envir = ORDER_CACHE)
    row$order_selection_seconds <- 0
    row$from_cache <- TRUE
    return(row)
  }

  row <- select_representative_order(panel, target, selection_date, window_months, ic) %>%
    mutate(from_cache = FALSE)

  assign(key, row, envir = ORDER_CACHE)
  append_order_log(row)
  row
}

get_order_log <- function() {
  if (length(ORDER_LOG_ENV$rows) == 0) return(tibble())
  bind_rows(ORDER_LOG_ENV$rows) %>% distinct()
}

# ==============================================================================
# 6. SARIMAX FITTING AND FORECASTING
# ==============================================================================

fit_fixed_sarimax <- function(y_train, xreg_train, order_row) {
  y_ts <- stats::ts(y_train, frequency = SEASONAL_PERIOD)

  forecast::Arima(
    y = y_ts,
    order = c(order_row$p, order_row$d, order_row$q),
    seasonal = list(
      order = c(order_row$P, order_row$D, order_row$Q),
      period = order_row$s
    ),
    xreg = xreg_train,
    include.mean = TRUE,
    method = "CSS-ML"
  )
}

update_sarimax_state <- function(y_current, xreg_current, fitted_model) {
  y_ts <- stats::ts(y_current, frequency = SEASONAL_PERIOD)
  forecast::Arima(
    y = y_ts,
    model = fitted_model,
    xreg = xreg_current
  )
}

make_fit_log_row <- function(target, zone, split, forecast_date, delivery_date,
                             recalibration_date, strategy_id, window_months, ic,
                             order_row, fit_start_time, fit_end_time,
                             fit_seconds, update_seconds, forecast_seconds,
                             order_selection_seconds, n_train, status,
                             error_message, xreg_pp) {
  tibble(
    target = as.character(target),
    zone = as.character(zone),
    split = as.character(split),
    forecast_date = as.Date(forecast_date),
    delivery_date = as.Date(delivery_date),
    recalibration_date = as.Date(recalibration_date),
    strategy_id = as.character(strategy_id),
    window_months = as.integer(window_months),
    ic = as.character(ic),
    fit_start_time = as.POSIXct(fit_start_time, tz = "UTC"),
    fit_end_time = as.POSIXct(fit_end_time, tz = "UTC"),
    fit_seconds = as.numeric(fit_seconds),
    update_seconds = as.numeric(update_seconds),
    forecast_seconds = as.numeric(forecast_seconds),
    order_selection_seconds = as.numeric(order_selection_seconds),
    total_seconds = as.numeric(fit_seconds) + as.numeric(update_seconds) + as.numeric(forecast_seconds) + as.numeric(order_selection_seconds),
    n_train = as.integer(n_train),
    selected_p = as.integer(order_row$p),
    selected_d = as.integer(order_row$d),
    selected_q = as.integer(order_row$q),
    selected_P = as.integer(order_row$P),
    selected_D = as.integer(order_row$D),
    selected_Q = as.integer(order_row$Q),
    selected_s = as.integer(order_row$s),
    order_status = as.character(order_row$order_status %||% NA_character_),
    order_source = as.character(order_row$order_source %||% NA_character_),
    aic = NA_real_,
    bic = NA_real_,
    status = as.character(status),
    error_message = as.character(error_message),
    xreg_n_raw = as.integer(xreg_pp$n_raw %||% NA_integer_),
    xreg_rank_raw = as.integer(xreg_pp$rank_raw %||% NA_integer_),
    xreg_rank_deficient = as.logical(xreg_pp$rank_deficient %||% NA),
    xreg_constant_cols = paste(xreg_pp$constant_cols %||% character(0), collapse = ";")
  )
}

forecast_one_delivery_from_recalibrated_model <- function(df_tz, eval_day, fitted_model,
                                                          xreg_pp, order_row,
                                                          target, zone, split_i,
                                                          recalibration_date,
                                                          strategy_id, window_months, ic,
                                                          calibration_fit_seconds,
                                                          order_selection_seconds) {
  eval_day <- eval_day %>% arrange(horizon)
  delivery_date_i <- unique(as.Date(eval_day$delivery_date))
  forecast_date_i <- unique(as.Date(eval_day$forecast_date))

  current_start <- forecast_date_i %m-% months(window_months)
  current_df <- df_tz %>%
    filter(date >= current_start, date <= forecast_date_i) %>%
    arrange(datetime_model)

  future_df <- df_tz %>%
    filter(date == delivery_date_i) %>%
    arrange(hour) %>%
    semi_join(eval_day %>% select(delivery_datetime_model), by = c("datetime_model" = "delivery_datetime_model")) %>%
    arrange(hour)

  if (nrow(current_df) < MIN_TRAIN_OBS || nrow(future_df) != 24 || nrow(eval_day) != 24) {
    pred <- make_na_predictions(eval_day, target, zone, window_months, ic, strategy_id)
    log <- make_fit_log_row(
      target, zone, split_i, forecast_date_i, delivery_date_i, recalibration_date,
      strategy_id, window_months, ic, order_row, Sys.time(), Sys.time(),
      calibration_fit_seconds, 0, 0, order_selection_seconds, nrow(current_df),
      "failed", "Invalid current/future/eval rows for day-ahead forecast", xreg_pp
    )
    return(list(predictions = pred, fit_log = log))
  }

  y_current <- clean_numeric_series(current_df$y)
  xreg_current <- apply_xreg_preprocessor(current_df, xreg_pp)
  xreg_future <- apply_xreg_preprocessor(future_df, xreg_pp)

  update_start <- Sys.time()
  updated_fit <- tryCatch(
    update_sarimax_state(y_current, xreg_current, fitted_model),
    error = function(e) e
  )
  update_end <- Sys.time()
  update_seconds <- as.numeric(difftime(update_end, update_start, units = "secs"))

  if (inherits(updated_fit, "error")) {
    pred <- make_na_predictions(eval_day, target, zone, window_months, ic, strategy_id)
    log <- make_fit_log_row(
      target, zone, split_i, forecast_date_i, delivery_date_i, recalibration_date,
      strategy_id, window_months, ic, order_row, update_start, update_end,
      calibration_fit_seconds, update_seconds, 0, order_selection_seconds, nrow(current_df),
      "failed", paste("state update error:", updated_fit$message), xreg_pp
    )
    return(list(predictions = pred, fit_log = log))
  }

  fc_start <- Sys.time()
  fc <- tryCatch(
    forecast::forecast(updated_fit, h = 24, xreg = xreg_future),
    error = function(e) e
  )
  fc_end <- Sys.time()
  forecast_seconds <- as.numeric(difftime(fc_end, fc_start, units = "secs"))

  if (inherits(fc, "error")) {
    pred <- make_na_predictions(eval_day, target, zone, window_months, ic, strategy_id)
    log <- make_fit_log_row(
      target, zone, split_i, forecast_date_i, delivery_date_i, recalibration_date,
      strategy_id, window_months, ic, order_row, fc_start, fc_end,
      calibration_fit_seconds, update_seconds, forecast_seconds, order_selection_seconds, nrow(current_df),
      "failed", paste("forecast error:", fc$message), xreg_pp
    )
    return(list(predictions = pred, fit_log = log))
  }

  pred <- eval_day %>%
    arrange(horizon) %>%
    transmute(
      model = "sarimax",
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      forecast_date = as.Date(forecast_date),
      delivery_datetime_model = safe_as_posix_utc(delivery_datetime_model),
      delivery_date = as.Date(delivery_date),
      hour = as.integer(hour),
      horizon = as.integer(horizon),
      y_true = as.numeric(y_true),
      y_pred = as.numeric(fc$mean),
      strategy_id = as.character(strategy_id),
      window_months = as.integer(window_months),
      ic = as.character(ic),
      recalibration_date = as.Date(recalibration_date)
    )

  log <- make_fit_log_row(
    target, zone, split_i, forecast_date_i, delivery_date_i, recalibration_date,
    strategy_id, window_months, ic, order_row, fc_start, fc_end,
    calibration_fit_seconds, update_seconds, forecast_seconds, order_selection_seconds, nrow(current_df),
    "ok", NA_character_, xreg_pp
  ) %>%
    mutate(
      aic = as.numeric(updated_fit$aic %||% NA_real_),
      bic = as.numeric(updated_fit$bic %||% NA_real_)
    )

  list(predictions = pred, fit_log = log)
}

forecast_recalibration_block <- function(panel, df_tz, eval_block, target, zone,
                                         window_months, ic, strategy_id,
                                         initial_order_selection_date = NULL) {
  split_i <- unique(eval_block$split)
  if (length(split_i) != 1) stop("A recalibration block must contain a single split.")

  recalibration_date <- min(as.Date(eval_block$forecast_date))
  selection_date <- recalibration_date
  if (ORDER_SELECTION_MODE == "initial_fixed") {
    selection_date <- initial_order_selection_date
  }

  order_row <- get_representative_order_cached(
    panel = panel,
    target = target,
    selection_date = selection_date,
    window_months = window_months,
    ic = ic
  )

  train_start <- recalibration_date %m-% months(window_months)
  train_df <- df_tz %>%
    filter(date >= train_start, date <= recalibration_date) %>%
    arrange(datetime_model)

  if (nrow(train_df) < MIN_TRAIN_OBS) {
    pred <- eval_block %>%
      group_by(delivery_date) %>%
      group_split() %>%
      map_dfr(~ make_na_predictions(.x, target, zone, window_months, ic, strategy_id))

    xreg_pp_empty <- list(n_raw = NA_integer_, rank_raw = NA_integer_, rank_deficient = NA, constant_cols = character(0))
    log <- eval_block %>%
      distinct(forecast_date, delivery_date) %>%
      pmap_dfr(function(forecast_date, delivery_date) {
        make_fit_log_row(
          target, zone, split_i, forecast_date, delivery_date, recalibration_date,
          strategy_id, window_months, ic, order_row, Sys.time(), Sys.time(),
          0, 0, 0, order_row$order_selection_seconds %||% 0, nrow(train_df),
          "failed", paste0("Too few recalibration observations: ", nrow(train_df)), xreg_pp_empty
        )
      })
    return(list(predictions = pred, fit_log = log))
  }

  xreg_pp <- tryCatch(
    create_xreg_preprocessor(
      train_df,
      target = target,
      target_zone = zone
    ),
    error = function(e) e
  )
  if (inherits(xreg_pp, "error")) {
    pred <- eval_block %>%
      group_by(delivery_date) %>%
      group_split() %>%
      map_dfr(~ make_na_predictions(.x, target, zone, window_months, ic, strategy_id))

    xreg_pp_empty <- list(n_raw = NA_integer_, rank_raw = NA_integer_, rank_deficient = NA, constant_cols = character(0))
    log <- eval_block %>%
      distinct(forecast_date, delivery_date) %>%
      pmap_dfr(function(forecast_date, delivery_date) {
        make_fit_log_row(
          target, zone, split_i, forecast_date, delivery_date, recalibration_date,
          strategy_id, window_months, ic, order_row, Sys.time(), Sys.time(),
          0, 0, 0, order_row$order_selection_seconds %||% 0, nrow(train_df),
          "failed", paste("xreg preprocessor error:", xreg_pp$message), xreg_pp_empty
        )
      })
    return(list(predictions = pred, fit_log = log))
  }

  if (isTRUE(xreg_pp$rank_deficient)) {
    pred <- eval_block %>%
      group_by(delivery_date) %>%
      group_split() %>%
      map_dfr(~ make_na_predictions(.x, target, zone, window_months, ic, strategy_id))

    log <- eval_block %>%
      distinct(forecast_date, delivery_date) %>%
      pmap_dfr(function(forecast_date, delivery_date) {
        make_fit_log_row(
          target, zone, split_i, forecast_date, delivery_date, recalibration_date,
          strategy_id, window_months, ic, order_row, Sys.time(), Sys.time(),
          0, 0, 0, order_row$order_selection_seconds %||% 0, nrow(train_df),
          "failed", "xreg rank deficient at recalibration", xreg_pp
        )
      })
    return(list(predictions = pred, fit_log = log))
  }

  y_train <- clean_numeric_series(train_df$y)
  xreg_train <- apply_xreg_preprocessor(train_df, xreg_pp)

  fit_start <- Sys.time()
  fit <- tryCatch(
    fit_fixed_sarimax(y_train, xreg_train, order_row),
    error = function(e) e
  )
  fit_end <- Sys.time()
  fit_seconds <- as.numeric(difftime(fit_end, fit_start, units = "secs"))

  # If selected representative order fails for a zone, try the manual fallback order.
  if (inherits(fit, "error")) {
    fallback_order <- DEFAULT_ORDERS %>%
      filter(target == !!target) %>%
      mutate(
        selection_date = as.Date(selection_date),
        window_months = as.integer(window_months),
        ic = as.character(ic),
        order_selection_seconds = 0,
        order_status = "fallback_zonal_fit_failed",
        order_error_message = fit$message,
        order_source = "manual_default"
      )

    fit_start <- Sys.time()
    fit <- tryCatch(
      fit_fixed_sarimax(y_train, xreg_train, fallback_order),
      error = function(e) e
    )
    fit_end <- Sys.time()
    fit_seconds <- as.numeric(difftime(fit_end, fit_start, units = "secs"))
    order_row <- fallback_order
  }

  if (inherits(fit, "error")) {
    pred <- eval_block %>%
      group_by(delivery_date) %>%
      group_split() %>%
      map_dfr(~ make_na_predictions(.x, target, zone, window_months, ic, strategy_id))

    log <- eval_block %>%
      distinct(forecast_date, delivery_date) %>%
      pmap_dfr(function(forecast_date, delivery_date) {
        make_fit_log_row(
          target, zone, split_i, forecast_date, delivery_date, recalibration_date,
          strategy_id, window_months, ic, order_row, fit_start, fit_end,
          fit_seconds, 0, 0, order_row$order_selection_seconds %||% 0, nrow(train_df),
          "failed", paste("zonal Arima fit error:", fit$message), xreg_pp
        )
      })
    return(list(predictions = pred, fit_log = log))
  }

  days <- eval_block %>%
    arrange(delivery_date, horizon) %>%
    group_by(delivery_date) %>%
    group_split()

  results <- vector("list", length(days))
  for (i in seq_along(days)) {
    day_fit_seconds <- if (i == 1L) fit_seconds else 0
    day_order_seconds <- if (i == 1L) (order_row$order_selection_seconds %||% 0) else 0

    results[[i]] <- forecast_one_delivery_from_recalibrated_model(
      df_tz = df_tz,
      eval_day = days[[i]],
      fitted_model = fit,
      xreg_pp = xreg_pp,
      order_row = order_row,
      target = target,
      zone = zone,
      split_i = split_i,
      recalibration_date = recalibration_date,
      strategy_id = strategy_id,
      window_months = window_months,
      ic = ic,
      calibration_fit_seconds = day_fit_seconds,
      order_selection_seconds = day_order_seconds
    )
  }

  list(
    predictions = map_dfr(results, "predictions"),
    fit_log = map_dfr(results, "fit_log")
  )
}

# ==============================================================================
# 7. ROLLING / RECALIBRATION EXECUTION
# ==============================================================================

get_recalibration_period <- function(delivery_date) {
  if (RECALIBRATION_FREQUENCY == "weekly") {
    return(lubridate::floor_date(delivery_date, unit = "week", week_start = 1))
  }
  if (RECALIBRATION_FREQUENCY == "monthly") {
    return(lubridate::floor_date(delivery_date, unit = "month"))
  }
  stop("Invalid RECALIBRATION_FREQUENCY: ", RECALIBRATION_FREQUENCY)
}

build_origin_grid <- function(eval_index, split_filter, targets_filter, zones_filter,
                              max_days_per_series = Inf) {
  out <- eval_index %>%
    filter(split %in% split_filter, target %in% targets_filter, zone %in% zones_filter) %>%
    distinct(target, zone, split, forecast_date, delivery_date) %>%
    arrange(target, zone, split, delivery_date)

  if (is.finite(max_days_per_series)) {
    out <- out %>%
      group_by(target, zone, split) %>%
      slice_head(n = max_days_per_series) %>%
      ungroup()
  }

  out %>% mutate(recalibration_period = get_recalibration_period(delivery_date))
}

run_sarimax_predictions_for_strategy <- function(panel, eval_index, strategy,
                                                 split_filter, targets_filter, zones_filter,
                                                 max_days_per_series = Inf,
                                                 initial_order_selection_date = NULL) {
  origin_grid <- build_origin_grid(
    eval_index = eval_index,
    split_filter = split_filter,
    targets_filter = targets_filter,
    zones_filter = zones_filter,
    max_days_per_series = max_days_per_series
  )

  if (nrow(origin_grid) == 0) {
    return(list(predictions = tibble(), fit_log = tibble()))
  }

  tz_keys <- origin_grid %>% distinct(target, zone)
  tz_data <- pmap(tz_keys, function(target, zone) make_target_zone_data(panel, target, zone))
  names(tz_data) <- paste(tz_keys$target, tz_keys$zone, sep = "__")

  blocks <- origin_grid %>%
    mutate(
      window_months = as.integer(strategy$window_months),
      ic = as.character(strategy$ic),
      strategy_id = as.character(strategy$strategy_id)
    ) %>%
    group_by(target, zone, split, recalibration_period) %>%
    group_split()

  message(
    "Running strategy ", strategy$strategy_id,
    " | splits=", paste(split_filter, collapse = ","),
    " | targets=", paste(targets_filter, collapse = ","),
    " | zones=", paste(zones_filter, collapse = ","),
    " | recalibration=", RECALIBRATION_FREQUENCY,
    " | blocks=", length(blocks),
    " | origins=", nrow(origin_grid)
  )

  res <- map_safe(seq_along(blocks), function(i) {
    block <- blocks[[i]]
    target_i <- unique(block$target)
    zone_i <- unique(block$zone)
    key <- paste(target_i, zone_i, sep = "__")

    message(
      "[SARIMAX] block ", i, "/", length(blocks),
      " | target=", target_i,
      " | zone=", zone_i,
      " | split=", unique(block$split),
      " | recal_period=", unique(block$recalibration_period),
      " | n_origins=", nrow(block),
      " | strategy=", unique(block$strategy_id)
    )

    eval_block <- eval_index %>%
      semi_join(block %>% select(target, zone, split, forecast_date, delivery_date),
                by = c("target", "zone", "split", "forecast_date", "delivery_date")) %>%
      arrange(delivery_date, horizon)

    forecast_recalibration_block(
      panel = panel,
      df_tz = tz_data[[key]],
      eval_block = eval_block,
      target = target_i,
      zone = zone_i,
      window_months = unique(block$window_months),
      ic = unique(block$ic),
      strategy_id = unique(block$strategy_id),
      initial_order_selection_date = initial_order_selection_date
    )
  })

  list(
    predictions = map_dfr(res, "predictions"),
    fit_log = map_dfr(res, "fit_log")
  )
}

run_sarimax_strategy_validation <- function(panel, eval_index, initial_order_selection_date = NULL) {
  validation_zones <- if (RUN_FAST_VALIDATION) VALIDATION_ZONES_FAST else PHYSICAL_ZONES
  validation_targets <- if (RUN_FAST_VALIDATION) VALIDATION_TARGETS_FAST else TARGETS

  validation_runs <- pmap(
    SARIMAX_STRATEGIES,
    function(strategy_id, window_months, ic) {
      strategy <- tibble(strategy_id = strategy_id, window_months = as.integer(window_months), ic = ic)

      out <- run_sarimax_predictions_for_strategy(
        panel = panel,
        eval_index = eval_index,
        strategy = strategy,
        split_filter = "validation",
        targets_filter = validation_targets,
        zones_filter = validation_zones,
        max_days_per_series = MAX_VALIDATION_DAYS_PER_SERIES,
        initial_order_selection_date = initial_order_selection_date
      )

      candidate_file <- file.path(CANDIDATE_PRED_DIR, paste0("pred_", strategy_id, "_validation_candidates.parquet"))
      if (nrow(out$predictions) > 0) arrow::write_parquet(out$predictions, candidate_file)
      out
    }
  )

  list(
    predictions = map_dfr(validation_runs, "predictions"),
    fit_log = map_dfr(validation_runs, "fit_log")
  )
}

# ==============================================================================
# 8. VALIDATION METRICS AND STRATEGY SELECTION
# ==============================================================================

make_naive_week_predictions <- function(panel, eval_index) {
  if (file.exists(INPUT_NAIVE_WEEK)) {
    naive <- arrow::read_parquet(INPUT_NAIVE_WEEK) %>% as_tibble()
    return(
      naive %>%
        mutate(
          target = as.character(target),
          zone = as.character(zone),
          split = as.character(split),
          delivery_datetime_model = safe_as_posix_utc(delivery_datetime_model),
          y_pred_naive_week = as.numeric(y_pred)
        ) %>%
        select(target, zone, split, delivery_datetime_model, y_pred_naive_week)
    )
  }

  warning("Naive weekly parquet not found. Reconstructing weekly naive from panel lag 168.")

  panel_long <- panel %>%
    select(datetime_model, zone, all_of(TARGETS)) %>%
    pivot_longer(cols = all_of(TARGETS), names_to = "target", values_to = "value") %>%
    mutate(datetime_model = safe_as_posix_utc(datetime_model), zone = as.character(zone), target = as.character(target))

  eval_index %>%
    mutate(ref_datetime_model = safe_as_posix_utc(delivery_datetime_model) - hours(168)) %>%
    left_join(panel_long, by = c("target", "zone", "ref_datetime_model" = "datetime_model")) %>%
    transmute(target, zone, split, delivery_datetime_model = safe_as_posix_utc(delivery_datetime_model),
              y_pred_naive_week = as.numeric(value))
}

compute_sarimax_validation_metrics <- function(candidate_predictions, naive_week) {
  if (nrow(candidate_predictions) == 0) return(tibble())

  candidate_predictions %>%
    filter(split == "validation") %>%
    left_join(naive_week, by = c("target", "zone", "split", "delivery_datetime_model")) %>%
    mutate(
      error = y_true - y_pred,
      abs_error = abs(error),
      squared_error = error^2,
      abs_error_naive = abs(y_true - y_pred_naive_week)
    ) %>%
    group_by(target, zone, strategy_id, window_months, ic) %>%
    summarise(
      n = sum(!is.na(y_true) & !is.na(y_pred)),
      n_missing_pred = sum(is.na(y_pred)),
      MAE = mean(abs_error, na.rm = TRUE),
      RMSE = sqrt(mean(squared_error, na.rm = TRUE)),
      mean_abs_y_true = mean(abs(y_true), na.rm = TRUE),
      NMAE = if_else(first(target) %in% c("purchases", "sales") && mean_abs_y_true > 0, MAE / mean_abs_y_true, NA_real_),
      NRMSE = if_else(first(target) %in% c("purchases", "sales") && mean_abs_y_true > 0, RMSE / mean_abs_y_true, NA_real_),
      rMAE = sum(abs_error, na.rm = TRUE) / sum(abs_error_naive, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(target, zone, strategy_id)
}

select_strategy_by_target <- function(validation_metrics) {
  if (nrow(validation_metrics) == 0) stop("Cannot select SARIMAX strategy: validation_metrics is empty.")

  selected <- validation_metrics %>%
    group_by(target, strategy_id, window_months, ic) %>%
    summarise(
      n = sum(n, na.rm = TRUE),
      n_missing_pred = sum(n_missing_pred, na.rm = TRUE),
      MAE = mean(MAE, na.rm = TRUE),
      RMSE = mean(RMSE, na.rm = TRUE),
      NMAE = mean(NMAE, na.rm = TRUE),
      NRMSE = mean(NRMSE, na.rm = TRUE),
      rMAE = mean(rMAE, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      selection_metric_name = "rMAE",
      selection_metric_value = rMAE
    ) %>%
    filter(n > 0, is.finite(selection_metric_value)) %>%
    arrange(target, selection_metric_value, n_missing_pred, MAE) %>%
    group_by(target) %>%
    slice(1) %>%
    ungroup()

  if (nrow(selected) == 0) {
    stop("All SARIMAX validation strategies failed. Check sarimax_failures.csv and sarimax_order_log.csv.")
  }

  selected
}

# ==============================================================================
# 9. FINAL TEST PREDICTIONS
# ==============================================================================

run_sarimax_final_predictions <- function(panel, eval_index, selected_strategy_by_target,
                                          initial_order_selection_date) {

  final_zones <- if (RUN_FAST_FINAL) FINAL_ZONES_FAST else PHYSICAL_ZONES
  final_targets_available <- if (RUN_FAST_FINAL) FINAL_TARGETS_FAST else TARGETS

  selected_to_run <- selected_strategy_by_target %>%
    filter(target %in% final_targets_available)

  runs <- vector("list", nrow(selected_to_run))

  for (j in seq_len(nrow(selected_to_run))) {

    target_j <- selected_to_run$target[j]
    strategy_id_j <- selected_to_run$strategy_id[j]
    window_months_j <- as.integer(selected_to_run$window_months[j])
    ic_j <- as.character(selected_to_run$ic[j])

    checkpoint_pred_file <- file.path(
      LOG_DIR,
      paste0("checkpoint_final_test_", target_j, ".parquet")
    )

    checkpoint_log_file <- file.path(
      LOG_DIR,
      paste0("checkpoint_final_test_log_", target_j, ".parquet")
    )

    if (file.exists(checkpoint_pred_file) && file.exists(checkpoint_log_file)) {

      message("Loading existing final-test checkpoint for target: ", target_j)

      out <- list(
        predictions = arrow::read_parquet(checkpoint_pred_file) %>% as_tibble(),
        fit_log = arrow::read_parquet(checkpoint_log_file) %>% as_tibble()
      )

    } else {

      message("Running final-test target: ", target_j)

      strategy <- tibble(
        strategy_id = strategy_id_j,
        window_months = window_months_j,
        ic = ic_j
      )

      out <- run_sarimax_predictions_for_strategy(
        panel = panel,
        eval_index = eval_index,
        strategy = strategy,
        split_filter = "test",
        targets_filter = target_j,
        zones_filter = final_zones,
        max_days_per_series = MAX_FINAL_DAYS_PER_SERIES,
        initial_order_selection_date = initial_order_selection_date
      )

      if (nrow(out$predictions) > 0) {
        arrow::write_parquet(out$predictions, checkpoint_pred_file)
      }

      if (nrow(out$fit_log) > 0) {
        arrow::write_parquet(out$fit_log, checkpoint_log_file)
      }
    }

    runs[[j]] <- out
  }

  final_predictions <- map_dfr(runs, "predictions") %>%
    mutate(model = "sarimax") %>%
    arrange(model, target, zone, delivery_datetime_model)

  final_fit_log <- map_dfr(runs, "fit_log")

  list(predictions = final_predictions, fit_log = final_fit_log)
}
# ==============================================================================
# 10. COMPUTATIONAL TIME SUMMARY
# ==============================================================================

compute_time_summary <- function(fit_log) {
  if (nrow(fit_log) == 0) return(tibble())

  fit_log %>%
    group_by(target, zone, split, strategy_id, window_months, ic) %>%
    summarise(
      n_forecast_origins = n(),
      n_success = sum(status == "ok", na.rm = TRUE),
      n_failed = sum(status != "ok", na.rm = TRUE),
      failure_pct = 100 * n_failed / n_forecast_origins,
      total_seconds = sum(total_seconds, na.rm = TRUE),
      mean_seconds = mean(total_seconds, na.rm = TRUE),
      median_seconds = median(total_seconds, na.rm = TRUE),
      p90_seconds = as.numeric(quantile(total_seconds, probs = 0.90, na.rm = TRUE)),
      total_order_selection_seconds = sum(order_selection_seconds, na.rm = TRUE),
      total_fit_seconds = sum(fit_seconds, na.rm = TRUE),
      total_update_seconds = sum(update_seconds, na.rm = TRUE),
      total_forecast_seconds = sum(forecast_seconds, na.rm = TRUE),
      mean_xreg_n_raw = mean(xreg_n_raw, na.rm = TRUE),
      mean_xreg_rank_raw = mean(xreg_rank_raw, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(target, zone, split, strategy_id)
}

# ==============================================================================
# 11. FINAL DIAGNOSTICS
# ==============================================================================

fit_sarimax_for_diagnostics <- function(panel, target, zone, selected_strategy, initial_order_selection_date = NULL) {
  df_tz <- make_target_zone_data(panel, target, zone)
  end_date <- as.Date("2024-12-31")
  window_months <- as.integer(selected_strategy$window_months)
  ic <- as.character(selected_strategy$ic)
  train_start <- end_date %m-% months(window_months)

  order_date <- if (ORDER_SELECTION_MODE == "initial_fixed") initial_order_selection_date else end_date
  order_row <- get_representative_order_cached(panel, target, order_date, window_months, ic)

  train_df <- df_tz %>% filter(date >= train_start, date <= end_date) %>% arrange(datetime_model)
  if (nrow(train_df) < MIN_TRAIN_OBS) stop("Too few observations for diagnostics: ", target, " - ", zone)

  xreg_pp <- create_xreg_preprocessor(
    train_df,
    target = target,
    target_zone = zone
  )

  y_train <- clean_numeric_series(train_df$y)
  xreg_train <- apply_xreg_preprocessor(train_df, xreg_pp)
  fit <- fit_fixed_sarimax(y_train, xreg_train, order_row)

  list(fit = fit, train_df = train_df, target = target, zone = zone, window_months = window_months, ic = ic)
}

save_diagnostic_plots <- function(diag_obj) {
  fit <- diag_obj$fit
  res <- as.numeric(stats::residuals(fit))
  target <- diag_obj$target
  zone <- diag_obj$zone
  safe_name <- paste(target, zone, sep = "_")

  png(file.path(DIAG_DIR, paste0("residuals_", safe_name, ".png")), width = 1200, height = 700)
  plot(res, type = "l", main = paste("SARIMAX residuals -", target, zone), ylab = "Residuals", xlab = "Time index")
  abline(h = 0, lty = 2)
  dev.off()

  png(file.path(DIAG_DIR, paste0("acf_residuals_", safe_name, ".png")), width = 1200, height = 700)
  forecast::Acf(res, lag.max = 168, main = paste("ACF residuals -", target, zone))
  dev.off()

  png(file.path(DIAG_DIR, paste0("hist_residuals_", safe_name, ".png")), width = 1200, height = 700)
  hist(res, breaks = 60, main = paste("Residual histogram -", target, zone), xlab = "Residuals")
  dev.off()

  png(file.path(DIAG_DIR, paste0("qq_residuals_", safe_name, ".png")), width = 1200, height = 700)
  qqnorm(res, main = paste("QQ plot residuals -", target, zone))
  qqline(res)
  dev.off()
}

run_final_diagnostics <- function(panel, selected_strategy_by_target, initial_order_selection_date = NULL) {
  if (!RUN_DIAGNOSTICS) return(tibble())

  pmap_dfr(DIAGNOSTIC_CASES, function(target, zone) {
    selected_strategy <- selected_strategy_by_target %>% filter(target == !!target)
    if (nrow(selected_strategy) != 1) {
      return(tibble(target = target, zone = zone, lag = c(24, 168), statistic = NA_real_, p_value = NA_real_, fitdf = NA_integer_, note = "No selected strategy found"))
    }

    diag_obj <- tryCatch(
      fit_sarimax_for_diagnostics(panel, target, zone, selected_strategy, initial_order_selection_date),
      error = function(e) e
    )

    if (inherits(diag_obj, "error")) {
      return(tibble(target = target, zone = zone, lag = c(24, 168), statistic = NA_real_, p_value = NA_real_, fitdf = NA_integer_, note = paste("Diagnostic fit failed:", diag_obj$message)))
    }

    save_diagnostic_plots(diag_obj)
    res <- as.numeric(stats::residuals(diag_obj$fit))
    k_params <- length(stats::coef(diag_obj$fit))

    map_dfr(c(24, 168), function(lag_i) {
      fitdf_i <- min(k_params, lag_i - 1L)
      test <- tryCatch(stats::Box.test(res, lag = lag_i, type = "Ljung-Box", fitdf = fitdf_i), error = function(e) e)
      if (inherits(test, "error")) {
        tibble(target = target, zone = zone, lag = lag_i, statistic = NA_real_, p_value = NA_real_, fitdf = fitdf_i, note = paste("Ljung-Box failed:", test$message))
      } else {
        tibble(target = target, zone = zone, lag = lag_i, statistic = as.numeric(test$statistic), p_value = as.numeric(test$p.value), fitdf = fitdf_i, note = "OK")
      }
    })
  })
}

# ==============================================================================
# 12. OUTPUT CHECKS AND SAVE
# ==============================================================================

check_final_predictions <- function(predictions) {
  required_cols <- c("model", "target", "zone", "split", "forecast_date", "delivery_datetime_model", "delivery_date", "hour", "horizon", "y_true", "y_pred")
  missing_cols <- setdiff(required_cols, names(predictions))
  if (length(missing_cols) > 0) stop("Final predictions are missing required columns: ", paste(missing_cols, collapse = ", "))

  dup <- predictions %>% count(model, target, zone, delivery_datetime_model) %>% filter(n > 1)
  if (nrow(dup) > 0) {
    print(dup, n = 100)
    stop("Duplicated predictions found in final SARIMAX output.")
  }

  bad_24 <- predictions %>% count(model, target, zone, split, delivery_date) %>% filter(n != 24)
  if (nrow(bad_24) > 0) {
    warning("Some day-target-zone groups do not have exactly 24 predictions. See printed rows.")
    print(bad_24, n = 100)
  }

  bad_horizon <- predictions %>% filter(!(horizon %in% 1:24))
  if (nrow(bad_horizon) > 0) stop("Unexpected horizon values found in SARIMAX output.")
  missing_pred <- predictions %>%
    filter(is.na(y_pred))

  if (nrow(missing_pred) > 0) {
    print(
      missing_pred %>%
        count(target, zone, split),
      n = 100
    )

    stop(
      "SARIMAX final output contains missing predictions. ",
      "Check sarimax_failures.csv before running final evaluation."
    )
  }
  invisible(TRUE)
}

save_outputs <- function(final_predictions, validation_metrics, selected_strategy_by_target,
                         fit_log, time_summary, ljung_box, order_log) {
  final_predictions_for_metrics <- final_predictions %>%
    mutate(
      model = "sarimax",
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      forecast_date = as.Date(forecast_date),
      delivery_datetime_model = safe_as_posix_utc(delivery_datetime_model),
      delivery_date = as.Date(delivery_date),
      hour = as.integer(hour),
      horizon = as.integer(horizon),
      y_true = as.numeric(y_true),
      y_pred = as.numeric(y_pred)
    ) %>%
    arrange(model, target, zone, delivery_datetime_model)

  readr::write_csv(validation_metrics, file.path(LOG_DIR, "sarimax_validation_strategy_results.csv"))
  readr::write_csv(selected_strategy_by_target, file.path(LOG_DIR, "sarimax_selected_strategy_by_target.csv"))

  if (nrow(fit_log) > 0) {
    arrow::write_parquet(fit_log, file.path(LOG_DIR, "sarimax_fit_log.parquet"))
  }

  failures <- fit_log %>% filter(status != "ok")
  if (nrow(failures) > 0) {
    readr::write_csv(failures, file.path(LOG_DIR, "sarimax_failures.csv"))
  }

  # Now check and save final predictions.
  check_final_predictions(final_predictions_for_metrics)

  saveRDS(final_predictions_for_metrics, OUTPUT_PRED_RDS)
  arrow::write_parquet(final_predictions_for_metrics, OUTPUT_PRED_PARQUET)

  readr::write_csv(time_summary, file.path(LOG_DIR, "sarimax_compute_time_summary.csv"))

  MAIN_LOG_DIR <- "data/model_logs"
  dir.create(MAIN_LOG_DIR, recursive = TRUE, showWarnings = FALSE)

  runtime_sarimax <- time_summary %>%
    mutate(
      model = "sarimax",
      n_predictions = n_forecast_origins * 24,
      total_minutes = total_seconds / 60,
      mean_seconds_per_forecast_day = mean_seconds
    ) %>%
    select(
      model,
      target,
      zone,
      split,
      strategy_id,
      window_months,
      ic,
      n_predictions,
      n_forecast_origins,
      n_success,
      n_failed,
      failure_pct,
      total_seconds,
      total_minutes,
      mean_seconds_per_forecast_day,
      median_seconds,
      p90_seconds,
      total_order_selection_seconds,
      total_fit_seconds,
      total_update_seconds,
      total_forecast_seconds
    )

  arrow::write_parquet(
    runtime_sarimax,
    file.path(MAIN_LOG_DIR, "runtime_sarimax.parquet")
  )
  readr::write_csv(ljung_box, file.path(LOG_DIR, "sarimax_ljung_box.csv"))
  readr::write_csv(order_log, file.path(LOG_DIR, "sarimax_order_log.csv"))



  if (SAVE_XREG_AUDIT && nrow(fit_log) > 0) {
    fit_log %>%
      select(target, zone, split, forecast_date, delivery_date, recalibration_date, strategy_id,
             xreg_n_raw, xreg_rank_raw, xreg_rank_deficient, xreg_constant_cols) %>%
      readr::write_csv(file.path(LOG_DIR, "sarimax_xreg_audit.csv"))
  }

  invisible(TRUE)

}

# ==============================================================================
# 13. MAIN
# ==============================================================================

message("\n================ SARIMAX SCRIPT STARTED ================\n")

inputs <- load_inputs()
panel <- prepare_sarimax_panel(inputs$panel)

if (USE_TARGET_SPECIFIC_REGIONAL_XREG) {

  same_target_cross_zone_lags <- make_same_target_cross_zone_lags(
    panel,
    lag_hours = REGIONAL_LAG_HOURS
  )

  other_zone_aggregate_lags <- make_other_zone_aggregate_lags(
    panel,
    lag_hours = REGIONAL_LAG_HOURS
  )

  panel <- panel %>%
    left_join(same_target_cross_zone_lags, by = "datetime_model") %>%
    left_join(other_zone_aggregate_lags, by = c("datetime_model", "zone"))

} else {

  panel <- add_cross_regional_features(panel)
}
eval_index <- validate_eval_index(inputs$eval_index)
initial_order_selection_date <- get_initial_order_selection_date(eval_index)

message("Panel rows: ", nrow(panel))
message("Eval rows: ", nrow(eval_index))
message("RUN_FAST_VALIDATION: ", RUN_FAST_VALIDATION)
message("RECALIBRATION_FREQUENCY: ", RECALIBRATION_FREQUENCY)
message("ORDER_SELECTION_MODE: ", ORDER_SELECTION_MODE)
message("ORDER_SELECTION_IC: ", ORDER_SELECTION_IC)
message("USE_CROSS_REGION_LAGS_AS_XREG: ", USE_CROSS_REGION_LAGS_AS_XREG)
message("CROSS_REGION_LAG_HOURS: ", paste(CROSS_REGION_LAG_HOURS, collapse = ","))
message("USE_NATIONAL_LAGS_AS_XREG: ", USE_NATIONAL_LAGS_AS_XREG)
message("NATIONAL_LAG_HOURS: ", paste(NATIONAL_LAG_HOURS, collapse = ","))
message("USE_OWN_ZONE_STRUCTURE_LAGS_AS_XREG: ", USE_OWN_ZONE_STRUCTURE_LAGS_AS_XREG)
message("Initial order selection date: ", initial_order_selection_date)
message("RUN_PARALLEL: ", RUN_PARALLEL)

# 1. Candidate validation strategies on 2024.
naive_week <- make_naive_week_predictions(panel, eval_index)

if (RESUME_FROM_VALIDATION_CANDIDATES) {

  candidate_files <- list.files(
    CANDIDATE_PRED_DIR,
    pattern = "^pred_sarimax_(3m|6m)_validation_candidates\\.parquet$",
    full.names = TRUE
  )

  if (length(candidate_files) == 0) {
    stop("No validation candidate prediction files found in: ", CANDIDATE_PRED_DIR)
  }

  message("Resuming from validation candidate files:")
  print(candidate_files)

  validation_predictions <- purrr::map_dfr(
    candidate_files,
    ~ arrow::read_parquet(.x) %>% as_tibble()
  )

  validation_run <- list(
    predictions = validation_predictions,
    fit_log = tibble(
      target = character(),
      zone = character(),
      split = character(),
      forecast_date = as.Date(character()),
      delivery_date = as.Date(character()),
      strategy_id = character(),
      window_months = integer(),
      ic = character()
    )
  )

} else {

  validation_run <- run_sarimax_strategy_validation(
    panel,
    eval_index,
    initial_order_selection_date
  )
}

validation_metrics <- compute_sarimax_validation_metrics(
  validation_run$predictions,
  naive_week
)

selected_strategy_by_target <- select_strategy_by_target(validation_metrics)
message("\n================ SELECTED SARIMAX STRATEGY BY TARGET ================\n")
print(selected_strategy_by_target, n = 20)

if (!RUN_FINAL_STAGE) {

  fit_log_all <- validation_run$fit_log %>%
    mutate(run_stage = "validation_strategy_selection")

  time_summary <- compute_time_summary(fit_log_all)
  order_log <- get_order_log()

  readr::write_csv(validation_metrics, file.path(LOG_DIR, "sarimax_validation_strategy_results.csv"))
  readr::write_csv(selected_strategy_by_target, file.path(LOG_DIR, "sarimax_selected_strategy_by_target.csv"))

  if (nrow(fit_log_all) > 0) {
    arrow::write_parquet(fit_log_all, file.path(LOG_DIR, "sarimax_fit_log.parquet"))
  }

  readr::write_csv(time_summary, file.path(LOG_DIR, "sarimax_compute_time_summary.csv"))
  readr::write_csv(order_log, file.path(LOG_DIR, "sarimax_order_log.csv"))

  message("\nValidation-only SARIMAX experiment completed.")
  message("Validation metrics saved to: ", file.path(LOG_DIR, "sarimax_validation_strategy_results.csv"))
  message("Fit log saved to: ", file.path(LOG_DIR, "sarimax_fit_log.parquet"))

  stop("Stopped after validation stage because RUN_FINAL_STAGE = FALSE.")
}





message("\n================ SELECTED SARIMAX STRATEGY BY TARGET ================\n")
print(selected_strategy_by_target, n = 20)

# 2. Final test predictions with the strategy selected in validation.
final_run <- run_sarimax_final_predictions(
  panel,
  eval_index,
  selected_strategy_by_target,
  initial_order_selection_date
)

selected_keys <- selected_strategy_by_target %>%
  select(target, strategy_id, window_months, ic)

selected_validation_predictions <- validation_run$predictions %>%
  inner_join(
    selected_keys,
    by = c("target", "strategy_id", "window_months", "ic")
  )

selected_validation_fit_log <- validation_run$fit_log %>%
  inner_join(
    selected_keys,
    by = c("target", "strategy_id", "window_months", "ic")
  )

final_predictions_all <- bind_rows(
  selected_validation_predictions,
  final_run$predictions
) %>%
  arrange(model, target, zone, delivery_datetime_model)

fit_log_selected <- bind_rows(
  selected_validation_fit_log %>% mutate(run_stage = "selected_validation"),
  final_run$fit_log %>% mutate(run_stage = "final_test")
) %>%
  arrange(target, zone, split, forecast_date, run_stage)

fit_log_all <- bind_rows(
  validation_run$fit_log %>% mutate(run_stage = "validation_strategy_selection"),
  final_run$fit_log %>% mutate(run_stage = "final_test")
) %>%
  arrange(target, zone, split, forecast_date, run_stage)

time_summary <- compute_time_summary(fit_log_selected)

order_log <- get_order_log()

# 4. Final diagnostics.
ljung_box <- run_final_diagnostics(panel, selected_strategy_by_target, initial_order_selection_date)

# 5. Save outputs.
save_outputs(
  final_predictions = final_predictions_all,
  validation_metrics = validation_metrics,
  selected_strategy_by_target = selected_strategy_by_target,
  fit_log = fit_log_all,
  time_summary = time_summary,
  ljung_box = ljung_box,
  order_log = order_log
)

message("\n================ SARIMAX VALIDATION METRICS ================\n")
print(validation_metrics, n = 100)

message("\n================ SARIMAX ORDER LOG ================\n")
print(order_log, n = 100)

message("\n================ SARIMAX COMPUTE TIME SUMMARY ================\n")
print(time_summary, n = 100)

n_failures <- fit_log_all %>% filter(status != "ok") %>% nrow()
message("\nNumber of failed SARIMAX fits/forecasts: ", n_failures)

message("\nFinal SARIMAX predictions saved to:")
message("  - ", OUTPUT_PRED_RDS)
message("  - ", OUTPUT_PRED_PARQUET)
message("The final prediction file is compatible with evaluation_metrics.R.")
message("Order log saved to: ", file.path(LOG_DIR, "sarimax_order_log.csv"))
message("Diagnostics saved in: ", DIAG_DIR)
message("Logs saved in: ", LOG_DIR)

if (RUN_PARALLEL) {
  future::plan(future::sequential)
}

message("\n================ SARIMAX SCRIPT COMPLETED ================\n")
