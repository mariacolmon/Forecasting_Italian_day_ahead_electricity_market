# ==============================================================================
# SCRIPT 05 - EVALUATION METRICS AND STATISTICAL TESTS
#
# Input:
#   data/predictions/pred_*.parquet
# ==============================================================================


# ==============================================================================
# 0. LIBRARIES
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(lubridate)
  library(forecast)
})


# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================

PRED_DIR <- "data/predictions"
OUT_DIR  <- "data/metrics"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

BENCHMARKS <- c("naive_day_before", "naive_week_before")

# For rMAE in the main table.
# Lago et al. recommend the weekly seasonal naive because it captures weekly effects.
MAIN_RMAE_BENCHMARK <- "naive_week_before"

# DM tests can be heavy once many models are available.
# Keep TRUE if you want the table to be generated automatically.
RUN_DM_TESTS <- TRUE

# DM test settings.
DM_SPLIT <- "test"
DM_POWER <- 1       # 1 = absolute error loss, consistent with MAE
DM_H <- 1           # Approximation; tests are applied to hourly error series


# ==============================================================================
# 2. READ PREDICTION FILES
# ==============================================================================

prediction_files <- list.files(
  PRED_DIR,
  pattern = "^pred_.*\\.parquet$",
  full.names = TRUE
)

if (length(prediction_files) == 0) {
  stop("No prediction files found in ", PRED_DIR)
}

predictions_all <- map_dfr(prediction_files, read_parquet)

required_cols <- c(
  "model",
  "target",
  "zone",
  "split",
  "forecast_date",
  "delivery_datetime_model",
  "delivery_date",
  "hour",
  "horizon",
  "y_true",
  "y_pred"
)

missing_cols <- setdiff(required_cols, names(predictions_all))

if (length(missing_cols) > 0) {
  stop(
    "Missing required columns in predictions_all: ",
    paste(missing_cols, collapse = ", ")
  )
}

predictions_all <- predictions_all %>%
  mutate(
    model = as.character(model),
    target = as.character(target),
    zone = as.character(zone),
    split = as.character(split),
    forecast_date = as.Date(forecast_date),
    delivery_date = as.Date(delivery_date),
    delivery_datetime_model = as.POSIXct(delivery_datetime_model, tz = "UTC"),
    hour = as.integer(hour),
    horizon = as.integer(horizon),
    y_true = as.numeric(y_true),
    y_pred = as.numeric(y_pred)
  )


# ==============================================================================
# 3. QUALITY CHECKS
# ==============================================================================

message("\n================ PREDICTION QUALITY CHECKS ================\n")

qc_available_models <- predictions_all %>%
  distinct(model) %>%
  arrange(model)

print(qc_available_models)

qc_missing <- predictions_all %>%
  group_by(model, target, split) %>%
  summarise(
    n = n(),
    missing_y_true = sum(is.na(y_true)),
    missing_y_pred = sum(is.na(y_pred)),
    pct_missing_y_pred = 100 * mean(is.na(y_pred)),
    .groups = "drop"
  ) %>%
  arrange(model, target, split)

print(qc_missing, n = 100)

qc_duplicates <- predictions_all %>%
  count(model, target, zone, delivery_datetime_model) %>%
  filter(n > 1)

if (nrow(qc_duplicates) > 0) {
  warning("Duplicated predictions found for model-target-zone-datetime.")
  print(qc_duplicates, n = 100)
} else {
  message("No duplicated predictions found.")
}

qc_horizon <- predictions_all %>%
  count(horizon) %>%
  arrange(horizon)

print(qc_horizon, n = 50)

# In hourly regime, horizons should be 1,...,24.
unexpected_horizons <- qc_horizon %>%
  filter(!(horizon %in% 1:24))

if (nrow(unexpected_horizons) > 0) {
  warning("Unexpected horizons found. Check hourly indexing.")
  print(unexpected_horizons)
}


# ==============================================================================
# 4. ADD ERROR VARIABLES
# ==============================================================================

predictions_all <- predictions_all %>%
  mutate(
    error = y_true - y_pred,
    abs_error = abs(error),
    squared_error = error^2
  )


# ==============================================================================
# 5. CORE METRICS FUNCTION
# ==============================================================================

compute_core_metrics <- function(data, group_vars) {

  data %>%
    filter(!is.na(y_true), !is.na(y_pred)) %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      n = n(),
      MAE = mean(abs_error, na.rm = TRUE),
      RMSE = sqrt(mean(squared_error, na.rm = TRUE)),
      Bias = mean(y_pred - y_true, na.rm = TRUE),
      mean_y_true = mean(y_true, na.rm = TRUE),
      mean_abs_y_true = mean(abs(y_true), na.rm = TRUE),

      # Normalized metrics are meaningful mainly for positive volume variables.
      # They are reported as main metrics for purchases and sales.
      NMAE = if_else(
        first(target) %in% c("purchases", "sales") && mean_abs_y_true > 0,
        MAE / mean_abs_y_true,
        NA_real_
      ),
      NRMSE = if_else(
        first(target) %in% c("purchases", "sales") && mean_abs_y_true > 0,
        RMSE / mean_abs_y_true,
        NA_real_
      ),
      .groups = "drop"
    )
}


# ==============================================================================
# 6. CORE METRIC TABLES
# ==============================================================================

metrics_overall <- compute_core_metrics(
  predictions_all,
  c("model", "target", "split")
)

metrics_by_zone <- compute_core_metrics(
  predictions_all,
  c("model", "target", "zone", "split")
)

metrics_by_horizon <- compute_core_metrics(
  predictions_all,
  c("model", "target", "horizon", "split")
)

metrics_by_zone_horizon <- compute_core_metrics(
  predictions_all,
  c("model", "target", "zone", "horizon", "split")
)


# ==============================================================================
# 7. RELATIVE METRICS: rMAE AND rRMSE AGAINST EACH BENCHMARK
# ==============================================================================

# rMAE is computed as:
#   sum |e_model| / sum |e_benchmark|
#
# This is slightly preferable to MAE_model / MAE_benchmark if there are missing
# predictions, because both errors are aligned observation by observation.

compute_relative_metrics <- function(predictions, benchmark_model) {

  benchmark_errors <- predictions %>%
    filter(model == benchmark_model) %>%
    select(
      target,
      zone,
      split,
      delivery_datetime_model,
      abs_error_benchmark = abs_error,
      squared_error_benchmark = squared_error
    )

  predictions %>%
    filter(model != benchmark_model) %>%
    left_join(
      benchmark_errors,
      by = c("target", "zone", "split", "delivery_datetime_model")
    ) %>%
    filter(
      !is.na(abs_error),
      !is.na(abs_error_benchmark)
    ) %>%
    group_by(model, benchmark = benchmark_model, target, zone, split) %>%
    summarise(
      n = n(),
      sum_abs_error_model = sum(abs_error, na.rm = TRUE),
      sum_abs_error_benchmark = sum(abs_error_benchmark, na.rm = TRUE),
      sum_squared_error_model = sum(squared_error, na.rm = TRUE),
      sum_squared_error_benchmark = sum(squared_error_benchmark, na.rm = TRUE),

      rMAE = sum_abs_error_model / sum_abs_error_benchmark,
      rRMSE = sqrt(sum_squared_error_model / sum_squared_error_benchmark),

      MAE_improvement_pct = 100 * (1 - rMAE),
      RMSE_improvement_pct = 100 * (1 - rRMSE),
      .groups = "drop"
    )
}

available_benchmarks <- intersect(BENCHMARKS, unique(predictions_all$model))

if (length(available_benchmarks) == 0) {
  warning("No benchmark predictions found. Relative metrics will not be computed.")
  metrics_relative_by_zone <- tibble()
} else {
  metrics_relative_by_zone <- map_dfr(
    available_benchmarks,
    ~ compute_relative_metrics(predictions_all, .x)
  )
}

# Add explicit relative metrics for each benchmark against itself.
# In particular, the seasonal naive benchmark has rMAE = 1 by definition.
benchmark_self_relative <- metrics_by_zone %>%
  filter(model %in% available_benchmarks) %>%
  transmute(
    model,
    benchmark = model,
    target,
    zone,
    split,
    n,
    sum_abs_error_model = NA_real_,
    sum_abs_error_benchmark = NA_real_,
    sum_squared_error_model = NA_real_,
    sum_squared_error_benchmark = NA_real_,
    rMAE = 1,
    rRMSE = 1,
    MAE_improvement_pct = 0,
    RMSE_improvement_pct = 0
  )

metrics_relative_by_zone <- bind_rows(
  metrics_relative_by_zone,
  benchmark_self_relative
)
# ==============================================================================
# 7B. ADD MAIN RELATIVE METRICS TO HORIZON TABLES
# ==============================================================================

compute_relative_metrics_by_group <- function(predictions, benchmark_model, group_vars) {

  benchmark_errors <- predictions %>%
    filter(model == benchmark_model) %>%
    select(
      target,
      zone,
      split,
      delivery_datetime_model,
      abs_error_benchmark = abs_error,
      squared_error_benchmark = squared_error
    )

  predictions %>%
    filter(model != benchmark_model) %>%
    left_join(
      benchmark_errors,
      by = c("target", "zone", "split", "delivery_datetime_model")
    ) %>%
    filter(
      !is.na(abs_error),
      !is.na(abs_error_benchmark)
    ) %>%
    mutate(
      benchmark = benchmark_model
    ) %>%
    group_by(model, benchmark, across(all_of(group_vars))) %>%
    summarise(
      n_relative = n(),
      sum_abs_error_model = sum(abs_error, na.rm = TRUE),
      sum_abs_error_benchmark = sum(abs_error_benchmark, na.rm = TRUE),
      sum_squared_error_model = sum(squared_error, na.rm = TRUE),
      sum_squared_error_benchmark = sum(squared_error_benchmark, na.rm = TRUE),

      rMAE = sum_abs_error_model / sum_abs_error_benchmark,
      rRMSE = sqrt(sum_squared_error_model / sum_squared_error_benchmark),

      MAE_improvement_pct = 100 * (1 - rMAE),
      RMSE_improvement_pct = 100 * (1 - rRMSE),
      .groups = "drop"
    )
}

make_benchmark_self_relative_by_group <- function(core_metrics, group_vars, available_benchmarks) {

  core_metrics %>%
    filter(model %in% available_benchmarks) %>%
    transmute(
      model,
      benchmark = model,
      across(all_of(group_vars)),
      n_relative = n,
      sum_abs_error_model = NA_real_,
      sum_abs_error_benchmark = NA_real_,
      sum_squared_error_model = NA_real_,
      sum_squared_error_benchmark = NA_real_,
      rMAE = 1,
      rRMSE = 1,
      MAE_improvement_pct = 0,
      RMSE_improvement_pct = 0
    )
}

if (length(available_benchmarks) > 0) {

  # ---------------------------------------------------------------------------
  # Relative metrics by horizon
  # ---------------------------------------------------------------------------

  metrics_relative_by_horizon <- map_dfr(
    available_benchmarks,
    ~ compute_relative_metrics_by_group(
      predictions = predictions_all,
      benchmark_model = .x,
      group_vars = c("target", "horizon", "split")
    )
  )

  benchmark_self_relative_by_horizon <- make_benchmark_self_relative_by_group(
    core_metrics = metrics_by_horizon,
    group_vars = c("target", "horizon", "split"),
    available_benchmarks = available_benchmarks
  )

  metrics_relative_by_horizon <- bind_rows(
    metrics_relative_by_horizon,
    benchmark_self_relative_by_horizon
  )

  main_rmae_by_horizon <- metrics_relative_by_horizon %>%
    filter(benchmark == MAIN_RMAE_BENCHMARK) %>%
    select(
      model,
      target,
      horizon,
      split,
      rMAE,
      rRMSE,
      MAE_improvement_pct,
      RMSE_improvement_pct
    )

  metrics_by_horizon <- metrics_by_horizon %>%
    left_join(
      main_rmae_by_horizon,
      by = c("model", "target", "horizon", "split")
    ) %>%
    select(
      model,
      target,
      horizon,
      split,
      n,
      MAE,
      RMSE,
      NMAE,
      NRMSE,
      rMAE,
      rRMSE,
      MAE_improvement_pct,
      RMSE_improvement_pct,
      Bias,
      mean_y_true,
      mean_abs_y_true
    ) %>%
    arrange(target, horizon, split, rMAE)

  # ---------------------------------------------------------------------------
  # Relative metrics by zone and horizon
  # This keeps metrics_by_zone_horizon consistent with metrics_by_horizon.
  # ---------------------------------------------------------------------------

  metrics_relative_by_zone_horizon <- map_dfr(
    available_benchmarks,
    ~ compute_relative_metrics_by_group(
      predictions = predictions_all,
      benchmark_model = .x,
      group_vars = c("target", "zone", "horizon", "split")
    )
  )

  benchmark_self_relative_by_zone_horizon <- make_benchmark_self_relative_by_group(
    core_metrics = metrics_by_zone_horizon,
    group_vars = c("target", "zone", "horizon", "split"),
    available_benchmarks = available_benchmarks
  )

  metrics_relative_by_zone_horizon <- bind_rows(
    metrics_relative_by_zone_horizon,
    benchmark_self_relative_by_zone_horizon
  )

  main_rmae_by_zone_horizon <- metrics_relative_by_zone_horizon %>%
    filter(benchmark == MAIN_RMAE_BENCHMARK) %>%
    select(
      model,
      target,
      zone,
      horizon,
      split,
      rMAE,
      rRMSE,
      MAE_improvement_pct,
      RMSE_improvement_pct
    )

  metrics_by_zone_horizon <- metrics_by_zone_horizon %>%
    left_join(
      main_rmae_by_zone_horizon,
      by = c("model", "target", "zone", "horizon", "split")
    ) %>%
    select(
      model,
      target,
      zone,
      horizon,
      split,
      n,
      MAE,
      RMSE,
      NMAE,
      NRMSE,
      rMAE,
      rRMSE,
      MAE_improvement_pct,
      RMSE_improvement_pct,
      Bias,
      mean_y_true,
      mean_abs_y_true
    ) %>%
    arrange(target, zone, horizon, split, rMAE)
}
# ==============================================================================
# 8. MAIN FINAL TABLE
# ==============================================================================

# This table is designed for the thesis.
# It keeps:
#   - price: MAE, RMSE, rMAE
#   - purchases/sales: MAE, RMSE, NMAE, NRMSE, rMAE
#
# rMAE is taken against MAIN_RMAE_BENCHMARK.

main_rmae <- metrics_relative_by_zone %>%
  filter(benchmark == MAIN_RMAE_BENCHMARK) %>%
  select(
    model,
    target,
    zone,
    split,
    rMAE,
    MAE_improvement_pct
  )

metrics_final_main <- metrics_by_zone %>%
  left_join(
    main_rmae,
    by = c("model", "target", "zone", "split")
  ) %>%
  mutate(
    metric_set = case_when(
      target == "price" ~ "MAE, RMSE, rMAE",
      target %in% c("purchases", "sales") ~ "MAE, RMSE, rMAE; NMAE/NRMSE additionally reported",
      TRUE ~ "MAE, RMSE, rMAE"
    )
  ) %>%
  select(
    model,
    target,
    zone,
    split,
    metric_set,
    n,
    MAE,
    RMSE,
    NMAE,
    NRMSE,
    rMAE,
    MAE_improvement_pct,
    Bias,
    mean_y_true,
    mean_abs_y_true
  ) %>%
  arrange(target, zone, split, model)


# ==============================================================================
# 9. DIEBOLD-MARIANO TEST
# ==============================================================================

# DM test is used only for model-vs-benchmark comparisons in the selected split.
# We use absolute-error loss (power = 1), consistent with MAE.
#
# Null hypothesis:
#   Both forecasts have equal predictive accuracy.
#
# Interpretation:
#   p_value < 0.05 suggests a statistically significant difference
#   in predictive accuracy.

safe_dm_test <- function(df, model_a, model_b, target_i, zone_i, split_i) {

  tmp <- df %>%
    filter(
      model %in% c(model_a, model_b),
      target == target_i,
      zone == zone_i,
      split == split_i
    ) %>%
    select(model, delivery_datetime_model, y_true, y_pred) %>%
    pivot_wider(
      names_from = model,
      values_from = y_pred
    ) %>%
    drop_na()

  if (nrow(tmp) < 30) {
    return(tibble(
      model = model_a,
      benchmark = model_b,
      target = target_i,
      zone = zone_i,
      split = split_i,
      n = nrow(tmp),
      dm_statistic = NA_real_,
      p_value = NA_real_,
      alternative = NA_character_,
      note = "Too few aligned observations"
    ))
  }

  e_a <- tmp$y_true - tmp[[model_a]]
  e_b <- tmp$y_true - tmp[[model_b]]

  out <- tryCatch(
    {
      test <- forecast::dm.test(
        e1 = e_a,
        e2 = e_b,
        alternative = "two.sided",
        h = DM_H,
        power = DM_POWER
      )

      tibble(
        model = model_a,
        benchmark = model_b,
        target = target_i,
        zone = zone_i,
        split = split_i,
        n = nrow(tmp),
        dm_statistic = as.numeric(test$statistic),
        p_value = as.numeric(test$p.value),
        alternative = test$alternative,
        note = "OK"
      )
    },
    error = function(e) {
      tibble(
        model = model_a,
        benchmark = model_b,
        target = target_i,
        zone = zone_i,
        split = split_i,
        n = nrow(tmp),
        dm_statistic = NA_real_,
        p_value = NA_real_,
        alternative = NA_character_,
        note = paste("Error:", e$message)
      )
    }
  )

  out
}

if (RUN_DM_TESTS) {

  models_available <- sort(unique(predictions_all$model))
  targets_available <- sort(unique(predictions_all$target))
  zones_available <- sort(unique(predictions_all$zone))

  dm_models <- setdiff(models_available, BENCHMARKS)
  dm_benchmarks <- intersect(BENCHMARKS, models_available)

  if (length(dm_models) == 0 || length(dm_benchmarks) == 0) {

    message("DM tests skipped: need at least one non-benchmark model and one benchmark.")
    dm_tests <- tibble()

  } else {

    dm_grid <- expand_grid(
      model = dm_models,
      benchmark = dm_benchmarks,
      target = targets_available,
      zone = zones_available
    )

    dm_tests <- pmap_dfr(
      dm_grid,
      function(model, benchmark, target, zone) {
        safe_dm_test(
          df = predictions_all,
          model_a = model,
          model_b = benchmark,
          target_i = target,
          zone_i = zone,
          split_i = DM_SPLIT
        )
      }
    ) %>%
      arrange(target, zone, benchmark, model)
  }

} else {

  dm_tests <- tibble()
}

# ==============================================================================
# 9B. COMPUTATION TIMES
# ==============================================================================

LOG_DIR <- "data/model_logs"

runtime_files <- list.files(
  LOG_DIR,
  pattern = "^runtime_.*\\.parquet$",
  full.names = TRUE
)

if (length(runtime_files) > 0) {

  computation_times <- map_dfr(runtime_files, read_parquet) %>%
    mutate(
      model = as.character(model),
      target = as.character(target),
      zone = as.character(zone),
      split = as.character(split),
      n_predictions = as.integer(n_predictions),
      n_recalibrations = as.integer(n_recalibrations),
      total_seconds = as.numeric(total_seconds),
      total_minutes = as.numeric(total_minutes),
      mean_seconds_per_forecast_day = as.numeric(mean_seconds_per_forecast_day)
    )

  computation_times_overall <- computation_times %>%
    group_by(model, split) %>%
    summarise(
      mean_seconds_per_forecast_day =
        weighted.mean(
          x = mean_seconds_per_forecast_day,
          w = n_predictions,
          na.rm = TRUE
        ),
      n_predictions = sum(n_predictions, na.rm = TRUE),
      n_recalibrations = sum(n_recalibrations, na.rm = TRUE),
      total_minutes = sum(total_minutes, na.rm = TRUE),
      n_failed = sum(status != "OK", na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(split, model)

} else {

  warning("No runtime files found. Computation times will not be summarised.")
  computation_times <- tibble()
  computation_times_overall <- tibble()
}
# ==============================================================================
# 10. SAVE OUTPUTS
# ==============================================================================

saveRDS(
  predictions_all,
  file.path(OUT_DIR, "predictions_all.rds")
)

write_csv(
  metrics_overall,
  file.path(OUT_DIR, "metrics_overall.csv")
)

write_csv(
  metrics_by_zone,
  file.path(OUT_DIR, "metrics_by_zone.csv")
)

write_csv(
  metrics_by_horizon,
  file.path(OUT_DIR, "metrics_by_horizon.csv")
)

write_csv(
  metrics_by_zone_horizon,
  file.path(OUT_DIR, "metrics_by_zone_horizon.csv")
)

write_csv(
  metrics_relative_by_zone,
  file.path(OUT_DIR, "metrics_relative_by_zone.csv")
)

write_csv(
  metrics_final_main,
  file.path(OUT_DIR, "metrics_final_main.csv")
)

write_csv(
  dm_tests,
  file.path(OUT_DIR, "dm_tests.csv")
)
write_csv(
  computation_times,
  file.path(OUT_DIR, "computation_times_by_target_zone.csv")
)

write_csv(
  computation_times_overall,
  file.path(OUT_DIR, "computation_times_overall.csv")
)
metrics_with_computation <- metrics_final_main %>%
  left_join(
    computation_times %>%
      select(
        model,
        target,
        zone,
        split,
        n_recalibrations,
        total_minutes,
        mean_seconds_per_forecast_day
      ),
    by = c("model", "target", "zone", "split")
  )

write_csv(
  metrics_with_computation,
  file.path(OUT_DIR, "metrics_with_computation.csv")
)

message("\nEvaluation completed successfully.")
message("Main outputs saved in: ", OUT_DIR)