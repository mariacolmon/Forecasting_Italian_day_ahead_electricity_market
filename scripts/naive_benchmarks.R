# ==============================================================================
# SCRIPT 04 - NAIVE BENCHMARKS
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(arrow)
})

INPUT_PANEL <- "data/processed/gme_model_panel_weather_hourly.rds"
INPUT_EVAL  <- "data/evaluation/eval_index_hourly.rds"
OUTPUT_DIR  <- "data/predictions"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

targets <- c("price", "purchases", "sales")

panel <- readRDS(INPUT_PANEL)
eval_index <- readRDS(INPUT_EVAL)

stopifnot("datetime_model" %in% names(panel))
stopifnot("delivery_datetime_model" %in% names(eval_index))

panel_long <- panel %>%
  select(datetime_model, zone, all_of(targets)) %>%
  pivot_longer(
    cols = all_of(targets),
    names_to = "target",
    values_to = "value"
  ) %>%
  mutate(
    zone = as.character(zone),
    datetime_model = as.POSIXct(datetime_model, tz = "UTC")
  ) %>%
  select(target, zone, datetime_model, value)

make_naive_model_time <- function(eval_index, panel_long, lag_hours, model_name) {

  eval_index %>%
    mutate(
      ref_datetime_model = as.POSIXct(delivery_datetime_model, tz = "UTC") - hours(lag_hours)
    ) %>%
    left_join(
      panel_long,
      by = c(
        "target",
        "zone",
        "ref_datetime_model" = "datetime_model"
      )
    ) %>%
    transmute(
      model = model_name,
      target,
      zone,
      split,
      forecast_date,
      delivery_datetime_model,
      delivery_date,
      hour,
      horizon,
      y_true,
      y_pred = value
    ) %>%
    arrange(model, target, zone, delivery_datetime_model)
}

pred_naive_day <- make_naive_model_time(
  eval_index = eval_index,
  panel_long = panel_long,
  lag_hours = 24,
  model_name = "naive_day_before"
)

pred_naive_week <- make_naive_model_time(
  eval_index = eval_index,
  panel_long = panel_long,
  lag_hours = 168,
  model_name = "naive_week_before"
)

saveRDS(
  pred_naive_day,
  file.path(OUTPUT_DIR, "pred_naive_day_before.rds")
)

saveRDS(
  pred_naive_week,
  file.path(OUTPUT_DIR, "pred_naive_week_before.rds")
)

write_parquet(
  pred_naive_day,
  file.path(OUTPUT_DIR, "pred_naive_day_before.parquet")
)

write_parquet(
  pred_naive_week,
  file.path(OUTPUT_DIR, "pred_naive_week_before.parquet")
)

message("Naive benchmark predictions saved successfully.")