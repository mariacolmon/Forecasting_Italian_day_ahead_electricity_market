# ==============================================================================
# SCRIPT 03 - EVALUATION INDEX
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(arrow)
})

INPUT_FILE <- "data/processed/gme_model_panel_weather_hourly.rds"
OUTPUT_DIR <- "data/evaluation"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

targets <- c("price", "purchases", "sales")

panel <- readRDS(INPUT_FILE)

# Basic checks: after DST normalization, hourly regime must be 1,...,24
panel %>%
  count(hour) %>%
  arrange(hour) %>%
  print(n = 30)

stopifnot(all(panel$hour %in% 1:24))
stopifnot("datetime_model" %in% names(panel))

eval_index <- panel %>%
  mutate(
    year = year(date),
    split = case_when(
      year == 2024 ~ "validation",
      year == 2025 ~ "test",
      TRUE ~ "train"
    )
  ) %>%
  filter(split %in% c("validation", "test")) %>%
  select(
    datetime_model, date, hour, zone, split,
    all_of(targets)
  ) %>%
  pivot_longer(
    cols = all_of(targets),
    names_to = "target",
    values_to = "y_true"
  ) %>%
  mutate(
    zone = as.character(zone),
    delivery_date = as.Date(date),
    hour = as.integer(hour),
    horizon = hour,
    forecast_date = delivery_date - days(1)
  ) %>%
  select(
    target,
    zone,
    split,
    forecast_date,
    delivery_datetime_model = datetime_model,
    delivery_date,
    hour,
    horizon,
    y_true
  ) %>%
  arrange(target, zone, delivery_datetime_model)

saveRDS(
  eval_index,
  file.path(OUTPUT_DIR, "eval_index_hourly.rds")
)

write_parquet(
  eval_index,
  file.path(OUTPUT_DIR, "eval_index_hourly.parquet")
)

message("Evaluation index saved successfully.")