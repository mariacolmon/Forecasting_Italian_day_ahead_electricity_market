# ==============================================================================
# SCRIPT 01 - INGESTA GME HORARIA
# ==============================================================================

# ==============================================================================
# 0. LIBRARIES
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(lubridate)
  library(janitor)
})


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

DATA_DIR <- "data/raw/gme"
OUTPUT_DIR <- "data/processed"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

FILES <- list.files(
  path = DATA_DIR,
  pattern = "^Anno .*\\.xlsx$",
  full.names = TRUE
)

if (length(FILES) == 0) {
  stop("No Excel files found in ", DATA_DIR)
}

PHYSICAL_ZONES <- c("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")

SHEETS <- list(
  prices    = "Prezzi-Prices",
  purchases = "Acquisti-Purchases",
  sales     = "Vendite-Sales",
  unsold    = "Q invendute-Unsold volumes",
  hhi       = "HHI",
  rsi       = "IOR-RSI",
  mti       = "ITM-MTI",
  legend    = "Legenda"
)


# ==============================================================================
# 2. AUXILIARY FUNCTIONS
# ==============================================================================
clean_gme_names <- function(x) {
  x %>%
    str_replace_all("\\n", " ") %>%
    str_squish() %>%
    str_replace_all("^\\s+|\\s+$", "") %>%
    str_replace("^Data/Date \\(YYYYMMDD\\)$", "date_raw") %>%
    str_replace("^Ora /Hour$", "hour") %>%
    str_replace("^Totale Italia /Total Italy$", "TOTAL_ITALY") %>%
    str_replace("^PUN INDEX GME$", "PUN") %>%
    str_replace("^PUN$", "PUN")
}


parse_gme_datetime <- function(date_raw, hour) {

  # GME reports delivery hours as 1, ..., 24.
  # hour = 1 means 00:00-01:00 local Italian time.
  # Therefore, we subtract 1 hour when constructing the timestamp.

    date_clean <- ymd(as.character(date_raw))
    hour_clean <- as.integer(hour)

    datetime_rome <- as.POSIXct(
      date_clean,
      tz = "Europe/Rome"
    ) + hours(hour_clean - 1)

    tibble(
      datetime_rome = datetime_rome,
      datetime_utc  = with_tz(datetime_rome, "UTC"),
      date = as_date(datetime_rome)
    )
  }

read_gme_sheet <- function(file, sheet, variable_name, value_type = "numeric") {

  message("Reading: ", basename(file), " | ", sheet)

  raw <- read_excel(
    file,
    sheet = sheet,
    .name_repair = "minimal"
  )

  
  names(raw) <- clean_gme_names(names(raw))

  
  raw <- raw %>%
    select(where(~ !all(is.na(.x))))

  
  names(raw) <- clean_gme_names(names(raw))
  names(raw) <- make.unique(names(raw), sep = "_")
  
  names(raw)[1:2] <- c("date_raw", "hour")

  # Normalizar tipos básicos
  raw <- raw %>%
    mutate(
      date_raw = as.character(date_raw),
      hour = as.integer(as.character(hour))
    )

  # Si existe columna N, es relevante para MTI 2025.
  # Nos quedamos solo con N = 1, como decisión metodológica.
  has_mti_rank <- "N" %in% names(raw)

  if (has_mti_rank) {
    raw <- raw %>%
      mutate(N = as.integer(as.character(N))) %>%
      filter(is.na(N) | N == 1)
  }

  # Crear índice secuencial horario dentro del fichero anual.
  # Esto evita errores con hour = 25 y cambios horarios.
  file_year <- year(ymd(raw$date_raw[1]))

  start_rome <- as.POSIXct(
    paste0(file_year, "-01-01 00:00:00"),
    tz = "Europe/Rome"
  )

  start_utc <- with_tz(start_rome, "UTC")

  raw <- raw %>%
    mutate(
      delivery_index = row_number(),
      datetime_utc = start_utc + hours(delivery_index - 1),
      datetime_rome = with_tz(datetime_utc, "Europe/Rome"),
      date = ymd(date_raw),
      mti_rank = if (has_mti_rank) N else NA_integer_
    )

  id_cols <- c(
    "date_raw", "hour", "N",
    "delivery_index", "datetime_utc", "datetime_rome",
    "date", "mti_rank"
  )

  id_cols <- intersect(id_cols, names(raw))

  out <- raw %>%
    pivot_longer(
      cols = -all_of(id_cols),
      names_to = "area",
      values_to = "value"
    ) %>%
    mutate(
      source_file = basename(file),
      sheet = sheet,
      variable = variable_name
    ) %>%
    select(
      source_file, sheet,
      delivery_index,
      datetime_utc, datetime_rome, date, hour,
      mti_rank, area, variable, value
    )

  if (value_type == "numeric") {
    out <- out %>%
      mutate(value = as.numeric(value))
  } else {
    out <- out %>%
      mutate(value = as.character(value))
  }

  out
}

read_legend_safe <- function(file) {

  available_sheets <- excel_sheets(file)

  if (!(SHEETS$legend %in% available_sheets)) {
    return(tibble())
  }

  read_excel(file, sheet = SHEETS$legend, .name_repair = "minimal") %>%
    rename_with(clean_gme_names) %>%
    mutate(source_file = basename(file))
}


# ==============================================================================
# 3. READ ALL RELEVANT SHEETS
# ==============================================================================

prices_long <- map_dfr(
  FILES,
  read_gme_sheet,
  sheet = SHEETS$prices,
  variable_name = "price",
  value_type = "numeric"
)

purchases_long <- map_dfr(
  FILES,
  read_gme_sheet,
  sheet = SHEETS$purchases,
  variable_name = "purchases",
  value_type = "numeric"
)

sales_long <- map_dfr(
  FILES,
  read_gme_sheet,
  sheet = SHEETS$sales,
  variable_name = "sales",
  value_type = "numeric"
)

unsold_long <- map_dfr(
  FILES,
  read_gme_sheet,
  sheet = SHEETS$unsold,
  variable_name = "unsold",
  value_type = "numeric"
)

hhi_long <- map_dfr(
  FILES,
  read_gme_sheet,
  sheet = SHEETS$hhi,
  variable_name = "hhi",
  value_type = "numeric"
)

rsi_long <- map_dfr(
  FILES,
  read_gme_sheet,
  sheet = SHEETS$rsi,
  variable_name = "rsi",
  value_type = "numeric"
)

mti_long <- map_dfr(
  FILES,
  read_gme_sheet,
  sheet = SHEETS$mti,
  variable_name = "mti",
  value_type = "character"
)

gme_metadata <- map_dfr(FILES, read_legend_safe) %>%
  distinct()
check_sheet_order <- function(x, sheet_name) {
  x %>%
    distinct(source_file, date, hour) %>%
    arrange(source_file) %>%
    group_by(source_file) %>%
    summarise(
      sheet = sheet_name,
      n_rows = n(),
      first_date = first(date),
      first_hour = first(hour),
      is_ordered = all(order(date, hour) == seq_along(date)),
      .groups = "drop"
    )
}

qc_sheet_order <- bind_rows(
  check_sheet_order(prices_long, "prices"),
  check_sheet_order(purchases_long, "purchases"),
  check_sheet_order(sales_long, "sales"),
  check_sheet_order(hhi_long, "hhi"),
  check_sheet_order(rsi_long, "rsi"),
  check_sheet_order(mti_long, "mti")
)

print(qc_sheet_order, n=30)
# ==============================================================================
# 5. PHYSICAL-ZONE PANEL — corrected calendar-based construction
# ==============================================================================

# Master calendar from prices.
# Prices are the target and are chronologically reliable.
zonal_calendar <- prices_long %>%
  filter(area %in% PHYSICAL_ZONES) %>%
  transmute(
    datetime_utc,
    datetime_rome,
    date,
    hour,
    zone = as.character(area)
  ) %>%
  distinct(date, hour, zone, .keep_all = TRUE)

# Numeric variables are aligned using the original GME labels:
# date + hour + zone.
# This avoids errors when some sheets, especially HHI, are not chronologically sorted.
zonal_numeric_long <- bind_rows(
  prices_long,
  purchases_long,
  sales_long,
  hhi_long,
  rsi_long
) %>%
  filter(area %in% PHYSICAL_ZONES) %>%
  transmute(
    date,
    hour,
    zone = as.character(area),
    variable,
    value = as.numeric(value)
  ) %>%
  group_by(date, hour, zone, variable) %>%
  summarise(
    value = mean(value, na.rm = TRUE),
    .groups = "drop"
  )

zonal_numeric_wide <- zonal_numeric_long %>%
  pivot_wider(
    id_cols = c(date, hour, zone),
    names_from = variable,
    values_from = value
  )

# MTI is categorical and is also aligned by date + hour + zone.
zonal_mti <- mti_long %>%
  filter(area %in% PHYSICAL_ZONES) %>%
  filter(is.na(mti_rank) | mti_rank == 1) %>%
  transmute(
    date,
    hour,
    zone = as.character(area),
    mti = na_if(as.character(value), "")
  ) %>%
  group_by(date, hour, zone) %>%
  summarise(
    mti = first(mti[!is.na(mti)]),
    .groups = "drop"
  )

gme_physical_panel <- zonal_calendar %>%
  left_join(
    zonal_numeric_wide,
    by = c("date", "hour", "zone"),
    relationship = "one-to-one"
  ) %>%
  left_join(
    zonal_mti,
    by = c("date", "hour", "zone"),
    relationship = "many-to-one"
  ) %>%
  arrange(zone, datetime_utc) %>%
  mutate(
    zone = factor(zone, levels = PHYSICAL_ZONES)
  ) %>%
  select(
    datetime_utc, datetime_rome, date, hour, zone,
    price, purchases, sales, hhi, rsi, mti
  )

gme_physical_panel <- gme_physical_panel %>%
  mutate(
    hhi = if_else(
      date >= as.Date("2025-01-01") & date < as.Date("2025-10-01"),
      hhi * 4,
      hhi
    ),
    rsi = if_else(
      date >= as.Date("2025-01-01") & date < as.Date("2025-10-01"),
      rsi * 4,
      rsi
    )
  )
# ==============================================================================
# 6. NATIONAL FEATURES
# ==============================================================================

# National variables are kept as exogenous features.
# They are not treated as an additional target zone.

national_features <- bind_rows(
  prices_long %>%
    filter(area == "PUN") %>%
    transmute(
      datetime_utc, datetime_rome, date, hour,
      variable = "pun",
      value
    ),

  purchases_long %>%
    filter(area == "TOTAL_ITALY") %>%
    transmute(
      datetime_utc, datetime_rome, date, hour,
      variable = "purchases_italy",
      value
    ),

  sales_long %>%
    filter(area == "TOTAL_ITALY") %>%
    transmute(
      datetime_utc, datetime_rome, date, hour,
      variable = "sales_italy",
      value
    ),

  unsold_long %>%
    filter(area == "TOTAL_ITALY") %>%
    transmute(
      datetime_utc, datetime_rome, date, hour,
      variable = "unsold_italy",
      value
    )
) %>%
  pivot_wider(
    names_from = variable,
    values_from = value
  ) %>%
  arrange(datetime_utc)


# ==============================================================================
# 7. EXTERNAL / VIRTUAL AREA FEATURES
# ==============================================================================

# Areas that are neither physical Italian zones nor national indicators
# may contain information about foreign/virtual/interconnection dynamics.
# They are never used as targets.
#
# To avoid creating a very high-dimensional dataset at this stage,
# we aggregate them by delivery hour.

external_purchases_features <- purchases_long %>%
  filter(
    !(area %in% PHYSICAL_ZONES),
    area != "TOTAL_ITALY"
  ) %>%
  group_by(datetime_utc, datetime_rome, date, hour) %>%
  summarise(
    purchases_external_total = sum(value, na.rm = TRUE),
    purchases_external_n_active_areas =  n_distinct(area[!is.na(value) & value != 0]),
    .groups = "drop"
  )

external_sales_features <- sales_long %>%
  filter(
    !(area %in% PHYSICAL_ZONES),
    area != "TOTAL_ITALY"
  ) %>%
  group_by(datetime_utc, datetime_rome, date, hour) %>%
  summarise(
    sales_external_total = sum(value, na.rm = TRUE),
    sales_external_n_active_areas = n_distinct(area[!is.na(value) & value != 0]),
    .groups = "drop"
  )

external_features <- full_join(
  external_purchases_features,
  external_sales_features,
  by = c("datetime_utc", "datetime_rome", "date", "hour")
) %>%
  arrange(datetime_utc)

external_areas_long <- bind_rows(
  purchases_long,
  sales_long
) %>%
  filter(
    !(area %in% PHYSICAL_ZONES),
    area != "TOTAL_ITALY",
    area != "PUN"
  ) %>%
  select(
    source_file, sheet,
    datetime_utc, datetime_rome, date, hour,
    area, variable, value
  ) %>%
  arrange(area, variable, datetime_utc)


# ==============================================================================
# 8. FINAL MODEL PANEL
# ==============================================================================

# Final modelling dataset:
# one row = datetime × physical zone.
#
# National and external variables are repeated across physical zones at each hour,
# because they are exogenous predictors common to all regional targets.

gme_model_panel <- gme_physical_panel %>%
  left_join(
    national_features %>%
      select(
        datetime_utc,
        pun, purchases_italy, sales_italy, unsold_italy
      ),
    by = "datetime_utc"
  ) %>%
  left_join(
    external_features %>%
      select(
        datetime_utc,
        purchases_external_total,
        purchases_external_n_active_areas,
        sales_external_total,
        sales_external_n_active_areas
      ),
    by = "datetime_utc"
  ) %>%
  arrange(zone, datetime_utc)


# ==============================================================================
# 9. QUALITY CONTROL
# ==============================================================================

qc_time_range <- gme_model_panel %>%
  summarise(
    min_datetime_rome = min(datetime_rome, na.rm = TRUE),
    max_datetime_rome = max(datetime_rome, na.rm = TRUE),
    min_datetime_utc = min(datetime_utc, na.rm = TRUE),
    max_datetime_utc = max(datetime_utc, na.rm = TRUE)
  )

qc_rows_by_year <- gme_model_panel %>%
  mutate(year = year(date)) %>%
  count(year, zone, name = "n_rows") %>%
  arrange(zone, year)

qc_duplicates <- gme_model_panel %>%
  count(datetime_utc, zone) %>%
  filter(n > 1)

qc_missing <- gme_model_panel %>%
  summarise(
    n_rows = n(),

    missing_price = sum(is.na(price)),
    missing_purchases = sum(is.na(purchases)),
    missing_sales = sum(is.na(sales)),
    missing_hhi = sum(is.na(hhi)),
    missing_rsi = sum(is.na(rsi)),
    missing_mti = sum(is.na(mti)),

    missing_pun = sum(is.na(pun)),
    missing_purchases_italy = sum(is.na(purchases_italy)),
    missing_sales_italy = sum(is.na(sales_italy)),
    missing_unsold_italy = sum(is.na(unsold_italy)),

    missing_purchases_external_total = sum(is.na(purchases_external_total)),
    missing_sales_external_total = sum(is.na(sales_external_total))
  )

qc_hours <- gme_model_panel %>%
  count(hour) %>%
  arrange(hour)

qc_external_areas <- external_areas_long %>%
  distinct(area, variable) %>%
  arrange(variable, area)

qc_mti_multiple <- mti_long %>%
  filter(area %in% PHYSICAL_ZONES) %>%
  filter(!is.na(mti_rank)) %>%
  count(source_file, date, hour, area, name = "n_mti_entries") %>%
  filter(n_mti_entries > 1) %>%
  arrange(source_file, date, hour, area)

message("\n================ QUALITY CONTROL ================\n")

print(qc_time_range)
print(qc_rows_by_year)
print(qc_missing)
print(qc_hours)

if (nrow(qc_duplicates) > 0) {
  warning("Duplicated datetime-zone rows found in gme_model_panel.")
  print(qc_duplicates)
} else {
  message("No duplicated datetime-zone rows found.")
}

if (nrow(qc_mti_multiple) > 0) {
  message("MTI contains several marginal technologies in some zone-hours.")
  message("Only N = 1 has been retained in the final model panel.")
}

# ==============================================================================
# CHECK: HHI / IOR-RSI correction in Jan-Sep 2025
# ==============================================================================

check_structure_correction <- function(panel, raw_long, variable_panel, variable_name_raw) {

  raw_check <- raw_long %>%
    filter(area %in% PHYSICAL_ZONES) %>%
    transmute(
      date,
      hour,
      zone = as.character(area),
      raw_value = as.numeric(value)
    ) %>%
    group_by(date, hour, zone) %>%
    summarise(
      raw_value = mean(raw_value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      expected_value = if_else(
        date >= as.Date("2025-01-01") & date < as.Date("2025-10-01"),
        raw_value * 4,
        raw_value
      )
    )

  panel_check <- panel %>%
    transmute(
      date,
      hour,
      zone = as.character(zone),
      panel_value = as.numeric(.data[[variable_panel]])
    )

  out <- panel_check %>%
    left_join(raw_check, by = c("date", "hour", "zone")) %>%
    mutate(
      abs_diff = abs(panel_value - expected_value),
      period = case_when(
        date >= as.Date("2025-01-01") & date < as.Date("2025-10-01") ~ "2025 Jan-Sep corrected",
        date >= as.Date("2025-10-01") & date <= as.Date("2025-12-31") ~ "2025 Oct-Dec unchanged",
        TRUE ~ "Other years unchanged"
      )
    )

  message("\nCheck for: ", variable_name_raw)

  out %>%
    group_by(period) %>%
    summarise(
      n = n(),
      max_abs_diff = max(abs_diff, na.rm = TRUE),
      n_bad = sum(abs_diff > 1e-8, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    print()

  invisible(out)
}

hhi_check <- check_structure_correction(
  panel = gme_model_panel,
  raw_long = hhi_long,
  variable_panel = "hhi",
  variable_name_raw = "HHI"
)

rsi_check <- check_structure_correction(
  panel = gme_model_panel,
  raw_long = rsi_long,
  variable_panel = "rsi",
  variable_name_raw = "IOR-RSI"
)
# ==============================================================================
# 10. SAVE OUTPUTS
# ==============================================================================

saveRDS(
  gme_model_panel,
  file.path(OUTPUT_DIR, "gme_model_panel_hourly.rds")
)

write_csv(
  gme_model_panel,
  file.path(OUTPUT_DIR, "gme_model_panel_hourly.csv")
)

saveRDS(
  gme_physical_panel,
  file.path(OUTPUT_DIR, "gme_physical_panel_hourly.rds")
)

saveRDS(
  national_features,
  file.path(OUTPUT_DIR, "gme_national_features_hourly.rds")
)

saveRDS(
  external_features,
  file.path(OUTPUT_DIR, "gme_external_features_hourly.rds")
)

saveRDS(
  external_areas_long,
  file.path(OUTPUT_DIR, "gme_external_areas_long_hourly.rds")
)

saveRDS(
  gme_metadata,
  file.path(OUTPUT_DIR, "gme_metadata.rds")
)


message("\nIngestion completed successfully.")
message("Main modelling dataset saved as:")
message(file.path(OUTPUT_DIR, "gme_model_panel_hourly.rds"))
