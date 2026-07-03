# ==============================================================================
# SCRIPT 02 - WEATHER DATA INGESTION AND JOIN
# ==============================================================================


# ==============================================================================
# 0. LIBRARIES
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(httr)
  library(jsonlite)
  library(zoo)
  library(arrow)
})


# ==============================================================================
# 1. PATHS AND INPUT DATA
# ==============================================================================

OUTPUT_DIR <- "data/processed"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

gme_model_panel <- readRDS(file.path(OUTPUT_DIR, "gme_model_panel_hourly.rds"))

PHYSICAL_ZONES <- c("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")

# ==============================================================================
# 1B. DST NORMALIZATION OF GME PANEL TO STANDARD 24H LOCAL DAYS
# ==============================================================================

# Following common EPF preprocessing practice, daylight-saving time transitions
# are normalized to obtain exactly 24 local hourly observations per day.
#
# - Duplicated autumn hours are averaged.
# - Missing spring hours are linearly interpolated using neighbouring hours.
#
# The original GME panel is preserved in gme_model_panel_hourly.rds.
# The normalized panel is used from this point onwards.

mode_or_na <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

numeric_cols_gme <- c(
  "price", "purchases", "sales",
  "hhi", "rsi",
  "pun", "purchases_italy", "sales_italy", "unsold_italy",
  "purchases_external_total", "purchases_external_n_active_areas",
  "sales_external_total", "sales_external_n_active_areas"
)

numeric_cols_gme <- intersect(numeric_cols_gme, names(gme_model_panel))

gme_model_panel_local <- gme_model_panel %>%
  mutate(
    zone = as.character(zone),
    date = as.Date(date),
    hour_gme = as.integer(hour),
    hour_std = lubridate::hour(datetime_rome) + 1L
  )

qc_dst_before <- gme_model_panel_local %>%
  group_by(date, zone) %>%
  summarise(
    n_rows = n(),
    n_distinct_hour_std = n_distinct(hour_std),
    min_hour_std = min(hour_std, na.rm = TRUE),
    max_hour_std = max(hour_std, na.rm = TRUE),
    hours_std = paste(sort(unique(hour_std)), collapse = ","),
    hours_gme = paste(sort(unique(hour_gme)), collapse = ","),
    .groups = "drop"
  ) %>%
  filter(n_rows != 24 | n_distinct_hour_std != 24)

message("\n================ DST CHECK BEFORE NORMALIZATION ================\n")
print(qc_dst_before, n = 100)

# 1. Collapse duplicated local hours by arithmetic mean
gme_collapsed <- gme_model_panel_local %>%
  group_by(date, zone, hour_std) %>%
  summarise(
    across(all_of(numeric_cols_gme),
           \(x) mean(x, na.rm = TRUE)),
    mti = if ("mti" %in% names(gme_model_panel_local)) mode_or_na(mti) else NA_character_,
    n_original_rows = n(),
    datetime_utc_original = first(datetime_utc),
    datetime_rome_original = first(datetime_rome),
    hour_gme_original = first(hour_gme),
    .groups = "drop"
  )

# 2. Complete 24 local hours and interpolate missing numeric values
gme_model_panel_std24 <- gme_collapsed %>%
  group_by(date, zone) %>%
  complete(hour_std = 1:24) %>%
  arrange(date, zone, hour_std) %>%
  mutate(
    across(
      all_of(numeric_cols_gme),
      ~ zoo::na.approx(.x, x = hour_std, na.rm = FALSE, rule = 2)
    )
  ) %>%
  tidyr::fill(mti, .direction = "downup") %>%
  ungroup() %>%
  mutate(
    hour = as.integer(hour_std),
    hour_of_day = hour,

    # Standard artificial modelling timestamp.
    # It represents a regular 24h local grid, not physical UTC delivery time.
    datetime_model = as.POSIXct(date, tz = "UTC") + hours(hour - 1),

    zone = factor(zone, levels = PHYSICAL_ZONES)
  ) %>%
  select(
    datetime_model,
    date,
    hour,
    zone,
    all_of(numeric_cols_gme),
    mti,
    n_original_rows,
    datetime_utc_original,
    datetime_rome_original,
    hour_gme_original
  ) %>%
  arrange(zone, date, hour)

# 3. Quality control after normalization
qc_dst_after <- gme_model_panel_std24 %>%
  group_by(date, zone) %>%
  summarise(
    n_rows = n(),
    min_hour = min(hour, na.rm = TRUE),
    max_hour = max(hour, na.rm = TRUE),
    n_distinct_hours = n_distinct(hour),
    .groups = "drop"
  )

bad_dst_after <- qc_dst_after %>%
  filter(n_rows != 24 | min_hour != 1 | max_hour != 24 | n_distinct_hours != 24)

message("\n================ DST CHECK AFTER NORMALIZATION ================\n")

if (nrow(bad_dst_after) > 0) {
  warning("Some date-zone combinations are still not standard 24h.")
  print(bad_dst_after, n = 100)
} else {
  message("All date-zone combinations now have exactly 24 hourly observations.")
}

qc_missing_gme_std24 <- gme_model_panel_std24 %>%
  summarise(
    n_rows = n(),
    missing_price = sum(is.na(price)),
    missing_purchases = sum(is.na(purchases)),
    missing_sales = sum(is.na(sales)),
    missing_hhi = sum(is.na(hhi)),
    missing_rsi = sum(is.na(rsi)),
    missing_pun = sum(is.na(pun))
  )

print(qc_missing_gme_std24)

# From this point onwards, use the DST-normalized GME panel.
gme_model_panel <- gme_model_panel_std24

# ==============================================================================
# 2. CITY-ZONE MAP
# ==============================================================================

# Representative cities by GME physical zone.
# These are not weather stations; Open-Meteo returns gridded reanalysis data
# for each coordinate.

df_ciudades_geo <- tibble(
  zone   = c(rep("NORD", 3), rep("CNOR", 3), rep("CSUD", 3),
             rep("SUD", 3), rep("CALA", 2), rep("SICI", 2), rep("SARD", 2)),
  ciudad = c("Milano", "Torino", "Venezia",
             "Bologna", "Firenze", "Ancona",
             "Roma", "Pescara", "Napoli",
             "Bari", "Foggia", "Potenza",
             "Catanzaro", "Reggio Calabria",
             "Palermo", "Catania",
             "Cagliari", "Sassari"),
  lat    = c(45.4642, 45.0703, 45.4343,
             44.4949, 43.7696, 43.6158,
             41.9028, 42.4618, 40.8518,
             41.1171, 41.4622, 40.6383,
             38.9054, 38.1144,
             38.1157, 37.5079,
             39.2238, 40.7259),
  lon    = c(9.1900,  7.6868,  12.3388,
             11.3426, 11.2558, 13.5189,
             12.4964, 14.2142, 14.2681,
             16.8719, 15.5446, 15.8022,
             16.5948, 15.6500,
             13.3613, 15.0830,
             9.1217,  8.5556)
)


# ==============================================================================
# 3. DATE RANGE
# ==============================================================================

# The GME panel is indexed in UTC. Since 2021-01-01 00:00 Rome corresponds to
# 2020-12-31 23:00 UTC, we derive the weather request window from datetime_utc.

WEATHER_START <- min(gme_model_panel$date, na.rm = TRUE) - days(1)
WEATHER_END   <- max(gme_model_panel$date, na.rm = TRUE) + days(1)

message("Weather date range:")
message("  Start: ", WEATHER_START)
message("  End:   ", WEATHER_END)


# ==============================================================================
# 4. OPEN-METEO REQUEST FUNCTION
# ==============================================================================

get_weather_city <- function(zone, ciudad, lat, lon, start_date, end_date) {

  url <- "https://archive-api.open-meteo.com/v1/archive"

  res <- GET(
    url,
    query = list(
      latitude   = lat,
      longitude  = lon,
      start_date = as.character(start_date),
      end_date   = as.character(end_date),
      hourly     = "temperature_2m,wind_speed_100m,shortwave_radiation",
      timezone   = "GMT"
    )
  )

  if (status_code(res) != 200) {
    warning("Error downloading weather for ", ciudad, " (", zone, ")")
    return(NULL)
  }

  contenido <- fromJSON(rawToChar(res$content))

  if (is.null(contenido$hourly)) {
    warning("No hourly data returned for ", ciudad, " (", zone, ")")
    return(NULL)
  }

  as_tibble(contenido$hourly) %>%
    mutate(
      datetime_utc = ymd_hm(time, tz = "UTC"),
      zone = zone,
      ciudad = ciudad
    ) %>%
    select(
      datetime_utc, zone, ciudad,
      temperature_2m,
      wind_speed_100m,
      shortwave_radiation
    )
}


# ==============================================================================
# 5. DOWNLOAD CITY-LEVEL WEATHER
# ==============================================================================

message("=== Downloading hourly weather by city ===")

df_weather_cities <- pmap_dfr(
  df_ciudades_geo,
  function(zone, ciudad, lat, lon) {
    message("  Downloading: ", ciudad, " (", zone, ")")
    get_weather_city(
      zone = zone,
      ciudad = ciudad,
      lat = lat,
      lon = lon,
      start_date = WEATHER_START,
      end_date = WEATHER_END
    )
  }
)


# ==============================================================================
# 6. AGGREGATE CITY WEATHER TO GME ZONES
# ==============================================================================

message("=== Aggregating city weather to GME zones ===")

weather_zonal_hourly <- df_weather_cities %>%
  group_by(zone, datetime_utc) %>%
  summarise(
    temperature_2m = mean(temperature_2m, na.rm = TRUE),
    wind_speed_100m = mean(wind_speed_100m, na.rm = TRUE),
    shortwave_radiation = mean(shortwave_radiation, na.rm = TRUE),
    n_weather_cities = n_distinct(ciudad),
    .groups = "drop"
  ) %>%
  mutate(
    zone = factor(zone, levels = PHYSICAL_ZONES)
  ) %>%
  arrange(zone, datetime_utc)

# ==============================================================================
# 6B. DST NORMALIZATION OF WEATHER TO STANDARD 24H LOCAL DAYS
# ==============================================================================

weather_numeric_cols <- c(
  "temperature_2m",
  "wind_speed_100m",
  "shortwave_radiation"
)

weather_zonal_hourly_std24 <- weather_zonal_hourly %>%
  mutate(
    zone = as.character(zone),
    datetime_rome = with_tz(datetime_utc, "Europe/Rome"),
    date = as.Date(datetime_rome),
    hour = lubridate::hour(datetime_rome) + 1L
  ) %>%
  group_by(zone, date, hour) %>%
  summarise(
    across(all_of(weather_numeric_cols),  \(x) mean(x, na.rm = TRUE)),
    n_weather_cities = first(n_weather_cities),
    .groups = "drop"
  ) %>%
  filter(
    date >= min(gme_model_panel$date),
    date <= max(gme_model_panel$date)
  ) %>%
  group_by(zone, date) %>%
  complete(hour = 1:24) %>%
  arrange(zone, date, hour) %>%
  mutate(
    across(
      all_of(weather_numeric_cols),
      ~ zoo::na.approx(.x, x = hour, na.rm = FALSE, rule = 2)
    )
  ) %>%
  ungroup() %>%
  mutate(
    zone = factor(zone, levels = PHYSICAL_ZONES)
  ) %>%
  arrange(zone, date, hour)

qc_weather_std24 <- weather_zonal_hourly_std24 %>%
  group_by(zone, date) %>%
  summarise(
    n_hours = n(),
    min_hour = min(hour, na.rm = TRUE),
    max_hour = max(hour, na.rm = TRUE),
    missing_temperature = sum(is.na(temperature_2m)),
    missing_wind = sum(is.na(wind_speed_100m)),
    missing_radiation = sum(is.na(shortwave_radiation)),
    .groups = "drop"
  )

bad_weather_std24 <- qc_weather_std24 %>%
  filter(n_hours != 24 | min_hour != 1 | max_hour != 24)

message("\n================ WEATHER DST CHECK AFTER NORMALIZATION ================\n")

if (nrow(bad_weather_std24) > 0) {
  warning("Some weather date-zone combinations are not standard 24h.")
  print(bad_weather_std24, n = 100)
} else {
  message("Weather normalized to standard 24h local days.")
}
# ==============================================================================
# 7. QUALITY CONTROL - WEATHER
# ==============================================================================

qc_weather_range <- weather_zonal_hourly %>%
  group_by(zone) %>%
  summarise(
    n_rows = n(),
    start = min(datetime_utc, na.rm = TRUE),
    end = max(datetime_utc, na.rm = TRUE),
    missing_temperature = sum(is.na(temperature_2m)),
    missing_wind = sum(is.na(wind_speed_100m)),
    missing_radiation = sum(is.na(shortwave_radiation)),
    .groups = "drop"
  )

qc_weather_duplicates <- weather_zonal_hourly %>%
  count(datetime_utc, zone) %>%
  filter(n > 1)

print(qc_weather_range)

if (nrow(qc_weather_duplicates) > 0) {
  warning("Duplicated datetime-zone rows found in weather_zonal_hourly.")
  print(qc_weather_duplicates)
} else {
  message("No duplicated datetime-zone rows found in weather_zonal_hourly.")
}


# ==============================================================================
# 8. JOIN GME + WEATHER
# ==============================================================================

gme_model_panel_weather <- gme_model_panel %>%
  left_join(
    weather_zonal_hourly_std24 %>%
      select(
        zone, date, hour,
        temperature_2m,
        wind_speed_100m,
        shortwave_radiation,
        n_weather_cities
      ),
    by = c("zone", "date", "hour"),
    relationship = "many-to-one"
  ) %>%
  arrange(zone, date, hour)

gme_model_panel_weather <- gme_model_panel_weather %>%
  mutate(
    year = year(date),
    month = month(date),
    weekday = wday(date, label = TRUE, week_start = 1),
    is_weekend = weekday %in% c("Sat", "Sun"),
    hour_of_day = hour
  )



gme_model_panel_weather_wide <- gme_model_panel_weather %>%
  mutate(zone = as.character(zone)) %>%
  pivot_wider(
    id_cols = c(
      datetime_model, date, hour,
      year, month, weekday, is_weekend, hour_of_day,
      pun, purchases_italy, sales_italy, unsold_italy,
      purchases_external_total, purchases_external_n_active_areas,
      sales_external_total, sales_external_n_active_areas
    ),
    names_from = zone,
    values_from = c(
      price, purchases, sales, hhi, rsi,
      temperature_2m, wind_speed_100m, shortwave_radiation
    ),
    names_glue = "{.value}_{zone}"
  ) %>%
  arrange(datetime_model)

# ==============================================================================
# 9. QUALITY CONTROL - JOINED PANEL
# ==============================================================================

qc_join_missing <- gme_model_panel_weather %>%
  summarise(
    n_rows = n(),
    missing_price = sum(is.na(price)),
    missing_purchases = sum(is.na(purchases)),
    missing_sales = sum(is.na(sales)),
    missing_hhi = sum(is.na(hhi)),
    missing_rsi = sum(is.na(rsi)),
    missing_pun = sum(is.na(pun)),
    missing_temperature = sum(is.na(temperature_2m)),
    missing_wind = sum(is.na(wind_speed_100m)),
    missing_radiation = sum(is.na(shortwave_radiation))
  )

qc_join_duplicates <- gme_model_panel_weather %>%
  count(date, hour, zone) %>%
  filter(n > 1)

print(qc_join_missing)

if (nrow(qc_join_duplicates) > 0) {
  warning("Duplicated datetime-zone rows found after joining weather.")
  print(qc_join_duplicates)
} else {
  message("No duplicated datetime-zone rows found after joining weather.")
}

qc_wide_duplicates <- gme_model_panel_weather_wide %>%
  count(date, hour) %>%
  filter(n > 1)

qc_long_vs_wide <- tibble(
  n_long_rows = nrow(gme_model_panel_weather),
  n_wide_rows = nrow(gme_model_panel_weather_wide),
  expected_wide_rows = n_distinct(paste(gme_model_panel_weather$date, gme_model_panel_weather$hour)),
  expected_long_rows = n_distinct(paste(gme_model_panel_weather$date, gme_model_panel_weather$hour)) * length(PHYSICAL_ZONES)
)

print(qc_wide_duplicates)
print(qc_long_vs_wide)
# ==============================================================================
# 10. SAVE OUTPUTS
# ==============================================================================

saveRDS(
  df_weather_cities,
  file.path(OUTPUT_DIR, "weather_cities_hourly.rds")
)

saveRDS(
  weather_zonal_hourly,
  file.path(OUTPUT_DIR, "weather_zonal_hourly.rds")
)

saveRDS(
  gme_model_panel_weather,
  file.path(OUTPUT_DIR, "gme_model_panel_weather_hourly.rds")
)

write_csv(
  weather_zonal_hourly,
  file.path(OUTPUT_DIR, "weather_zonal_hourly.csv")
)

write_csv(
  gme_model_panel_weather,
  file.path(OUTPUT_DIR, "gme_model_panel_weather_hourly.csv")
)


saveRDS(
  gme_model_panel_weather_wide,
  file.path(OUTPUT_DIR, "gme_model_panel_weather_wide_hourly.rds")
)

write_csv(
  gme_model_panel_weather_wide,
  file.path(OUTPUT_DIR, "gme_model_panel_weather_wide_hourly.csv")
)

message("\nWeather ingestion and join completed successfully.")
message("Final dataset saved as:")
message(file.path(OUTPUT_DIR, "gme_model_panel_weather_hourly.rds"))


# Optional Parquet outputs for Python compatibility
suppressPackageStartupMessages({
  library(arrow)
})

write_parquet(
  gme_model_panel_weather,
  file.path(OUTPUT_DIR, "gme_model_panel_weather_hourly.parquet")
)

write_parquet(
  gme_model_panel_weather_wide,
  file.path(OUTPUT_DIR, "gme_model_panel_weather_wide_hourly.parquet")
)