#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_xgboost_long_hourly.py

"""

from __future__ import annotations

import gc
import logging
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - this is a runtime dependency check.
    raise ImportError(
        "The package 'xgboost' is required to run this script. "
        "Install it in the active Python environment, for example with: "
        "pip install xgboost"
    ) from exc


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Execution modes --------------------------------------------------------------
# Keep the smoke test active by default. It writes the official prediction format
# but only runs a tiny validation subset.
RUN_SMOKE_TEST = False
RUN_FAST_VALIDATION = False
RUN_FULL_VALIDATION = False
RUN_FINAL_TEST = True

# Only one mode can run at a time.
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

# Main entities ----------------------------------------------------------------
TARGETS = ["price", "purchases", "sales"]
PHYSICAL_ZONES = ["NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD"]

# Input files ------------------------------------------------------------------
INPUT_PANEL_RDS = "data/processed/gme_model_panel_weather_hourly.rds"
INPUT_PANEL_PARQUET = "data/processed/gme_model_panel_weather_hourly.parquet"

INPUT_EVAL_RDS = "data/evaluation/eval_index_hourly.rds"
INPUT_EVAL_PARQUET = "data/evaluation/eval_index_hourly.parquet"

INPUT_NAIVE_WEEK = "data/predictions/pred_naive_week_before.parquet"
INPUT_NAIVE_WEEK_RDS = "data/predictions/pred_naive_week_before.rds"

# Output files -----------------------------------------------------------------
OUTPUT_PRED_PARQUET = "data/predictions/pred_xgboost.parquet"
OUTPUT_PRED_RDS = "data/predictions/pred_xgboost.rds"

EXPERIMENT_DIR = "experiments/xgboost_long_hourly"
CHECKPOINT_DIR = "experiments/xgboost_long_hourly/checkpoints"
LOG_DIR = "experiments/xgboost_long_hourly/logs"

FIT_LOG_FILE = "experiments/xgboost_long_hourly/logs/xgboost_fit_log.csv"
VALIDATION_SUMMARY_FILE = "experiments/xgboost_long_hourly/logs/xgboost_validation_summary.csv"
SELECTED_STRATEGY_FILE = "experiments/xgboost_long_hourly/logs/xgboost_selected_strategy_by_target.csv"
FEATURE_IMPORTANCE_FILE = "experiments/xgboost_long_hourly/logs/xgboost_feature_importance.csv"
LEARNING_CURVE_FILE = "experiments/xgboost_long_hourly/logs/xgboost_learning_curves.csv"

# Checkpoints and resume --------------------------------------------------------
ENABLE_CHECKPOINTS = True
RESUME_FROM_CHECKPOINTS = True
OVERWRITE_EXISTING_CHECKPOINTS = False

# Recalibration and windows -----------------------------------------------------
RECALIBRATION_FREQUENCY = "monthly"
WINDOW_STRATEGIES = ["rolling_12m", "rolling_24m", "expanding"]
INITIAL_TRAIN_START_DATE = "2021-01-01"

# Early stopping uses only the current training window. The external validation
# year and the test year are never used as early stopping sets.
INNER_VALIDATION_DAYS = 30
EARLY_STOPPING_ROUNDS = 50
MIN_TOTAL_OBS = 24 * 45
MIN_INNER_TRAIN_OBS = 24 * 30
MIN_INNER_VALID_OBS = 24 * 7

# Runtime ----------------------------------------------------------------------
RANDOM_STATE = 123
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))

# Smoke mode -------------------------------------------------------------------
SMOKE_TARGETS = ["price"]
SMOKE_ZONES = ["NORD", "CSUD"]
SMOKE_MAX_FORECAST_DATES = 7
SMOKE_HYPERPARAMETER_GRID = "tiny"

# Fast validation mode ----------------------------------------------------------
FAST_TARGETS = ["price", "purchases", "sales"]
FAST_ZONES = ["NORD", "CSUD", "SICI"]
FAST_MAX_VALIDATION_MONTHS = 2
FAST_MAX_FORECAST_DATES = None
FAST_HYPERPARAMETER_GRID = "small"

# Final test behaviour ----------------------------------------------------------
# Final test reads SELECTED_STRATEGY_FILE. If the file is missing, the script
# stops by default because test tuning would be leakage.
ALLOW_FINAL_FALLBACK_STRATEGY = False
FINAL_FALLBACK_WINDOW_STRATEGY = "rolling_24m"
FINAL_FALLBACK_PARAM_CONFIG = "xgb_medium_lr003"

# Weather ----------------------------------------------------------------------
# If True, delivery-day weather is treated as a weather forecast proxy. If False,
# only lagged weather variables are used.
USE_DELIVERY_DAY_WEATHER_AS_PROXY = True
WEATHER_VARIABLES = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]
WEATHER_LAGS = [24, 168]

# Own-zone and national lags ----------------------------------------------------
TARGET_LAGS = [24, 48, 168]
MARKET_LAG_VARIABLES = [
    "price",
    "purchases",
    "sales",
    "hhi",
    "rsi",
    "mti",
    "pun",
    "purchases_italy",
    "sales_italy",
    "unsold_italy",
    "purchases_external_total",
    "sales_external_total",
    "purchases_external_n_active_areas",
    "sales_external_n_active_areas",
]
MARKET_LAGS = [24, 168]

# Cross-zone lagged predictors --------------------------------------------------
INCLUDE_CROSS_ZONE_LAGS = True
CROSS_ZONE_VARIABLES = ["price", "purchases", "sales"]
CROSS_ZONE_LAGS = [24, 168]

# Hyperparameter grids ----------------------------------------------------------
SMOKE_XGB_PARAMS = {
    "xgb_smoke": {
        "max_depth": 2,
        "learning_rate": 0.10,
        "n_estimators": 200,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    }
}

XGB_PARAM_GRID = {
    "xgb_shallow_lr005": {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    "xgb_medium_lr003": {
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 1200,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
        "reg_alpha": 0.1,
    },
    "xgb_regularized_lr005": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "min_child_weight": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 20.0,
        "reg_alpha": 0.1,
    },
}

SMALL_XGB_PARAM_GRID = {
    key: value
    for key, value in XGB_PARAM_GRID.items()
    if key in ["xgb_shallow_lr005", "xgb_regularized_lr005"]
}

# Output format required by 05_evaluation_metrics.R ----------------------------
REQUIRED_PREDICTION_COLUMNS = [
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
    "y_pred",
]

# Additional columns are useful for diagnostics and are kept in the parquet file.
DIAGNOSTIC_PREDICTION_COLUMNS = [
    "window_strategy",
    "param_config",
    "recalibration_date",
]


# =============================================================================
# 2. PATHS AND LOGGING
# =============================================================================

def infer_project_root() -> Path:
    """Infer the project root assuming this script is stored in scripts/."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "scripts":
        return script_dir.parent
    return script_dir


PROJECT_ROOT = infer_project_root()


def p(path_like: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_directories() -> None:
    for folder in [
        p("data/predictions"),
        p(EXPERIMENT_DIR),
        p(CHECKPOINT_DIR),
        p(LOG_DIR),
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_directories()
    log_file = p(LOG_DIR) / "xgboost_long_hourly_run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


# =============================================================================
# 3. DATA LOADING
# =============================================================================

def check_execution_mode() -> str:
    modes = {
        "smoke": RUN_SMOKE_TEST,
        "fast_validation": RUN_FAST_VALIDATION,
        "full_validation": RUN_FULL_VALIDATION,
        "final_test": RUN_FINAL_TEST,
    }
    active = [name for name, enabled in modes.items() if enabled]

    if len(active) != 1:
        raise ValueError(
            "Exactly one execution mode must be True. Current active modes: "
            f"{active}. Set only one of RUN_SMOKE_TEST, RUN_FAST_VALIDATION, "
            "RUN_FULL_VALIDATION, RUN_FINAL_TEST to True."
        )

    return active[0]


def read_table_prefer_parquet(parquet_path: Path, rds_path: Optional[Path], object_name: str) -> pd.DataFrame:
    """Read parquet if available; otherwise try .rds via pyreadr."""
    if parquet_path.exists():
        logging.info("Reading %s from parquet: %s", object_name, parquet_path)
        return pd.read_parquet(parquet_path)

    if rds_path is not None and rds_path.exists():
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError(
                f"{object_name}: parquet file not found at {parquet_path} and pyreadr "
                f"is not installed to read {rds_path}. Export the .rds file to parquet "
                "from R or install pyreadr."
            ) from exc

        logging.info("Reading %s from RDS: %s", object_name, rds_path)
        result = pyreadr.read_r(str(rds_path))
        if len(result) == 0:
            raise ValueError(f"Could not read any object from {rds_path}")
        return next(iter(result.values()))

    raise FileNotFoundError(
        f"Could not find {object_name}. Tried parquet: {parquet_path} "
        f"and RDS: {rds_path}. Please export the .rds files to parquet from R."
    )


def read_input_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = read_table_prefer_parquet(
        p(INPUT_PANEL_PARQUET),
        p(INPUT_PANEL_RDS),
        "panel",
    )

    eval_index = read_table_prefer_parquet(
        p(INPUT_EVAL_PARQUET),
        p(INPUT_EVAL_RDS),
        "evaluation index",
    )

    naive_week = read_table_prefer_parquet(
        p(INPUT_NAIVE_WEEK),
        p(INPUT_NAIVE_WEEK_RDS),
        "weekly naive benchmark",
    )

    return panel, eval_index, naive_week


def to_naive_utc_datetime(series: pd.Series) -> pd.Series:
   
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def standardize_dates(panel: pd.DataFrame, eval_index: pd.DataFrame, naive_week: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize date/time columns used throughout the pipeline."""
    panel = panel.copy()
    eval_index = eval_index.copy()
    naive_week = naive_week.copy()

    if "datetime_model" not in panel.columns:
        raise ValueError("The panel must contain 'datetime_model'.")

    panel["datetime_model"] = to_naive_utc_datetime(panel["datetime_model"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["hour"] = panel["hour"].astype(int)
    panel["zone"] = panel["zone"].astype(str)

    eval_index["delivery_datetime_model"] = to_naive_utc_datetime(eval_index["delivery_datetime_model"])
    eval_index["forecast_date"] = pd.to_datetime(eval_index["forecast_date"]).dt.normalize()
    eval_index["delivery_date"] = pd.to_datetime(eval_index["delivery_date"]).dt.normalize()
    eval_index["hour"] = eval_index["hour"].astype(int)
    eval_index["horizon"] = eval_index["horizon"].astype(int)
    eval_index["target"] = eval_index["target"].astype(str)
    eval_index["zone"] = eval_index["zone"].astype(str)
    eval_index["split"] = eval_index["split"].astype(str)

    naive_week["delivery_datetime_model"] = to_naive_utc_datetime(naive_week["delivery_datetime_model"])
    naive_week["forecast_date"] = pd.to_datetime(naive_week["forecast_date"]).dt.normalize()
    naive_week["delivery_date"] = pd.to_datetime(naive_week["delivery_date"]).dt.normalize()
    naive_week["target"] = naive_week["target"].astype(str)
    naive_week["zone"] = naive_week["zone"].astype(str)
    naive_week["split"] = naive_week["split"].astype(str)

    return panel, eval_index, naive_week


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclic intraday features."""
    out = panel.copy()
    out["weekday"] = out["date"].dt.weekday.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)

    # The raw hour is also one-hot encoded as categorical. The sine/cosine pair
    # gives the model a smooth cyclic representation of the 24-hour profile.
    radians = 2.0 * math.pi * (out["hour"].astype(float) - 1.0) / 24.0
    out["hour_sin"] = np.sin(radians)
    out["hour_cos"] = np.cos(radians)

    return out


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add own-zone lagged features.

    Lags are computed using the already DST-normalized 24h `datetime_model` grid.
    """
    out = panel.sort_values(["zone", "datetime_model"]).copy()
    existing_lag_vars = [var for var in MARKET_LAG_VARIABLES if var in out.columns]

    for var in existing_lag_vars:
        lags = TARGET_LAGS if var in TARGETS else MARKET_LAGS
        for lag in lags:
            out[f"{var}_lag_{lag}"] = out.groupby("zone", observed=True)[var].shift(lag)

    for weather_var in WEATHER_VARIABLES:
        if weather_var not in out.columns:
            continue
        for lag in WEATHER_LAGS:
            out[f"{weather_var}_lag_{lag}"] = out.groupby("zone", observed=True)[weather_var].shift(lag)

    return out


def add_cross_zone_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged cross-zone features.

    The wide columns contain lagged values from every zone. The current zone is
    excluded later when feature columns are selected for a given target-zone model.
    """
    if not INCLUDE_CROSS_ZONE_LAGS:
        return panel

    out = panel.copy()
    new_blocks: List[pd.DataFrame] = []

    for variable in CROSS_ZONE_VARIABLES:
        for lag in CROSS_ZONE_LAGS:
            lag_col = f"{variable}_lag_{lag}"
            if lag_col not in out.columns:
                continue

            wide = (
                out[["datetime_model", "zone", lag_col]]
                .pivot_table(
                    index="datetime_model",
                    columns="zone",
                    values=lag_col,
                    aggfunc="first",
                )
                .rename(columns=lambda z: f"{variable}_lag_{lag}_{z}")
            )
            new_blocks.append(wide)

    if not new_blocks:
        return out

    cross_wide = pd.concat(new_blocks, axis=1).reset_index()
    out = out.merge(cross_wide, on="datetime_model", how="left")

    return out


def prepare_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build all feature columns once. Target-specific aliases are added later."""
    panel = add_calendar_features(panel)
    panel = add_lag_features(panel)
    panel = add_cross_zone_lag_features(panel)

    # MTI is categorical. Keep raw/lagged forms as object/string-like variables.
    for col in ["mti", "mti_lag_24", "mti_lag_168"]:
        if col in panel.columns:
            panel[col] = panel[col].astype("object")

    return panel


def get_cross_zone_features_for_zone(panel: pd.DataFrame, current_zone: str) -> List[str]:
    """Return cross-zone lag features, excluding the current zone."""
    if not INCLUDE_CROSS_ZONE_LAGS:
        return []

    cross_features: List[str] = []
    for variable in CROSS_ZONE_VARIABLES:
        for lag in CROSS_ZONE_LAGS:
            for zone in PHYSICAL_ZONES:
                if zone == current_zone:
                    continue
                col = f"{variable}_lag_{lag}_{zone}"
                if col in panel.columns:
                    cross_features.append(col)

    check_no_own_zone_cross_features(cross_features, current_zone)
    return cross_features


def check_no_own_zone_cross_features(cross_features: Sequence[str], current_zone: str) -> None:
    """Raise an error if same-zone columns are included as cross-zone features."""
    bad = [col for col in cross_features if col.endswith(f"_{current_zone}")]
    if bad:
        raise ValueError(
            f"Own-zone duplicated cross-zone features detected for zone {current_zone}: "
            f"{bad}. Cross-zone predictors must include only the other zones."
        )


def make_target_zone_frame(panel_features: pd.DataFrame, target: str, zone: str) -> pd.DataFrame:
    """Subset the panel for one target-zone equation and add target_lag aliases."""
    if target not in panel_features.columns:
        raise ValueError(f"Target column '{target}' not found in panel.")

    df = panel_features.loc[panel_features["zone"] == zone].copy()
    df["y"] = pd.to_numeric(df[target], errors="coerce")

    for lag in TARGET_LAGS:
        source_col = f"{target}_lag_{lag}"
        alias_col = f"target_lag_{lag}"
        if source_col not in df.columns:
            raise ValueError(
                f"Required same-target lag source column '{source_col}' is missing."
            )
        df[alias_col] = pd.to_numeric(df[source_col], errors="coerce")

    return df


def get_feature_columns(panel_features: pd.DataFrame, target: str, zone: str) -> Tuple[List[str], List[str]]:
    """
    Define numeric and categorical predictors for a target-zone model.

    `hour`, `weekday`, and `month` are one-hot encoded. The sine/cosine version
    of hour is also included as numeric to help capture cyclic intraday variation.
    """
    categorical_features = ["hour", "weekday", "month"]

    # MTI is categorical by definition. Its lagged values must not be treated as
    # numerical market indicators.
    for col in ["mti_lag_24", "mti_lag_168"]:
        if col in panel_features.columns:
            categorical_features.append(col)

    numeric_features = [
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "target_lag_24",
        "target_lag_48",
        "target_lag_168",
    ]

    own_lag_candidates = []
    for variable in MARKET_LAG_VARIABLES:
        if variable == "mti":
            continue
        
        if variable == target:
            continue

        for lag in MARKET_LAGS:
            own_lag_candidates.append(f"{variable}_lag_{lag}")

    numeric_features.extend([col for col in own_lag_candidates if col in panel_features.columns])

    if USE_DELIVERY_DAY_WEATHER_AS_PROXY:
        # These variables are treated as day-ahead weather forecast proxies.
        numeric_features.extend([col for col in WEATHER_VARIABLES if col in panel_features.columns])
    else:
        lagged_weather = []
        for variable in WEATHER_VARIABLES:
            for lag in WEATHER_LAGS:
                lagged_weather.append(f"{variable}_lag_{lag}")
        numeric_features.extend([col for col in lagged_weather if col in panel_features.columns])

    numeric_features.extend(get_cross_zone_features_for_zone(panel_features, zone))

    # Remove duplicates while preserving order.
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))

    return numeric_features, categorical_features


# =============================================================================
# 5. RECALIBRATION BLOCKS AND TRAINING WINDOWS
# =============================================================================

@dataclass(frozen=True)
class ModeSettings:
    mode: str
    split: str
    targets: List[str]
    zones: List[str]
    window_strategies: List[str]
    param_grid: Dict[str, Dict[str, Any]]
    max_forecast_dates: Optional[int] = None
    max_validation_months: Optional[int] = None


@dataclass(frozen=True)
class RecalibrationBlock:
    split: str
    block_id: str
    recalibration_date: pd.Timestamp
    forecast_dates: List[pd.Timestamp]


def get_mode_settings(mode: str) -> ModeSettings:
    if mode == "smoke":
        return ModeSettings(
            mode=mode,
            split="validation",
            targets=SMOKE_TARGETS,
            zones=SMOKE_ZONES,
            window_strategies=["rolling_12m"],
            param_grid=SMOKE_XGB_PARAMS,
            max_forecast_dates=SMOKE_MAX_FORECAST_DATES,
            max_validation_months=None,
        )

    if mode == "fast_validation":
        grid = SMALL_XGB_PARAM_GRID if FAST_HYPERPARAMETER_GRID == "small" else XGB_PARAM_GRID
        return ModeSettings(
            mode=mode,
            split="validation",
            targets=FAST_TARGETS,
            zones=FAST_ZONES,
            window_strategies=WINDOW_STRATEGIES,
            param_grid=grid,
            max_forecast_dates=FAST_MAX_FORECAST_DATES,
            max_validation_months=FAST_MAX_VALIDATION_MONTHS,
        )

    if mode == "full_validation":
        return ModeSettings(
            mode=mode,
            split="validation",
            targets=TARGETS,
            zones=PHYSICAL_ZONES,
            window_strategies=WINDOW_STRATEGIES,
            param_grid=XGB_PARAM_GRID,
        )

    if mode == "final_test":
        selected = read_selected_strategy_or_fallback()
        # Final mode always predicts all zones. The selected table can be
        # target-specific, so window/param pairs are resolved later per target.
        return ModeSettings(
            mode=mode,
            split="test",
            targets=TARGETS,
            zones=PHYSICAL_ZONES,
            window_strategies=sorted(selected["window_strategy"].unique().tolist()),
            param_grid={name: XGB_PARAM_GRID.get(name, SMOKE_XGB_PARAMS.get(name, {}))
                        for name in selected["param_config"].unique()},
        )

    raise ValueError(f"Unknown mode: {mode}")


def limit_eval_index(eval_index: pd.DataFrame, settings: ModeSettings) -> pd.DataFrame:
    """Limit targets, zones and forecast dates according to the active mode."""
    data = eval_index.loc[
        (eval_index["split"] == settings.split)
        & (eval_index["target"].isin(settings.targets))
        & (eval_index["zone"].isin(settings.zones))
    ].copy()

    if data.empty:
        raise ValueError(
            f"No evaluation rows found for split={settings.split}, "
            f"targets={settings.targets}, zones={settings.zones}."
        )

    allowed_dates = sorted(data["forecast_date"].drop_duplicates().tolist())

    if settings.max_validation_months is not None:
        month_periods = pd.Series(allowed_dates).dt.to_period("M")
        keep_months = sorted(month_periods.unique())[: settings.max_validation_months]
        allowed_dates = [
            dt for dt in allowed_dates
            if pd.Timestamp(dt).to_period("M") in keep_months
        ]

    if settings.max_forecast_dates is not None:
        allowed_dates = allowed_dates[: settings.max_forecast_dates]

    data = data.loc[data["forecast_date"].isin(allowed_dates)].copy()
    return data


def get_recalibration_blocks(eval_subset: pd.DataFrame, split: str) -> List[RecalibrationBlock]:
    """Group forecast dates into recalibration blocks."""
    forecast_dates = sorted(pd.to_datetime(eval_subset["forecast_date"].drop_duplicates()).tolist())
    if not forecast_dates:
        return []

    if RECALIBRATION_FREQUENCY != "monthly":
        raise ValueError("Only monthly recalibration is currently implemented.")

    blocks: List[RecalibrationBlock] = []
    by_month: Dict[str, List[pd.Timestamp]] = {}

    for dt in forecast_dates:
        month_id = pd.Timestamp(dt).strftime("%Y-%m")
        by_month.setdefault(month_id, []).append(pd.Timestamp(dt))

    for month_id, dates in by_month.items():
        dates_sorted = sorted(dates)
        blocks.append(
            RecalibrationBlock(
                split=split,
                block_id=month_id,
                recalibration_date=dates_sorted[0],
                forecast_dates=dates_sorted,
            )
        )

    return blocks


def get_training_window(df_tz: pd.DataFrame, block: RecalibrationBlock, window_strategy: str) -> pd.DataFrame:
    """
    Select the training window for one recalibration block.

    Training uses only observations with date <= block.recalibration_date, i.e.
    data available before the first delivery day of the block.
    """
    train_end_date = block.recalibration_date.normalize()

    if window_strategy == "expanding":
        train_start_date = pd.Timestamp(INITIAL_TRAIN_START_DATE)
    elif window_strategy == "rolling_12m":
        train_start_date = train_end_date - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    elif window_strategy == "rolling_24m":
        train_start_date = train_end_date - pd.DateOffset(months=24) + pd.Timedelta(days=1)
    else:
        raise ValueError(f"Unknown window strategy: {window_strategy}")

    train = df_tz.loc[
        (df_tz["date"] >= train_start_date)
        & (df_tz["date"] <= train_end_date)
    ].copy()

    train = train.loc[~train["y"].isna()].copy()
    return train


def split_inner_validation(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Split the current training window into inner-train and inner-validation."""
    if train_df.empty:
        return train_df, None

    train_end_date = train_df["date"].max()
    inner_valid_start = train_end_date - pd.Timedelta(days=INNER_VALIDATION_DAYS - 1)

    inner_train = train_df.loc[train_df["date"] < inner_valid_start].copy()
    inner_valid = train_df.loc[train_df["date"] >= inner_valid_start].copy()

    enough_data = (
        len(train_df) >= MIN_TOTAL_OBS
        and len(inner_train) >= MIN_INNER_TRAIN_OBS
        and len(inner_valid) >= MIN_INNER_VALID_OBS
    )

    if not enough_data:
        return train_df.copy(), None

    return inner_train, inner_valid


# =============================================================================
# 6. DESIGN MATRICES
# =============================================================================

@dataclass
class DesignMatrices:
    X_train: sparse.csr_matrix
    y_train: np.ndarray
    X_inner_valid: Optional[sparse.csr_matrix]
    y_inner_valid: Optional[np.ndarray]
    X_pred: sparse.csr_matrix
    feature_names: List[str]
    numeric_features_used: List[str]
    categorical_features_used: List[str]


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _drop_all_missing_columns(train_df: pd.DataFrame, columns: Sequence[str], kind: str) -> List[str]:
    """Drop columns that are entirely missing in the current training window."""
    kept: List[str] = []
    dropped: List[str] = []

    for col in columns:
        if col not in train_df.columns:
            continue
        if train_df[col].notna().sum() == 0:
            dropped.append(col)
        else:
            kept.append(col)

    if dropped:
        warnings.warn(
            f"Dropping {kind} predictors that are entirely missing in the current "
            f"training window: {dropped}. No future data are used for imputation.",
            RuntimeWarning,
        )

    return kept


def _fit_transform_numeric(
    train_df: pd.DataFrame,
    inner_valid_df: Optional[pd.DataFrame],
    pred_df: pd.DataFrame,
    numeric_features: Sequence[str],
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix], sparse.csr_matrix, List[str]]:
    numeric_features_used = _drop_all_missing_columns(train_df, numeric_features, "numeric")

    if not numeric_features_used:
        n_train = len(train_df)
        n_pred = len(pred_df)
        n_valid = 0 if inner_valid_df is None else len(inner_valid_df)
        return (
            sparse.csr_matrix((n_train, 0)),
            None if inner_valid_df is None else sparse.csr_matrix((n_valid, 0)),
            sparse.csr_matrix((n_pred, 0)),
            [],
        )

    imputer = SimpleImputer(strategy="median")

    train_num = train_df[numeric_features_used].apply(pd.to_numeric, errors="coerce")
    pred_num = pred_df[numeric_features_used].apply(pd.to_numeric, errors="coerce")

    X_train = sparse.csr_matrix(imputer.fit_transform(train_num))
    X_pred = sparse.csr_matrix(imputer.transform(pred_num))

    X_valid = None
    if inner_valid_df is not None:
        valid_num = inner_valid_df[numeric_features_used].apply(pd.to_numeric, errors="coerce")
        X_valid = sparse.csr_matrix(imputer.transform(valid_num))

    return X_train, X_valid, X_pred, numeric_features_used


def _fit_transform_categorical(
    train_df: pd.DataFrame,
    inner_valid_df: Optional[pd.DataFrame],
    pred_df: pd.DataFrame,
    categorical_features: Sequence[str],
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix], sparse.csr_matrix, List[str], List[str]]:
    categorical_features_used = _drop_all_missing_columns(train_df, categorical_features, "categorical")

    if not categorical_features_used:
        n_train = len(train_df)
        n_pred = len(pred_df)
        n_valid = 0 if inner_valid_df is None else len(inner_valid_df)
        return (
            sparse.csr_matrix((n_train, 0)),
            None if inner_valid_df is None else sparse.csr_matrix((n_valid, 0)),
            sparse.csr_matrix((n_pred, 0)),
            [],
            [],
        )

    mode_values: Dict[str, str] = {}
    for col in categorical_features_used:
        mode = train_df[col].dropna().astype(str).mode()
        if mode.empty:
            # This should not happen because all-missing columns were dropped,
            # but keep a documented fallback for safety.
            warnings.warn(
                f"Categorical feature {col} has no valid mode after dropping NA. "
                "Using 'unknown_past' as fallback based only on the training window.",
                RuntimeWarning,
            )
            mode_values[col] = "unknown_past"
        else:
            mode_values[col] = str(mode.iloc[0])

    def prepare_cat(df: pd.DataFrame) -> pd.DataFrame:
        out = df[categorical_features_used].copy()
        for col in categorical_features_used:
            out[col] = out[col].astype("object")
            out[col] = out[col].where(out[col].notna(), mode_values[col])
            out[col] = out[col].astype(str)
        return out

    train_cat = prepare_cat(train_df)
    pred_cat = prepare_cat(pred_df)

    encoder = _make_one_hot_encoder()
    X_train = encoder.fit_transform(train_cat)
    X_pred = encoder.transform(pred_cat)

    X_valid = None
    if inner_valid_df is not None:
        valid_cat = prepare_cat(inner_valid_df)
        X_valid = encoder.transform(valid_cat)

    try:
        cat_feature_names = encoder.get_feature_names_out(categorical_features_used).tolist()
    except AttributeError:
        cat_feature_names = encoder.get_feature_names(categorical_features_used).tolist()

    return (
        sparse.csr_matrix(X_train),
        None if X_valid is None else sparse.csr_matrix(X_valid),
        sparse.csr_matrix(X_pred),
        categorical_features_used,
        cat_feature_names,
    )


def build_design_matrices(
    train_df: pd.DataFrame,
    inner_valid_df: Optional[pd.DataFrame],
    pred_df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> DesignMatrices:
    """
    Build aligned train, inner-validation and prediction matrices.

    All imputers and encoders are fitted only on the current training window.
    """
    X_train_num, X_valid_num, X_pred_num, numeric_used = _fit_transform_numeric(
        train_df,
        inner_valid_df,
        pred_df,
        numeric_features,
    )

    X_train_cat, X_valid_cat, X_pred_cat, categorical_used, cat_feature_names = _fit_transform_categorical(
        train_df,
        inner_valid_df,
        pred_df,
        categorical_features,
    )

    X_train = sparse.hstack([X_train_num, X_train_cat], format="csr")
    X_pred = sparse.hstack([X_pred_num, X_pred_cat], format="csr")

    X_inner_valid = None
    if inner_valid_df is not None:
        if X_valid_num is None or X_valid_cat is None:
            raise RuntimeError("Internal error: expected validation matrices.")
        X_inner_valid = sparse.hstack([X_valid_num, X_valid_cat], format="csr")

    feature_names = list(numeric_used) + list(cat_feature_names)

    return DesignMatrices(
        X_train=X_train,
        y_train=train_df["y"].to_numpy(dtype=float),
        X_inner_valid=X_inner_valid,
        y_inner_valid=None if inner_valid_df is None else inner_valid_df["y"].to_numpy(dtype=float),
        X_pred=X_pred,
        feature_names=feature_names,
        numeric_features_used=list(numeric_used),
        categorical_features_used=list(categorical_used),
    )


# =============================================================================
# 7. MODEL FITTING AND PREDICTION
# =============================================================================

def make_xgb_model(params: Dict[str, Any]) -> XGBRegressor:
    """Create an XGBRegressor with project-wide defaults."""
    model_params = dict(params)
    model_params.update(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "eval_metric": "mae",
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
            "verbosity": 0,
        }
    )
    return XGBRegressor(**model_params)


def fit_xgboost_model(
    params: Dict[str, Any],
    design: DesignMatrices,
) -> Tuple[XGBRegressor, Optional[int], str]:
    """
    Fit XGBoost using the scikit-learn API.

    Early stopping is activated through the estimator parameter
    early_stopping_rounds. The monitored set is the inner validation set, which
    is built only from the current training window.
    """
    has_valid = (
        design.X_inner_valid is not None
        and design.y_inner_valid is not None
        and design.X_inner_valid.shape[0] > 0
    )

    model_params = dict(params)

    if has_valid:
        model_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    model = make_xgb_model(model_params)

    if has_valid:
        model.fit(
            design.X_train,
            design.y_train,
            eval_set=[
                (design.X_train, design.y_train),
                (design.X_inner_valid, design.y_inner_valid),
            ],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        return model, best_iteration, "fit_with_early_stopping"

    model.fit(
        design.X_train,
        design.y_train,
        verbose=False,
    )
    best_iteration = getattr(model, "best_iteration", None)
    return model, best_iteration, "fit_without_early_stopping_no_inner_valid"

def extract_feature_importance(
    model: XGBRegressor,
    feature_names: Sequence[str],
    target: str,
    zone: str,
    window_strategy: str,
    param_config: str,
) -> pd.DataFrame:
    """Extract built-in XGBoost importance. Diagnostic only, not causal."""
    booster = model.get_booster()
    rows: Dict[str, Dict[str, Any]] = {}

    for importance_type, output_name in [
        ("gain", "importance_gain"),
        ("weight", "importance_weight"),
        ("cover", "importance_cover"),
    ]:
        scores = booster.get_score(importance_type=importance_type)
        for raw_feature, value in scores.items():
            feature = map_xgb_feature_name(raw_feature, feature_names)
            rows.setdefault(feature, {})[output_name] = float(value)

    if not rows:
        return pd.DataFrame(
            columns=[
                "target",
                "zone",
                "window_strategy",
                "param_config",
                "feature",
                "importance_gain",
                "importance_weight",
                "importance_cover",
            ]
        )

    out = pd.DataFrame(
        [
            {
                "target": target,
                "zone": zone,
                "window_strategy": window_strategy,
                "param_config": param_config,
                "feature": feature,
                "importance_gain": values.get("importance_gain", 0.0),
                "importance_weight": values.get("importance_weight", 0.0),
                "importance_cover": values.get("importance_cover", 0.0),
            }
            for feature, values in rows.items()
        ]
    )

    return out.sort_values(
        ["target", "zone", "window_strategy", "param_config", "importance_gain"],
        ascending=[True, True, True, True, False],
    )


def map_xgb_feature_name(raw_feature: str, feature_names: Sequence[str]) -> str:
    """Map XGBoost f0/f1 names back to human-readable feature names."""
    match = re.fullmatch(r"f(\d+)", raw_feature)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(feature_names):
            return feature_names[idx]
    return raw_feature


# =============================================================================
# 8. CHECKPOINTS AND LOGS
# =============================================================================

def safe_name(value: Any) -> str:
    """Create a filesystem-safe string."""
    text = str(value)
    text = text.replace(os.sep, "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def checkpoint_path(
    split: str,
    target: str,
    zone: str,
    window_strategy: str,
    param_config: str,
    block_id: str,
) -> Path:
    filename = (
        f"checkpoint_{safe_name(split)}_{safe_name(target)}_{safe_name(zone)}_"
        f"{safe_name(window_strategy)}_{safe_name(param_config)}_{safe_name(block_id)}.parquet"
    )
    return p(CHECKPOINT_DIR) / filename


def save_checkpoint(path: Path, predictions: pd.DataFrame) -> None:
    if not ENABLE_CHECKPOINTS:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)


def load_checkpoint(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def append_csv(path: Path, row_or_df: Dict[str, Any] | pd.DataFrame) -> None:
    """Append one row or a DataFrame to a CSV log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row_or_df]) if isinstance(row_or_df, dict) else row_or_df
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_fit_log(record: Dict[str, Any]) -> None:
    append_csv(p(FIT_LOG_FILE), record)


def append_feature_importance(importance: pd.DataFrame) -> None:
    if importance.empty:
        return
    append_csv(p(FEATURE_IMPORTANCE_FILE), importance)

def make_learning_curve_log(
    model: XGBRegressor,
    target: str,
    zone: str,
    split: str,
    window_strategy: str,
    param_config: str,
    block: RecalibrationBlock,
) -> pd.DataFrame:
    """
    Extract the XGBoost learning curve from evals_result().

    validation_0 corresponds to the inner training set.
    validation_1 corresponds to the inner validation set used for early stopping.
    """
    try:
        evals_result = model.evals_result()
    except Exception:
        return pd.DataFrame()

    if not evals_result:
        return pd.DataFrame()

    train_metrics = evals_result.get("validation_0", {})
    valid_metrics = evals_result.get("validation_1", {})

    metric_name = "mae"

    if metric_name not in train_metrics:
        if train_metrics:
            metric_name = next(iter(train_metrics.keys()))
        else:
            return pd.DataFrame()

    train_values = train_metrics.get(metric_name, [])
    valid_values = valid_metrics.get(metric_name, [])

    n_rounds = max(len(train_values), len(valid_values))
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)

    rows = []

    for iteration in range(n_rounds):
        rows.append(
            {
                "target": target,
                "zone": zone,
                "split": split,
                "window_strategy": window_strategy,
                "param_config": param_config,
                "recalibration_date": block.recalibration_date.date().isoformat(),
                "block_id": block.block_id,
                "iteration": iteration,
                "metric": metric_name,
                "inner_train_metric": (
                    train_values[iteration]
                    if iteration < len(train_values)
                    else np.nan
                ),
                "inner_valid_metric": (
                    valid_values[iteration]
                    if iteration < len(valid_values)
                    else np.nan
                ),
                "best_iteration": (
                    np.nan
                    if best_iteration is None
                    else best_iteration
                ),
                "best_score": (
                    np.nan
                    if best_score is None
                    else best_score
                ),
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            }
        )

    return pd.DataFrame(rows)


def append_learning_curve(curve: pd.DataFrame) -> None:
    if curve.empty:
        return
    append_csv(p(LEARNING_CURVE_FILE), curve)

# =============================================================================
# 9. PREDICTION BLOCK
# =============================================================================

def get_prediction_rows_for_block(
    eval_subset: pd.DataFrame,
    target: str,
    zone: str,
    block: RecalibrationBlock,
) -> pd.DataFrame:
    rows = eval_subset.loc[
        (eval_subset["target"] == target)
        & (eval_subset["zone"] == zone)
        & (eval_subset["forecast_date"].isin(block.forecast_dates))
    ].copy()

    return rows.sort_values("delivery_datetime_model")


def build_prediction_frame(
    df_tz: pd.DataFrame,
    eval_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach engineered features to evaluation rows."""
    feature_rows = df_tz.merge(
        eval_rows[
            [
                "target",
                "zone",
                "split",
                "forecast_date",
                "delivery_datetime_model",
                "delivery_date",
                "hour",
                "horizon",
                "y_true",
            ]
        ],
        left_on=["zone", "datetime_model"],
        right_on=["zone", "delivery_datetime_model"],
        how="inner",
        suffixes=("", "_eval"),
    )

    if len(feature_rows) != len(eval_rows):
        raise ValueError(
            "Feature/evaluation merge lost rows. "
            f"Expected {len(eval_rows)}, got {len(feature_rows)}."
        )

    return feature_rows.sort_values("delivery_datetime_model")


def get_model_name(mode: str, window_strategy: str, param_config: str) -> str:
    """Return the model name written to the prediction file."""
    if mode == "final_test":
        return "xgboost"
    if mode == "smoke":
        return "xgboost_smoke"
    return f"xgboost__{window_strategy}__{param_config}"


def predict_block(
    panel_features: pd.DataFrame,
    eval_subset: pd.DataFrame,
    settings: ModeSettings,
    target: str,
    zone: str,
    window_strategy: str,
    param_config: str,
    params: Dict[str, Any],
    block: RecalibrationBlock,
) -> pd.DataFrame:
    """Fit one target-zone model for one recalibration block and predict it."""
    ckpt = checkpoint_path(
        split=block.split,
        target=target,
        zone=zone,
        window_strategy=window_strategy,
        param_config=param_config,
        block_id=block.block_id,
    )

    if (
        ENABLE_CHECKPOINTS
        and RESUME_FROM_CHECKPOINTS
        and ckpt.exists()
        and not OVERWRITE_EXISTING_CHECKPOINTS
    ):
        loaded = load_checkpoint(ckpt)
        append_fit_log(
            {
                "target": target,
                "zone": zone,
                "split": block.split,
                "window_strategy": window_strategy,
                "param_config": param_config,
                "recalibration_date": block.recalibration_date.date().isoformat(),
                "block_start": min(block.forecast_dates).date().isoformat(),
                "block_end": max(block.forecast_dates).date().isoformat(),
                "n_train": np.nan,
                "n_inner_train": np.nan,
                "n_inner_valid": np.nan,
                "n_pred": len(loaded),
                "n_features": np.nan,
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
                "best_iteration": np.nan,
                "status": "checkpoint_loaded",
                "error_message": "",
            }
        )
        return loaded

    df_tz = make_target_zone_frame(panel_features, target, zone)
    train_df = get_training_window(df_tz, block, window_strategy)
    eval_rows = get_prediction_rows_for_block(eval_subset, target, zone, block)

    if eval_rows.empty:
        raise ValueError(f"No prediction rows for {target}-{zone}-{block.block_id}")

    pred_df = build_prediction_frame(df_tz, eval_rows)

    numeric_features, categorical_features = get_feature_columns(panel_features, target, zone)

    if len(train_df) < MIN_TOTAL_OBS:
        raise ValueError(
            f"Not enough training observations for {target}-{zone}-{window_strategy}-"
            f"{block.block_id}: {len(train_df)} < {MIN_TOTAL_OBS}"
        )

    inner_train_df, inner_valid_df = split_inner_validation(train_df)

    fit_start = time.time()
    status = "started"
    error_message = ""
    best_iteration: Optional[int] = None
    feature_importance = pd.DataFrame()

    try:
        design = build_design_matrices(
            train_df=inner_train_df,
            inner_valid_df=inner_valid_df,
            pred_df=pred_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        model, best_iteration, status = fit_xgboost_model(params, design)
        fit_seconds = time.time() - fit_start

        learning_curve = make_learning_curve_log(
            model=model,
            target=target,
            zone=zone,
            split=block.split,
            window_strategy=window_strategy,
            param_config=param_config,
            block=block,
        )
        append_learning_curve(learning_curve)

        pred_start = time.time()
        y_pred = model.predict(design.X_pred)
        predict_seconds = time.time() - pred_start

        feature_importance = extract_feature_importance(
            model=model,
            feature_names=design.feature_names,
            target=target,
            zone=zone,
            window_strategy=window_strategy,
            param_config=param_config,
        )

        predictions = pred_df[
            [
                "target",
                "zone",
                "split",
                "forecast_date",
                "delivery_datetime_model",
                "delivery_date",
                "hour_eval",
                "horizon",
                "y_true",
            ]
        ].copy()

        # Depending on the merge, the evaluation hour may be called hour_eval
        # because the feature frame also contains an `hour` predictor.
        predictions = predictions.rename(columns={"hour_eval": "hour"})
        predictions["model"] = get_model_name(settings.mode, window_strategy, param_config)
        predictions["y_pred"] = y_pred.astype(float)
        predictions["window_strategy"] = window_strategy
        predictions["param_config"] = param_config
        predictions["recalibration_date"] = block.recalibration_date.normalize()

        predictions = predictions[
            REQUIRED_PREDICTION_COLUMNS + DIAGNOSTIC_PREDICTION_COLUMNS
        ].sort_values(["target", "zone", "delivery_datetime_model"])

        save_checkpoint(ckpt, predictions)
        append_feature_importance(feature_importance)

    except Exception as exc:
        fit_seconds = time.time() - fit_start
        predict_seconds = 0.0
        status = "failed"
        error_message = repr(exc)

        append_fit_log(
            {
                "target": target,
                "zone": zone,
                "split": block.split,
                "window_strategy": window_strategy,
                "param_config": param_config,
                "recalibration_date": block.recalibration_date.date().isoformat(),
                "block_start": min(block.forecast_dates).date().isoformat(),
                "block_end": max(block.forecast_dates).date().isoformat(),
                "n_train": len(train_df) if "train_df" in locals() else np.nan,
                "n_inner_train": len(inner_train_df) if "inner_train_df" in locals() else np.nan,
                "n_inner_valid": 0 if "inner_valid_df" not in locals() or inner_valid_df is None else len(inner_valid_df),
                "n_pred": len(eval_rows) if "eval_rows" in locals() else np.nan,
                "n_features": np.nan,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "best_iteration": np.nan,
                "status": status,
                "error_message": error_message,
            }
        )
        raise

    append_fit_log(
        {
            "target": target,
            "zone": zone,
            "split": block.split,
            "window_strategy": window_strategy,
            "param_config": param_config,
            "recalibration_date": block.recalibration_date.date().isoformat(),
            "block_start": min(block.forecast_dates).date().isoformat(),
            "block_end": max(block.forecast_dates).date().isoformat(),
            "n_train": len(train_df),
            "n_inner_train": len(inner_train_df),
            "n_inner_valid": 0 if inner_valid_df is None else len(inner_valid_df),
            "n_pred": len(predictions),
            "n_features": design.X_train.shape[1],
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "best_iteration": np.nan if best_iteration is None else best_iteration,
            "status": status,
            "error_message": error_message,
        }
    )

    del df_tz, train_df, inner_train_df, inner_valid_df, pred_df
    gc.collect()

    return predictions


# =============================================================================
# 10. VALIDATION METRICS AND MODEL SELECTION
# =============================================================================

def compute_validation_metrics(predictions: pd.DataFrame, naive_week: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE, RMSE and rMAE against the weekly naive benchmark."""
    if predictions.empty:
        raise ValueError("Cannot compute validation metrics from empty predictions.")

    required_naive = ["target", "zone", "split", "delivery_datetime_model", "y_pred"]
    missing_naive = set(required_naive) - set(naive_week.columns)
    if missing_naive:
        raise ValueError(f"Naive benchmark is missing columns: {sorted(missing_naive)}")

    naive = naive_week[required_naive].rename(columns={"y_pred": "y_pred_naive_week"})

    merged = predictions.merge(
        naive,
        on=["target", "zone", "split", "delivery_datetime_model"],
        how="left",
    )

    merged["abs_error"] = (merged["y_true"] - merged["y_pred"]).abs()
    merged["squared_error"] = (merged["y_true"] - merged["y_pred"]) ** 2
    merged["abs_error_naive"] = (merged["y_true"] - merged["y_pred_naive_week"]).abs()

    group_cols = ["target", "model", "window_strategy", "param_config"]

    summary = (
        merged.dropna(subset=["y_true", "y_pred"])
        .groupby(group_cols, dropna=False)
        .agg(
            n=("y_pred", "size"),
            MAE=("abs_error", "mean"),
            RMSE=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            naive_week_MAE=("abs_error_naive", "mean"),
        )
        .reset_index()
    )

    summary["rMAE"] = summary["MAE"] / summary["naive_week_MAE"]
    summary = summary.rename(
        columns={
            "MAE": "validation_MAE",
            "RMSE": "validation_RMSE",
            "rMAE": "validation_rMAE",
        }
    )

    return summary.sort_values(["target", "validation_rMAE", "validation_MAE"])


def select_best_strategy(validation_summary: pd.DataFrame) -> pd.DataFrame:
    """Select the best strategy by target using validation rMAE."""
    needed = {"target", "window_strategy", "param_config", "validation_MAE", "validation_RMSE", "validation_rMAE"}
    missing = needed - set(validation_summary.columns)
    if missing:
        raise ValueError(f"Validation summary missing required columns: {sorted(missing)}")

    selected = (
        validation_summary.sort_values(["target", "validation_rMAE", "validation_MAE"])
        .groupby("target", as_index=False)
        .first()
    )

    selected = selected[
        [
            "target",
            "window_strategy",
            "param_config",
            "validation_MAE",
            "validation_RMSE",
            "validation_rMAE",
        ]
    ]

    return selected


def read_selected_strategy_or_fallback() -> pd.DataFrame:
    """Read selected validation strategy for final test."""
    selected_path = p(SELECTED_STRATEGY_FILE)

    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        required = {"target", "window_strategy", "param_config"}
        missing = required - set(selected.columns)
        if missing:
            raise ValueError(
                f"Selected strategy file exists but is missing columns: {sorted(missing)}"
            )
        return selected

    if not ALLOW_FINAL_FALLBACK_STRATEGY:
        raise FileNotFoundError(
            f"Final test requires selected validation strategies at {selected_path}. "
            "Run RUN_FULL_VALIDATION=True first, or set "
            "ALLOW_FINAL_FALLBACK_STRATEGY=True explicitly."
        )

    warnings.warn(
        "Using explicit final fallback strategy. This is acceptable only for debugging; "
        "the final thesis comparison should use validation-selected strategies.",
        RuntimeWarning,
    )

    return pd.DataFrame(
        {
            "target": TARGETS,
            "window_strategy": [FINAL_FALLBACK_WINDOW_STRATEGY] * len(TARGETS),
            "param_config": [FINAL_FALLBACK_PARAM_CONFIG] * len(TARGETS),
            "validation_MAE": [np.nan] * len(TARGETS),
            "validation_RMSE": [np.nan] * len(TARGETS),
            "validation_rMAE": [np.nan] * len(TARGETS),
        }
    )


def get_final_strategy_for_target(target: str, selected: pd.DataFrame) -> Tuple[str, str]:
    row = selected.loc[selected["target"] == target]
    if row.empty:
        raise ValueError(f"No selected final strategy found for target '{target}'.")
    return str(row.iloc[0]["window_strategy"]), str(row.iloc[0]["param_config"])


# =============================================================================
# 11. FINAL OUTPUTS AND QUALITY CHECKS
# =============================================================================

def quality_check_predictions(predictions: pd.DataFrame, mode: str) -> None:
    """Run critical checks before saving predictions."""
    missing_cols = set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_cols:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing_cols)}")

    duplicated = predictions.duplicated(
        subset=["model", "target", "zone", "delivery_datetime_model"]
    )
    if duplicated.any():
        examples = predictions.loc[
            duplicated,
            ["model", "target", "zone", "delivery_datetime_model"],
        ].head(10)
        raise ValueError(f"Duplicated prediction rows found. Examples:\n{examples}")

    horizons = sorted(predictions["horizon"].dropna().unique().tolist())
    bad_horizons = [h for h in horizons if h not in list(range(1, 25))]
    if bad_horizons:
        raise ValueError(f"Unexpected horizons found: {bad_horizons}. Expected 1,...,24.")

    if predictions["y_true"].isna().any():
        n_missing = int(predictions["y_true"].isna().sum())
        raise ValueError(f"Missing y_true values found in predictions: {n_missing}")

    allowed_splits = {"validation", "test"}
    observed_splits = set(predictions["split"].astype(str).unique())
    if not observed_splits.issubset(allowed_splits):
        raise ValueError(
            f"Predictions contain splits outside validation/test: {observed_splits}"
        )

    if mode == "final_test" and observed_splits != {"test"}:
        raise ValueError(
            f"Final test output must contain only split='test'. Found: {observed_splits}"
        )

    n = len(predictions)
    missing_pred = int(predictions["y_pred"].isna().sum())
    pct_missing = 100 * missing_pred / max(1, n)
    logging.info(
        "Prediction quality check: n=%s, missing y_pred=%s (%.3f%%)",
        n,
        missing_pred,
        pct_missing,
    )

    logging.info("Observed horizons: %s", horizons)
    logging.info("Observed splits: %s", sorted(observed_splits))


def save_final_predictions(predictions: pd.DataFrame) -> None:
    """Save predictions in parquet and optionally RDS."""
    output_parquet = p(OUTPUT_PRED_PARQUET)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    predictions = predictions.sort_values(
        ["model", "target", "zone", "delivery_datetime_model"]
    ).reset_index(drop=True)

    predictions.to_parquet(output_parquet, index=False)
    logging.info("Saved predictions to %s", output_parquet)

    try:
        import pyreadr

        output_rds = p(OUTPUT_PRED_RDS)
        pyreadr.write_rds(str(output_rds), predictions)
        logging.info("Saved predictions to %s", output_rds)
    except Exception as exc:
        logging.warning("Could not save RDS output. Parquet was saved. Reason: %s", exc)


def save_validation_outputs(predictions: pd.DataFrame, naive_week: pd.DataFrame, mode: str) -> None:
    """Save validation summary and selected strategy when applicable."""
    if mode not in {"fast_validation", "full_validation", "smoke"}:
        return

    try:
        summary = compute_validation_metrics(predictions, naive_week)
        summary.to_csv(p(VALIDATION_SUMMARY_FILE), index=False)
        logging.info("Saved validation summary to %s", p(VALIDATION_SUMMARY_FILE))

        if mode in {"fast_validation", "full_validation"}:
            selected = select_best_strategy(summary)
            selected.to_csv(p(SELECTED_STRATEGY_FILE), index=False)
            logging.info("Saved selected strategy to %s", p(SELECTED_STRATEGY_FILE))

    except Exception as exc:
        logging.warning("Validation summary could not be computed: %s", exc)


# =============================================================================
# 12. MAIN EXPERIMENT LOOP
# =============================================================================

def resolve_param_grid_for_mode(settings: ModeSettings, selected: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Resolve parameter configs. Final mode uses selected configs by target."""
    if settings.mode != "final_test":
        return settings.param_grid

    if selected is None:
        selected = read_selected_strategy_or_fallback()

    configs = selected["param_config"].drop_duplicates().tolist()
    out: Dict[str, Dict[str, Any]] = {}
    for name in configs:
        if name in XGB_PARAM_GRID:
            out[name] = XGB_PARAM_GRID[name]
        elif name in SMOKE_XGB_PARAMS:
            out[name] = SMOKE_XGB_PARAMS[name]
        elif name in SMALL_XGB_PARAM_GRID:
            out[name] = SMALL_XGB_PARAM_GRID[name]
        else:
            raise ValueError(
                f"Parameter configuration '{name}' was selected but is not defined "
                "in XGB_PARAM_GRID."
            )
    return out


def run_experiment(
    panel_features: pd.DataFrame,
    eval_index: pd.DataFrame,
    naive_week: pd.DataFrame,
    settings: ModeSettings,
) -> pd.DataFrame:
    eval_subset = limit_eval_index(eval_index, settings)
    blocks = get_recalibration_blocks(eval_subset, settings.split)

    if not blocks:
        raise ValueError("No recalibration blocks were created.")

    logging.info(
        "Running mode=%s | split=%s | targets=%s | zones=%s | blocks=%s",
        settings.mode,
        settings.split,
        settings.targets,
        settings.zones,
        len(blocks),
    )

    selected = read_selected_strategy_or_fallback() if settings.mode == "final_test" else None
    param_grid = resolve_param_grid_for_mode(settings, selected)

    all_predictions: List[pd.DataFrame] = []

    for target in settings.targets:
        if settings.mode == "final_test":
            window_strategy, param_config = get_final_strategy_for_target(target, selected)
            target_window_strategies = [window_strategy]
            target_param_configs = [param_config]
        else:
            target_window_strategies = settings.window_strategies
            target_param_configs = list(param_grid.keys())

        for zone in settings.zones:
            logging.info("Target-zone: %s-%s", target, zone)

            # Explicit check before entering the heavy loop.
            cross_features = get_cross_zone_features_for_zone(panel_features, zone)
            check_no_own_zone_cross_features(cross_features, zone)

            for window_strategy in target_window_strategies:
                for param_config in target_param_configs:
                    params = param_grid[param_config]

                    for block in blocks:
                        # Skip blocks with no rows for this target-zone.
                        eval_rows = get_prediction_rows_for_block(
                            eval_subset,
                            target,
                            zone,
                            block,
                        )
                        if eval_rows.empty:
                            continue

                        logging.info(
                            "Fitting/predicting %s | %s | %s | %s | %s",
                            target,
                            zone,
                            window_strategy,
                            param_config,
                            block.block_id,
                        )

                        preds = predict_block(
                            panel_features=panel_features,
                            eval_subset=eval_subset,
                            settings=settings,
                            target=target,
                            zone=zone,
                            window_strategy=window_strategy,
                            param_config=param_config,
                            params=params,
                            block=block,
                        )
                        all_predictions.append(preds)

    if not all_predictions:
        raise ValueError("No predictions were produced.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(
        ["model", "target", "zone", "delivery_datetime_model"]
    ).reset_index(drop=True)

    quality_check_predictions(predictions, settings.mode)
    save_validation_outputs(predictions, naive_week, settings.mode)
    save_final_predictions(predictions)

    return predictions


def main() -> None:
    mode = check_execution_mode()
    setup_logging()
    ensure_directories()

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Execution mode: %s", mode)

    panel, eval_index, naive_week = read_input_data()
    panel, eval_index, naive_week = standardize_dates(panel, eval_index, naive_week)

    logging.info("Panel rows: %s | Eval rows: %s", len(panel), len(eval_index))

    missing_targets = [target for target in TARGETS if target not in panel.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in panel: {missing_targets}")

    panel_features = prepare_panel_features(panel)
    settings = get_mode_settings(mode)

    predictions = run_experiment(
        panel_features=panel_features,
        eval_index=eval_index,
        naive_week=naive_week,
        settings=settings,
    )

    logging.info("Finished. Prediction rows written: %s", len(predictions))
    logging.info("Main output: %s", p(OUTPUT_PRED_PARQUET))


if __name__ == "__main__":
    main()
