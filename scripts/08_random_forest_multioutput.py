#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_random_forest_multioutput.py

"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Define project root once at module level to ensure all paths are absolute
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT_DIR = _SCRIPT_DIR.parent


@dataclass
class Config:
    """Central configuration for the Random Forest multi-output experiment."""

    # -------------------------------------------------------------------------
    # Project paths
    # -------------------------------------------------------------------------
    PROJECT_ROOT: Path = _PROJECT_ROOT_DIR

    PANEL_RDS_PATH: Path = _PROJECT_ROOT_DIR / "data/processed/gme_model_panel_weather_hourly.rds"
    PANEL_PARQUET_PATH: Path = _PROJECT_ROOT_DIR / "data/processed/gme_model_panel_weather_hourly.parquet"

    EVAL_INDEX_RDS_PATH: Path = _PROJECT_ROOT_DIR / "data/evaluation/eval_index_hourly.rds"
    EVAL_INDEX_PARQUET_PATH: Path = _PROJECT_ROOT_DIR / "data/evaluation/eval_index_hourly.parquet"

    PREDICTIONS_DIR: Path = _PROJECT_ROOT_DIR / "data/predictions"
    FINAL_PREDICTION_PARQUET: Path = _PROJECT_ROOT_DIR / "data/predictions/pred_random_forest_multioutput.parquet"
    FINAL_PREDICTION_RDS: Path = _PROJECT_ROOT_DIR / "data/predictions/pred_random_forest_multioutput.rds"

    WEEKLY_NAIVE_PARQUET_PATH: Path = _PROJECT_ROOT_DIR / "data/predictions/pred_naive_week_before.parquet"
    WEEKLY_NAIVE_RDS_PATH: Path = _PROJECT_ROOT_DIR / "data/predictions/pred_naive_week_before.rds"

    EXPERIMENT_DIR: Path = _PROJECT_ROOT_DIR / "experiments/random_forest_multioutput" 

    # -------------------------------------------------------------------------
    # Execution modes
    # -------------------------------------------------------------------------
    # For the first real check, keep RUN_FAST_VALIDATION=True and the rest False.
    RUN_SMOKE_TEST: bool = False
    RUN_FAST_VALIDATION: bool = False
    RUN_FULL_VALIDATION: bool = False
    RUN_FINAL_TEST: bool = True

    # If several modes are True, RUN_SMOKE_TEST has priority and runs alone.
    # RUN_FINAL_TEST should only be enabled after choosing the structure/window
    # using validation results.

    # -------------------------------------------------------------------------
    # Checkpointing and resume
    # -------------------------------------------------------------------------
    SAVE_CHECKPOINTS: bool = True
    RESUME_FROM_CHECKPOINTS: bool = True
    OVERWRITE_EXISTING_CHECKPOINTS: bool = False
    SAVE_FITTED_MODELS: bool = False

    # -------------------------------------------------------------------------
    # Main modelling choices
    # -------------------------------------------------------------------------
    TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    ZONES: Tuple[str, ...] = ("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")

    # Candidate output structures.
    VARIANTS_FAST_VALIDATION: Tuple[str, ...] = (
        "rf_H_24h",
        "rf_Z_7zones",
        "rf_HZ_24h_7zones",
        "rf_T_3targets",
    )
    VARIANTS_FULL_VALIDATION: Tuple[str, ...] = (
        "rf_H_24h",
        "rf_Z_7zones",
    )
    # Final test should usually contain only the selected best variant(s).
    # If USE_TARGET_SPECIFIC_FINAL_SELECTION=True, this tuple is only used for
    # validation/logging; tasks are generated from SELECTED_VARIANT_BY_TARGET.
    VARIANTS_FINAL_TEST: Tuple[str, ...] = ("rf_H_24h", "rf_Z_7zones")

    # Optional target-specific final model. Use only after validation has selected
    # the structure and window on 2024. The test predictions are then written with
    # one common model name so they can be compared against LEAR/SARIMAX/benchmarks
    # as a single Random Forest selected model.
    USE_TARGET_SPECIFIC_FINAL_SELECTION: bool = True
    FINAL_SELECTED_MODEL_NAME: str = "random_forest_selected"
    SELECTED_VARIANT_BY_TARGET: Dict[str, str] = field(
        default_factory=lambda: {
            "price": "rf_Z_7zones",
            "purchases": "rf_H_24h",
            "sales": "rf_H_24h",
        }
    )
    SELECTED_WINDOW_STRATEGY_BY_TARGET: Dict[str, str] = field(
        default_factory=lambda: {
            "price": "rolling_24m",
            "purchases": "rolling_24m",
            "sales": "rolling_24m",
        }
    )

    RUN_OPTIONAL_RF_HT: bool = False
    OPTIONAL_VARIANT_NAME: str = "rf_HT_24h_3targets"

    # Window strategies implemented. Use one by default to avoid duplicate model
    # names and excessive computation. Add "expanding" to compare windows.
    WINDOW_STRATEGIES_FAST_VALIDATION: Tuple[str, ...] = ("rolling_24m",)
    WINDOW_STRATEGIES_FULL_VALIDATION: Tuple[str, ...] = ("rolling_12m", "rolling_24m")
    WINDOW_STRATEGIES_FINAL_TEST: Tuple[str, ...] = ("rolling_24m",)

    # If more than one window strategy is active, the model name automatically
    # includes the window strategy to avoid duplicates.
    ALWAYS_INCLUDE_WINDOW_STRATEGY_IN_MODEL_NAME: bool = True

    # -------------------------------------------------------------------------
    # Validation/test periods
    # -------------------------------------------------------------------------
    VALIDATION_YEAR: int = 2024
    TEST_YEAR: int = 2025

    # Monthly recalibration by default. The script groups forecast dates by month.
    RECALIBRATION_FREQUENCY: str = "monthly"

    # Training windows.
    INITIAL_TRAIN_START_DATE: str = "2021-01-01"
    ROLLING_WINDOW_MONTHS: int = 24

    # -------------------------------------------------------------------------
    # Fast/smoke limits
    # -------------------------------------------------------------------------
    # Smoke test: very small end-to-end run. It still writes official predictions.
    SMOKE_TARGETS: Tuple[str, ...] = ("price",)
    SMOKE_ZONES: Tuple[str, ...] = ("NORD", "CSUD")
    SMOKE_VARIANTS: Tuple[str, ...] = ("rf_H_24h",)
    SMOKE_MAX_FORECAST_DATES: int = 14
    SMOKE_MAX_VALIDATION_MONTHS: Optional[int] = 1
    SMOKE_N_ESTIMATORS: int = 20

    # Fast validation: reduced experiment for comparing output structures.
    # You can reduce these further if your laptop struggles.
    FAST_TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    FAST_ZONES: Tuple[str, ...] = ("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")
    FAST_MAX_VALIDATION_MONTHS: Optional[int] = 3
    FAST_MAX_FORECAST_DATES: Optional[int] = None
    FAST_N_ESTIMATORS: int = 80

    # Full validation and final test defaults.
    FULL_TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    FULL_ZONES: Tuple[str, ...] = ("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")
    FINAL_TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    FINAL_ZONES: Tuple[str, ...] = ("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")

    # -------------------------------------------------------------------------
    # Random Forest hyperparameters
    # -------------------------------------------------------------------------
    RF_N_ESTIMATORS: int = 300
    RF_MAX_FEATURES: Any = 0.5  # alternatives: "sqrt", None
    RF_MIN_SAMPLES_LEAF: int = 5
    RF_MAX_DEPTH: Optional[int] = None
    RF_N_JOBS: int = -1
    RF_RANDOM_STATE: int = 123
    RF_BOOTSTRAP: bool = True
    RF_CRITERION: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = "squared_error"

    # Optional small tuning section, disabled by default. This script only exposes
    # the hook; it does not run a costly hyperparameter optimization by default.
    RUN_SMALL_TUNING_FOR_SELECTED_STRUCTURE: bool = False

    # -------------------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------------------
    USE_WEATHER_FOR_DELIVERY_DAY: bool = True
    USE_CROSS_ZONE_LAGS: bool = True
    REDUCED_CROSS_ZONE_LAGS: bool = False

    # If USE_WEATHER_FOR_DELIVERY_DAY=False, weather_lag24/weather_lag168 are used.
    WEATHER_LAGS: Tuple[int, ...] = (24, 168)

    TARGET_LAGS: Tuple[int, ...] = (24, 48, 168)
    MARKET_LAGS: Tuple[int, ...] = (24, 168)
    NATIONAL_LAGS: Tuple[int, ...] = (24, 168)
    EXTERNAL_LAGS: Tuple[int, ...] = (24, 168)
    MTI_LAGS: Tuple[int, ...] = (24, 168)

    INCLUDE_CATEGORICAL_CALENDAR: bool = False
    INCLUDE_CYCLICAL_CALENDAR: bool = True
    INCLUDE_ZONE_AS_FEATURE_FOR_T: bool = True

    # -------------------------------------------------------------------------
    # Feature importance and model analysis
    # -------------------------------------------------------------------------
    RUN_PERMUTATION_IMPORTANCE: bool = False
    PERMUTATION_N_REPEATS: int = 5
    PERMUTATION_MAX_ROWS: int = 500
    PERMUTATION_RANDOM_STATE: int = 123

    # Impurity importance is cheap and enabled by default.
    SAVE_IMPURITY_IMPORTANCE: bool = True

    # -------------------------------------------------------------------------
    # IO, diagnostics and robustness
    # -------------------------------------------------------------------------
    TRY_WRITE_RDS_OUTPUT: bool = True
    COMPUTE_QUICK_METRICS: bool = True
    LOG_LEVEL: str = "INFO"

    # Parquet engine. pyarrow is recommended for compatibility with R/arrow.
    PARQUET_ENGINE: Literal["auto", "pyarrow", "fastparquet"] = "pyarrow"

    # One-hot encoding can be sparse to reduce memory. RandomForestRegressor can
    # handle sparse inputs in recent scikit-learn versions.
    OHE_SPARSE_OUTPUT: bool = True

    # Minimum data checks.
    MIN_TRAIN_ROWS: int = 30
    MIN_PREDICTION_ROWS: int = 1

    # Output order for rf_HZ_24h_7zones. The default is deterministic and documented:
    # output index = zone index first, then hour 1..24.
    HZ_OUTPUT_ORDER: str = "zone_then_hour"  # do not change unless you also update mapping logic

    # Safety/debugging.
    FAIL_ON_EMPTY_FINAL_OUTPUT: bool = True
    GARBAGE_COLLECT_AFTER_TASK: bool = True


CFG = Config()


# =============================================================================
# 2. CONSTANTS
# =============================================================================


OFFICIAL_PREDICTION_COLUMNS = [
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

ESSENTIAL_PANEL_COLUMNS = ["datetime_model", "date", "hour", "zone", "price", "purchases", "sales"]

REGIONAL_TARGET_VARIABLES = ["price", "purchases", "sales"]
REGIONAL_MARKET_VARIABLES = ["hhi", "rsi", "mti"]
NATIONAL_VARIABLES = ["pun", "purchases_italy", "sales_italy", "unsold_italy"]
EXTERNAL_VARIABLES = [
    "purchases_external_total",
    "purchases_external_n_active_areas",
    "sales_external_total",
    "sales_external_n_active_areas",
]
WEATHER_VARIABLES = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]

UNSAFE_SAME_DAY_MARKET_COLUMNS = set(
    REGIONAL_TARGET_VARIABLES
    + REGIONAL_MARKET_VARIABLES
    + NATIONAL_VARIABLES
    + EXTERNAL_VARIABLES
)

VALID_VARIANTS = {
    "rf_H_24h",
    "rf_Z_7zones",
    "rf_HZ_24h_7zones",
    "rf_T_3targets",
    "rf_HT_24h_3targets",
}


# =============================================================================
# 3. DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class RunSpec:
    mode_name: str
    split: str
    variants: Tuple[str, ...]
    window_strategies: Tuple[str, ...]
    targets: Tuple[str, ...]
    zones: Tuple[str, ...]
    n_estimators: int
    max_months: Optional[int] = None
    max_forecast_dates: Optional[int] = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    mode_name: str
    split: str
    variant: str
    window_strategy: str
    target: Optional[str]
    zone: Optional[str]
    zones: Tuple[str, ...]
    targets: Tuple[str, ...]
    recalibration_date: pd.Timestamp
    forecast_dates: Tuple[pd.Timestamp, ...]
    n_estimators: int
    model_name: str
    config_signature: str


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    Y: np.ndarray
    meta: pd.DataFrame
    numeric_features: List[str]
    categorical_features: List[str]
    output_mapping: pd.DataFrame
    target_scaling_required: bool = False


# =============================================================================
# 4. LOGGING AND BASIC IO HELPERS
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config) -> logging.Logger:
    ensure_dir(cfg.EXPERIMENT_DIR)
    log_path = cfg.EXPERIMENT_DIR / "rf_script.log"

    logger = logging.getLogger("random_forest_multioutput")
    logger.setLevel(getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def atomic_write_parquet(df: pd.DataFrame, path: Path, engine: Literal["auto", "pyarrow", "fastparquet"] = "pyarrow") -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False, engine=engine)
    os.replace(tmp_path, path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp_path, path)


def read_rds(path: Path) -> pd.DataFrame:
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyreadr is not installed. Install it with `pip install pyreadr`, "
            "or provide a parquet version of the input file."
        ) from exc

    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"No object found inside RDS file: {path}")
    first_key = next(iter(result.keys()))
    return result[first_key]


def read_dataframe_with_fallback(
    rds_path: Path,
    parquet_path: Path,
    description: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if parquet_path.exists():
        logger.info("Reading %s from parquet: %s", description, parquet_path)
        return pd.read_parquet(parquet_path)

    if rds_path.exists():
        logger.info("Reading %s from RDS: %s", description, rds_path)
        return read_rds(rds_path)

    raise FileNotFoundError(
        f"Could not find {description}. Expected either:\n"
        f"  - {parquet_path}\n"
        f"  - {rds_path}"
    )


def maybe_write_rds(df: pd.DataFrame, path: Path, logger: logging.Logger) -> bool:
    try:
        import pyreadr  # type: ignore
    except ImportError:
        logger.warning("pyreadr is not installed; skipping RDS output: %s", path)
        return False

    try:
        ensure_dir(path.parent)
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        if tmp_path.exists():
            tmp_path.unlink()
        pyreadr.write_rds(str(tmp_path), df)
        os.replace(tmp_path, path)
        logger.info("Saved RDS output: %s", path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write RDS output %s. Error: %s", path, exc)
        return False

def normalize_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value).normalize()

# =============================================================================
# 5. GENERAL UTILITIES
# =============================================================================


def to_timestamp_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.normalize()


def normalize_zone_names(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def sanitize_for_filename(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def config_signature(cfg: Config, n_estimators: int) -> str:
    """Short signature to avoid checkpoint collisions when key settings change."""
    relevant = {
        "n_estimators": n_estimators,
        "max_features": cfg.RF_MAX_FEATURES,
        "min_samples_leaf": cfg.RF_MIN_SAMPLES_LEAF,
        "max_depth": cfg.RF_MAX_DEPTH,
        "bootstrap": cfg.RF_BOOTSTRAP,
        "criterion": cfg.RF_CRITERION,
        "use_weather_delivery_day": cfg.USE_WEATHER_FOR_DELIVERY_DAY,
        "use_cross_zone_lags": cfg.USE_CROSS_ZONE_LAGS,
        "reduced_cross_zone_lags": cfg.REDUCED_CROSS_ZONE_LAGS,
        "target_lags": cfg.TARGET_LAGS,
        "market_lags": cfg.MARKET_LAGS,
        "national_lags": cfg.NATIONAL_LAGS,
        "external_lags": cfg.EXTERNAL_LAGS,
        "mti_lags": cfg.MTI_LAGS,
        "rolling_window_months": cfg.ROLLING_WINDOW_MONTHS,
        "always_include_window_strategy": cfg.ALWAYS_INCLUDE_WINDOW_STRATEGY_IN_MODEL_NAME,
        "script_feature_version": "rf_window_and_target_selection_v2",
    }
    payload = json.dumps(relevant, sort_keys=True, default=str)
    # stable small hash without importing hashlib? Use hashlib for determinism.
    import hashlib

    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def make_task_id(
    mode_name: str,
    split: str,
    variant: str,
    window_strategy: str,
    target: Optional[str],
    zone: Optional[str],
    zones: Sequence[str],
    recalibration_date: pd.Timestamp,
    signature: str,
) -> str:
    target_key = target if target is not None else "alltargets"
    if zone is not None:
        zone_key = zone
    else:
        zone_key = "zones_" + "-".join(zones)
    parts = [
        mode_name,
        split,
        variant,
        window_strategy,
        target_key,
        zone_key,
        recalibration_date.strftime("%Y-%m-%d"),
        f"sig{signature}",
    ]
    return sanitize_for_filename("__".join(parts))


def get_checkpoint_paths(cfg: Config, task_id: str) -> Dict[str, Path]:
    import hashlib

    base = cfg.EXPERIMENT_DIR / "checkpoints"
    file_id = hashlib.md5(task_id.encode("utf-8")).hexdigest()[:16]

    return {
        "prediction": base / "predictions" / f"{file_id}.parquet",
        "log": base / "logs" / f"{file_id}_log.json",
        "model": base / "models" / f"{file_id}.joblib",
        "importance_detailed": base / "feature_importance_detailed" / f"{file_id}.parquet",
        "importance_aggregated": base / "feature_importance_aggregated" / f"{file_id}.parquet",
        "permutation": base / "permutation_importance" / f"{file_id}.parquet",
        "feature_metadata": base / "feature_metadata" / f"{file_id}.json",
        "output_mapping": base / "output_mapping" / f"{file_id}.parquet",
    }


def get_model_name(cfg: Config, variant: str, window_strategy: str, active_window_count: int) -> str:
    base = f"random_forest_{variant}"
    if cfg.ALWAYS_INCLUDE_WINDOW_STRATEGY_IN_MODEL_NAME or active_window_count > 1:
        base = f"{base}_{window_strategy}"
    return base


def is_valid_window_strategy(window_strategy: str) -> bool:
    """Return True for expanding or rolling windows such as rolling_12m."""
    return window_strategy == "expanding" or re.fullmatch(r"rolling_\d+m", window_strategy) is not None


def rolling_months_from_strategy(window_strategy: str) -> int:
    """Extract the number of months from strings such as rolling_12m."""
    match = re.fullmatch(r"rolling_(\d+)m", window_strategy)
    if match is None:
        raise ValueError(f"Invalid rolling window strategy: {window_strategy}")
    return int(match.group(1))


def finite_or_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


# =============================================================================
# 6. INPUT LOADING AND PANEL PREPARATION
# =============================================================================


def load_inputs(cfg: Config, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel = read_dataframe_with_fallback(
        cfg.PANEL_RDS_PATH,
        cfg.PANEL_PARQUET_PATH,
        description="hourly model panel",
        logger=logger,
    )
    eval_index = read_dataframe_with_fallback(
        cfg.EVAL_INDEX_RDS_PATH,
        cfg.EVAL_INDEX_PARQUET_PATH,
        description="common evaluation index",
        logger=logger,
    )
    return panel, eval_index


def check_essential_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing essential columns in {label}: {missing}")


def prepare_panel(panel: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    """Normalize the panel and create calendar variables.

    This function does not create supervised samples yet. It only ensures that
    the core panel has consistent types and calendar information.
    """
    panel = panel.copy()
    check_essential_columns(panel, ESSENTIAL_PANEL_COLUMNS, "panel")

    panel["datetime_model"] = pd.to_datetime(panel["datetime_model"])
    panel["delivery_datetime_model"] = panel["datetime_model"]
    panel["date"] = to_timestamp_date(panel["date"])
    panel["delivery_date"] = panel["date"]
    panel["hour"] = panel["hour"].astype(int)
    panel["horizon"] = panel["hour"].astype(int)
    panel["zone"] = normalize_zone_names(panel["zone"])

    # Keep only configured zones if other zones/areas are present in the raw data.
    unknown_zones = sorted(set(panel["zone"].dropna()) - set(cfg.ZONES))
    if unknown_zones:
        logger.info("Panel contains non-physical or extra zones/areas not used as targets: %s", unknown_zones)

    # Calendar variables are deterministic and safe to recompute.
    # This avoids problems when R/parquet imports them as categorical variables.
    panel["hour"] = pd.to_numeric(panel["hour"], errors="coerce").astype(int)

    panel["weekday"] = panel["delivery_date"].dt.weekday.astype(int) + 1  # Monday=1, Sunday=7
    panel["month"] = panel["delivery_date"].dt.month.astype(int)
    panel["is_weekend"] = panel["weekday"].isin([6, 7]).astype(int)

    # Cyclical encodings.
    if cfg.INCLUDE_CYCLICAL_CALENDAR:
        panel["sin_hour"] = np.sin(2.0 * np.pi * panel["hour"].astype(float) / 24.0)
        panel["cos_hour"] = np.cos(2.0 * np.pi * panel["hour"].astype(float) / 24.0)
        panel["sin_weekday"] = np.sin(2.0 * np.pi * panel["weekday"].astype(float) / 7.0)
        panel["cos_weekday"] = np.cos(2.0 * np.pi * panel["weekday"].astype(float) / 7.0)
        panel["sin_month"] = np.sin(2.0 * np.pi * panel["month"].astype(float) / 12.0)
        panel["cos_month"] = np.cos(2.0 * np.pi * panel["month"].astype(float) / 12.0)

    # Categorical encodings. Current mti_f is deliberately created but not used
    # as a feature by default because same-day mti is not allowed. Lagged mti_f
    # variables are created later and can be used safely.
    if cfg.INCLUDE_CATEGORICAL_CALENDAR:
        panel["hour_f"] = panel["hour"].astype("Int64").astype(str)
        panel["weekday_f"] = panel["weekday"].astype("Int64").astype(str)
        panel["month_f"] = panel["month"].astype("Int64").astype(str)
    panel["zone_f"] = panel["zone"].astype(str)
    if "mti" in panel.columns:
        panel["mti_f"] = panel["mti"].astype("string").fillna("MISSING").astype(str)

    # Robust sorting is essential before lag creation.
    panel = panel.sort_values(["zone", "datetime_model"]).reset_index(drop=True)

    duplicates = panel.duplicated(["zone", "datetime_model"]).sum()
    if duplicates > 0:
        logger.warning("Panel contains %s duplicate zone-datetime rows. Keeping all rows; check upstream data.", duplicates)

    # Warn, do not crash, for optional missing variables.
    optional_vars = REGIONAL_MARKET_VARIABLES + NATIONAL_VARIABLES + EXTERNAL_VARIABLES + WEATHER_VARIABLES
    missing_optional = [col for col in optional_vars if col not in panel.columns]
    if missing_optional:
        logger.warning("Optional variables missing and will be skipped: %s", missing_optional)

    return panel


def create_lag_features(panel: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    """Create lagged variables before splitting.

    Leakage note
    ------------
    Lag features are computed on the full chronological panel but only refer to
    past timestamps through group-wise shift operations. The later rolling
    evaluation still enforces that model fitting uses only training dates before
    each recalibration date.
    """
    df = panel.copy()
    df = df.sort_values(["zone", "datetime_model"]).reset_index(drop=True)

    regional_lag_plan: Dict[str, Tuple[int, ...]] = {}
    for var in REGIONAL_TARGET_VARIABLES:
        if var in df.columns:
            regional_lag_plan[var] = cfg.TARGET_LAGS
    for var in ["hhi", "rsi"]:
        if var in df.columns:
            regional_lag_plan[var] = cfg.MARKET_LAGS
    if "mti" in df.columns:
        regional_lag_plan["mti"] = cfg.MTI_LAGS

    if not cfg.USE_WEATHER_FOR_DELIVERY_DAY:
        for var in WEATHER_VARIABLES:
            if var in df.columns:
                regional_lag_plan[var] = cfg.WEATHER_LAGS

    # Own-zone lags.
    for var, lags in regional_lag_plan.items():
        for lag in lags:
            col = f"{var}_lag{lag}"
            df[col] = df.groupby("zone", sort=False)[var].shift(lag)
            if var == "mti":
                df[f"{col}_f"] = df[col].astype("string").fillna("MISSING").astype(str)

    # National/common variable lags: one value per timestamp, merged back to all zones.
    common_lag_plan: Dict[str, Tuple[int, ...]] = {}
    for var in NATIONAL_VARIABLES:
        if var in df.columns:
            common_lag_plan[var] = cfg.NATIONAL_LAGS
    for var in EXTERNAL_VARIABLES:
        if var in df.columns:
            common_lag_plan[var] = cfg.EXTERNAL_LAGS

    if common_lag_plan:
        common = df.sort_values("datetime_model").drop_duplicates("datetime_model")[["datetime_model"] + list(common_lag_plan.keys())]
        common = common.sort_values("datetime_model").reset_index(drop=True)
        lag_cols = []
        for var, lags in common_lag_plan.items():
            for lag in lags:
                col = f"{var}_lag{lag}"
                common[col] = common[var].shift(lag)
                lag_cols.append(col)
        df = df.merge(common[["datetime_model"] + lag_cols], on="datetime_model", how="left")

    # Cross-zone lag features. These are especially important for multiregional RF.
    if cfg.USE_CROSS_ZONE_LAGS:
        cross_vars = [v for v in regional_lag_plan.keys() if v != "mti"]
        if cfg.REDUCED_CROSS_ZONE_LAGS:
            # Reduced version: keep cross-zone lags only for the three targets and
            # market concentration indicators. National/external variables are
            # already common lags.
            cross_vars = [v for v in cross_vars if v in REGIONAL_TARGET_VARIABLES + ["hhi", "rsi"]]

        for var in cross_vars:
            if var not in df.columns:
                continue
            lags = regional_lag_plan[var]
            wide = df.pivot_table(index="datetime_model", columns="zone", values=var, aggfunc="first").sort_index()
            for lag in lags:
                shifted = wide.shift(lag)
                shifted.columns = [f"{var}_{zone}_lag{lag}" for zone in shifted.columns]
                shifted = shifted.reset_index()
                df = df.merge(shifted, on="datetime_model", how="left")

    # Delivery-day weather as forecast proxy: create wide current weather variables
    # so spatial models can use weather information for all zones at each hour.
    if cfg.USE_WEATHER_FOR_DELIVERY_DAY:
        for var in WEATHER_VARIABLES:
            if var not in df.columns:
                continue
            wide_weather = df.pivot_table(index="datetime_model", columns="zone", values=var, aggfunc="first").sort_index()
            wide_weather.columns = [f"{var}_{zone}" for zone in wide_weather.columns]
            wide_weather = wide_weather.reset_index()
            df = df.merge(wide_weather, on="datetime_model", how="left")

    logger.info("Created lag/cross-zone features. Panel now has %s columns.", df.shape[1])
    return df


def prepare_eval_index(eval_index: pd.DataFrame, panel: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    """Convert the common evaluation index to official long format.

    The existing R evaluation script expects predictions in long format. This
    function guarantees that the evaluation index used by Python has the same
    target-zone-hour structure and includes y_true from the panel when needed.
    """
    idx = eval_index.copy()

    rename_map = {}
    if "datetime_model" in idx.columns and "delivery_datetime_model" not in idx.columns:
        rename_map["datetime_model"] = "delivery_datetime_model"
    if "date" in idx.columns and "delivery_date" not in idx.columns:
        rename_map["date"] = "delivery_date"
    idx = idx.rename(columns=rename_map)

    if "delivery_datetime_model" in idx.columns:
        idx["delivery_datetime_model"] = pd.to_datetime(idx["delivery_datetime_model"])
    else:
        raise ValueError(
            "Evaluation index must contain either `delivery_datetime_model` or `datetime_model`. "
            "The Random Forest predictions must be aligned with the common evaluation index."
        )

    if "delivery_date" not in idx.columns:
        idx["delivery_date"] = idx["delivery_datetime_model"].dt.normalize()
    else:
        idx["delivery_date"] = to_timestamp_date(idx["delivery_date"])

    if "forecast_date" not in idx.columns:
        # In this thesis pipeline, if forecast_date is absent, we preserve the
        # evaluation grid by using delivery_date as forecast_date. If your index
        # explicitly has a previous-day forecast origin, this branch will not run.
        idx["forecast_date"] = idx["delivery_date"]
    else:
        idx["forecast_date"] = to_timestamp_date(idx["forecast_date"])

    if "hour" not in idx.columns:
        # Recover hour from the panel if possible; otherwise use clock hour + 1.
        idx["hour"] = idx["delivery_datetime_model"].dt.hour + 1
    idx["hour"] = idx["hour"].astype(int)

    if "horizon" not in idx.columns:
        idx["horizon"] = idx["hour"].astype(int)
    idx["horizon"] = idx["horizon"].astype(int)

    if "split" not in idx.columns:
        year = idx["delivery_date"].dt.year
        idx["split"] = np.where(
            year == cfg.VALIDATION_YEAR,
            "validation",
            np.where(year == cfg.TEST_YEAR, "test", "other"),
        )

    # If target and zone are absent, expand the time index to all thesis targets/zones.
    time_cols = ["split", "forecast_date", "delivery_datetime_model", "delivery_date", "hour", "horizon"]
    has_target = "target" in idx.columns
    has_zone = "zone" in idx.columns

    if has_zone:
        idx["zone"] = normalize_zone_names(idx["zone"])

    if not (has_target and has_zone):
        base_time = idx[time_cols].drop_duplicates().reset_index(drop=True)
        target_zone = pd.MultiIndex.from_product(
            [cfg.TARGETS, cfg.ZONES], names=["target", "zone"]
        ).to_frame(index=False)
        base_time["_tmp_key"] = 1
        target_zone["_tmp_key"] = 1
        idx_long = base_time.merge(target_zone, on="_tmp_key", how="inner").drop(columns="_tmp_key")
        logger.info(
            "Evaluation index did not contain both target and zone. Expanded to long target-zone format: %s rows.",
            len(idx_long),
        )
    else:
        keep_cols = time_cols + ["target", "zone"]
        if "y_true" in idx.columns:
            keep_cols.append("y_true")
        idx_long = idx[keep_cols].drop_duplicates().reset_index(drop=True)

    idx_long["target"] = idx_long["target"].astype(str)
    idx_long["zone"] = normalize_zone_names(idx_long["zone"])

    # Attach y_true from panel if missing or partially missing.
    needs_y_true = "y_true" not in idx_long.columns or idx_long["y_true"].isna().any()
    if needs_y_true:
        panel_truth = panel[
            ["delivery_datetime_model", "zone"] + [t for t in cfg.TARGETS if t in panel.columns]
        ].copy()
        panel_truth = panel_truth[panel_truth["zone"].isin(cfg.ZONES)]
        panel_truth_long = panel_truth.melt(
            id_vars=["delivery_datetime_model", "zone"],
            value_vars=[t for t in cfg.TARGETS if t in panel_truth.columns],
            var_name="target",
            value_name="y_true_panel",
        )
        idx_long = idx_long.merge(
            panel_truth_long,
            on=["delivery_datetime_model", "zone", "target"],
            how="left",
        )
        if "y_true" in idx_long.columns:
            idx_long["y_true"] = idx_long["y_true"].where(idx_long["y_true"].notna(), idx_long["y_true_panel"])
            idx_long = idx_long.drop(columns=["y_true_panel"])
        else:
            idx_long = idx_long.rename(columns={"y_true_panel": "y_true"})

    idx_long = idx_long[OFFICIAL_PREDICTION_COLUMNS[1:-1]].copy()  # all official except model and y_pred
    idx_long = idx_long.sort_values(["split", "target", "zone", "delivery_datetime_model"]).reset_index(drop=True)

    missing_truth = idx_long["y_true"].isna().sum()
    if missing_truth > 0:
        logger.warning("Evaluation index has %s rows with missing y_true after merging panel truth.", missing_truth)

    return idx_long


# =============================================================================
# 7. FEATURE SELECTION
# =============================================================================


def is_lag_feature(col: str) -> bool:
    return bool(re.search(r"_lag\d+(_f)?$", col))


def is_cross_zone_lag(col: str, zones: Sequence[str]) -> bool:
    zone_pattern = "|".join(map(re.escape, zones))
    return bool(re.search(rf"_({zone_pattern})_lag\d+", col))


def is_own_regional_lag(col: str) -> bool:
    regional_prefix = "|".join(map(re.escape, REGIONAL_TARGET_VARIABLES + REGIONAL_MARKET_VARIABLES + WEATHER_VARIABLES))
    return bool(re.match(rf"^({regional_prefix})_lag\d+(_f)?$", col))


def is_current_weather_wide(col: str, zones: Sequence[str]) -> bool:
    for weather_var in WEATHER_VARIABLES:
        if any(col == f"{weather_var}_{zone}" for zone in zones):
            return True
    return False

def is_raw_mti_lag(col: str) -> bool:
    return bool(re.match(r"^mti(_[A-Z]+)?_lag\d+$", col))

def base_calendar_numeric_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "hour",
        "weekday",
        "month",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_month",
        "cos_month",
    ]
    return [col for col in candidates if col in df.columns]


def base_calendar_categorical_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["hour_f", "weekday_f", "month_f"]
    if "holiday" in df.columns:
        candidates.append("holiday")
    if "holiday_f" in df.columns:
        candidates.append("holiday_f")
    return [col for col in candidates if col in df.columns]


def select_row_level_features(
    df: pd.DataFrame,
    cfg: Config,
    variant: str,
    context: str,
    zones: Sequence[str],
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Select safe raw features for row-level observations.

    Parameters
    ----------
    context:
        - "zone_row": observations retain a concrete zone (rf_H, rf_T, rf_HT).
        - "time_row": observations represent a timestamp without a single output
          zone (rf_Z, rf_HZ), so own-zone lags are excluded unless they are wide
          cross-zone features.
    """
    numeric: List[str] = []
    categorical: List[str] = []

    # Calendar features are known for the delivery day.
    numeric.extend(base_calendar_numeric_columns(df))
    if cfg.INCLUDE_CATEGORICAL_CALENDAR:
        categorical.extend(base_calendar_categorical_columns(df))

    # Include zone as a feature only when the sample itself has a zone dimension.
    if context == "zone_row" and variant in {"rf_T_3targets", "rf_HT_24h_3targets"}:
        if cfg.INCLUDE_ZONE_AS_FEATURE_FOR_T and "zone_f" in df.columns:
            categorical.append("zone_f")

    # Lagged market features are safe because they refer to past timestamps.
    lag_cols = [col for col in df.columns if is_lag_feature(col)]

    if context == "time_row":
        # For spatial output structures, a single row should not accidentally use
        # arbitrary own-zone lag columns from whichever zone happens to be first.
        # Use cross-zone lag columns and common/national lags instead.
        lag_cols = [col for col in lag_cols if not is_own_regional_lag(col)]

    # If cross-zone lags are disabled, remove wide cross-zone lag columns.
    if not cfg.USE_CROSS_ZONE_LAGS:
        lag_cols = [col for col in lag_cols if not is_cross_zone_lag(col, zones)]

    # Lagged categorical features, currently mostly mti_lagXX_f. 
    lag_cat = [col for col in lag_cols if col.endswith("_f")]
    lag_num = [
        col for col in lag_cols
        if col not in lag_cat and not is_raw_mti_lag(col)
    ]
    numeric.extend(lag_num)
    categorical.extend(lag_cat)

    # Delivery-day weather as forecast proxy.
    if cfg.USE_WEATHER_FOR_DELIVERY_DAY:
        if context == "zone_row":
            # Own-zone current weather.
            numeric.extend([col for col in WEATHER_VARIABLES if col in df.columns])
            # Optional weather information for all zones at that timestamp.
            if cfg.USE_CROSS_ZONE_LAGS:
                numeric.extend([col for col in df.columns if is_current_weather_wide(col, zones)])
        elif context == "time_row":
            # Spatial structures need weather by zone, not the arbitrary weather
            # value from one zone row.
            numeric.extend([col for col in df.columns if is_current_weather_wide(col, zones)])

    # Remove unsafe raw market variables if they slipped in.
    numeric = [col for col in numeric if col not in UNSAFE_SAME_DAY_MARKET_COLUMNS]
    categorical = [col for col in categorical if col not in UNSAFE_SAME_DAY_MARKET_COLUMNS and col != "mti_f"]

    # Keep existing columns, preserve order, remove duplicates.
    numeric = list(dict.fromkeys([col for col in numeric if col in df.columns]))
    categorical = list(dict.fromkeys([col for col in categorical if col in df.columns]))

    # Avoid classifying the same column twice.
    categorical = [col for col in categorical if col not in numeric]

    meta = {
        "lag_features": [col for col in numeric + categorical if is_lag_feature(col)],
        "cross_zone_features": [col for col in numeric + categorical if is_cross_zone_lag(col, zones) or is_current_weather_wide(col, zones)],
    }
    return numeric, categorical, meta


def flatten_daily_features(
    day_rows: pd.DataFrame,
    numeric_base: Sequence[str],
    categorical_base: Sequence[str],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Flatten 24 hourly rows into one daily row with suffix _h01..._h24."""
    row: Dict[str, Any] = {}
    numeric_flat: List[str] = []
    categorical_flat: List[str] = []

    day_rows = day_rows.sort_values("hour")
    for _, r in day_rows.iterrows():
        h = int(r["hour"])
        suffix = f"_h{h:02d}"
        for col in numeric_base:
            new_col = f"{col}{suffix}"
            row[new_col] = r[col] if col in r.index else np.nan
            numeric_flat.append(new_col)
        for col in categorical_base:
            new_col = f"{col}{suffix}"
            value = r[col] if col in r.index else "MISSING"
            row[new_col] = "MISSING" if pd.isna(value) else str(value)
            categorical_flat.append(new_col)

    return row, numeric_flat, categorical_flat


# =============================================================================
# 8. OUTPUT MAPPING
# =============================================================================


def build_output_mapping(
    variant: str,
    target: Optional[str],
    zone: Optional[str],
    zones: Sequence[str],
    targets: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if variant == "rf_H_24h":
        assert target is not None and zone is not None
        for h in range(1, 25):
            rows.append(
                {
                    "variant": variant,
                    "target": target,
                    "zone": zone,
                    "output_index": h - 1,
                    "output_target": target,
                    "output_zone": zone,
                    "output_hour": h,
                    "mapping_note": "hour profile: output_index = hour - 1",
                }
            )

    elif variant == "rf_Z_7zones":
        assert target is not None
        for i, z in enumerate(zones):
            rows.append(
                {
                    "variant": variant,
                    "target": target,
                    "zone": None,
                    "output_index": i,
                    "output_target": target,
                    "output_zone": z,
                    "output_hour": None,
                    "mapping_note": "spatial vector: output_index = zone index in configured zone order",
                }
            )

    elif variant == "rf_HZ_24h_7zones":
        assert target is not None
        idx = 0
        for z in zones:
            for h in range(1, 25):
                rows.append(
                    {
                        "variant": variant,
                        "target": target,
                        "zone": None,
                        "output_index": idx,
                        "output_target": target,
                        "output_zone": z,
                        "output_hour": h,
                        "mapping_note": "zone_then_hour: zone index first, then hour 1..24",
                    }
                )
                idx += 1

    elif variant == "rf_T_3targets":
        for i, t in enumerate(targets):
            rows.append(
                {
                    "variant": variant,
                    "target": None,
                    "zone": None,
                    "output_index": i,
                    "output_target": t,
                    "output_zone": None,
                    "output_hour": None,
                    "mapping_note": "mixed-target vector: [price, purchases, sales] in configured target order",
                }
            )

    elif variant == "rf_HT_24h_3targets":
        assert zone is not None
        idx = 0
        for t in targets:
            for h in range(1, 25):
                rows.append(
                    {
                        "variant": variant,
                        "target": None,
                        "zone": zone,
                        "output_index": idx,
                        "output_target": t,
                        "output_zone": zone,
                        "output_hour": h,
                        "mapping_note": "target_then_hour: target index first, then hour 1..24",
                    }
                )
                idx += 1
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return pd.DataFrame(rows)


# =============================================================================
# 9. SUPERVISED DATASET BUILDERS
# =============================================================================


def filter_dates(df: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    date_set = set(pd.to_datetime(pd.Series(list(dates))).dt.normalize())
    return df[df["delivery_date"].isin(date_set)].copy()


def valid_complete_day(day_rows: pd.DataFrame, zones: Optional[Sequence[str]] = None) -> bool:
    if day_rows.empty:
        return False
    if zones is None:
        return set(day_rows["hour"].astype(int)) == set(range(1, 25))
    expected = {(z, h) for z in zones for h in range(1, 25)}
    got = set(zip(day_rows["zone"].astype(str), day_rows["hour"].astype(int)))
    return expected.issubset(got)


def build_supervised_dataset_H(
    panel: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    target: str,
    zone: str,
    cfg: Config,
) -> DatasetBundle:
    """rf_H_24h: one model per target-zone; output is 24 hourly values."""
    subset_all = panel[panel["zone"] == zone].copy()
    numeric_base, categorical_base, feature_meta = select_row_level_features(
        subset_all,
        cfg=cfg,
        variant="rf_H_24h",
        context="zone_row",
        zones=cfg.ZONES,
    )

    subset = filter_dates(subset_all, dates)
    rows: List[Dict[str, Any]] = []
    y_rows: List[List[float]] = []
    meta_rows: List[Dict[str, Any]] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for date, day in subset.groupby("delivery_date", sort=True):
        if not valid_complete_day(day):
            continue
        day = day.sort_values("hour")
        y = day[target].to_numpy(dtype=float)
        if len(y) != 24 or np.isnan(y).any():
            continue
        x_row, num_flat, cat_flat = flatten_daily_features(day, numeric_base, categorical_base)
        rows.append(x_row)
        y_rows.append(y.tolist())
        meta_rows.append({"delivery_date": normalize_timestamp(date), "zone": zone, "target": target})
        numeric_features = num_flat
        categorical_features = cat_flat

    X = pd.DataFrame(rows)
    Y = np.asarray(y_rows, dtype=float) if y_rows else np.empty((0, 24), dtype=float)
    meta = pd.DataFrame(meta_rows)
    output_mapping = build_output_mapping("rf_H_24h", target=target, zone=zone, zones=[zone], targets=[target])

    # Store metadata attributes for later logging.
    X.attrs["lag_features"] = [f for f in numeric_features + categorical_features if is_lag_feature(f)]
    X.attrs["cross_zone_features"] = [f for f in numeric_features + categorical_features if is_cross_zone_lag(f, cfg.ZONES)]
    X.attrs["base_lag_features"] = feature_meta["lag_features"]
    X.attrs["base_cross_zone_features"] = feature_meta["cross_zone_features"]

    return DatasetBundle(X, Y, meta, numeric_features, categorical_features, output_mapping, False)


def get_time_level_frame(panel: pd.DataFrame, zones: Sequence[str]) -> pd.DataFrame:
    """One row per timestamp for models whose sample is not zone-specific.

    The first row per timestamp is used only as a container for common calendar,
    common lags, cross-zone lags and wide weather variables. Own-zone lag columns
    are excluded by feature selection for these models.
    """
    df = panel[panel["zone"].isin(zones)].copy()
    time_df = df.sort_values(["datetime_model", "zone"]).drop_duplicates("datetime_model").reset_index(drop=True)
    return time_df


def build_supervised_dataset_Z(
    panel: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    target: str,
    zones: Sequence[str],
    cfg: Config,
) -> DatasetBundle:
    """rf_Z_7zones: one model per target; sample is one date-hour; output is zones."""
    time_df_all = get_time_level_frame(panel, zones)
    numeric_features, categorical_features, _ = select_row_level_features(
        time_df_all,
        cfg=cfg,
        variant="rf_Z_7zones",
        context="time_row",
        zones=zones,
    )
    time_df = filter_dates(time_df_all, dates).sort_values("delivery_datetime_model")

    # Y matrix from panel pivot: one column per zone.
    y_source = filter_dates(panel[panel["zone"].isin(zones)], dates)
    y_wide = y_source.pivot_table(
        index="delivery_datetime_model",
        columns="zone",
        values=target,
        aggfunc="first",
    ).reindex(columns=list(zones))

    meta_cols = ["delivery_datetime_model", "delivery_date", "hour"]
    feature_cols = list(dict.fromkeys(list(numeric_features) + list(categorical_features)))
    input_cols = list(dict.fromkeys(meta_cols + feature_cols))

    merged = time_df[input_cols].merge(
        y_wide.reset_index(),
        on="delivery_datetime_model",
        how="left",
    )
    y_cols = list(zones)
    merged = merged.dropna(subset=y_cols)

    X = merged[feature_cols].copy()
    Y = merged[y_cols].to_numpy(dtype=float)
    meta = merged[meta_cols].copy()
    output_mapping = build_output_mapping("rf_Z_7zones", target=target, zone=None, zones=zones, targets=[target])

    X.attrs["lag_features"] = [f for f in numeric_features + categorical_features if is_lag_feature(f)]
    X.attrs["cross_zone_features"] = [
        f for f in numeric_features + categorical_features if is_cross_zone_lag(f, zones) or is_current_weather_wide(f, zones)
    ]
    return DatasetBundle(X, Y, meta, numeric_features, categorical_features, output_mapping, False)


def build_supervised_dataset_HZ(
    panel: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    target: str,
    zones: Sequence[str],
    cfg: Config,
) -> DatasetBundle:
    """rf_HZ_24h_7zones: one model per target; output is zone x 24-hour profile.

    Deterministic output order: zone first, then hour. For configured zones
    [NORD, CNOR, ...], output columns are:
        NORD h1, ..., NORD h24, CNOR h1, ..., SARD h24.
    """
    if cfg.HZ_OUTPUT_ORDER != "zone_then_hour":
        raise ValueError("Only HZ_OUTPUT_ORDER='zone_then_hour' is currently implemented.")

    time_df_all = get_time_level_frame(panel, zones)
    numeric_base, categorical_base, feature_meta = select_row_level_features(
        time_df_all,
        cfg=cfg,
        variant="rf_HZ_24h_7zones",
        context="time_row",
        zones=zones,
    )
    time_df = filter_dates(time_df_all, dates)

    rows: List[Dict[str, Any]] = []
    y_rows: List[List[float]] = []
    meta_rows: List[Dict[str, Any]] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    y_source = filter_dates(panel[panel["zone"].isin(zones)], dates)

    for date, day_time in time_df.groupby("delivery_date", sort=True):
        if not valid_complete_day(day_time):
            continue
        day_panel = y_source[y_source["delivery_date"] == date]
        if not valid_complete_day(day_panel, zones=zones):
            continue
        day_time = day_time.sort_values("hour")
        x_row, num_flat, cat_flat = flatten_daily_features(day_time, numeric_base, categorical_base)

        y_vec: List[float] = []
        ok = True
        for z in zones:
            z_day = day_panel[day_panel["zone"] == z].sort_values("hour")
            vals = z_day[target].to_numpy(dtype=float)
            if len(vals) != 24 or np.isnan(vals).any():
                ok = False
                break
            y_vec.extend(vals.tolist())
        if not ok:
            continue

        rows.append(x_row)
        y_rows.append(y_vec)
        meta_rows.append({"delivery_date": normalize_timestamp(date), "target": target})
        numeric_features = num_flat
        categorical_features = cat_flat

    X = pd.DataFrame(rows)
    Y = np.asarray(y_rows, dtype=float) if y_rows else np.empty((0, len(zones) * 24), dtype=float)
    meta = pd.DataFrame(meta_rows)
    output_mapping = build_output_mapping("rf_HZ_24h_7zones", target=target, zone=None, zones=zones, targets=[target])

    X.attrs["lag_features"] = [f for f in numeric_features + categorical_features if is_lag_feature(f)]
    X.attrs["cross_zone_features"] = [f for f in numeric_features + categorical_features if is_cross_zone_lag(f, zones)]
    X.attrs["base_lag_features"] = feature_meta["lag_features"]
    X.attrs["base_cross_zone_features"] = feature_meta["cross_zone_features"]

    return DatasetBundle(X, Y, meta, numeric_features, categorical_features, output_mapping, False)


def build_supervised_dataset_T(
    panel: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    targets: Sequence[str],
    zones: Sequence[str],
    cfg: Config,
) -> DatasetBundle:
    """rf_T_3targets: sample is one date-zone-hour; output is [price, purchases, sales].

    Because the outputs have different units and scales, Y is standardized within
    each training window and inverse-transformed after prediction.
    """
    subset_all = panel[panel["zone"].isin(zones)].copy()
    numeric_features, categorical_features, _ = select_row_level_features(
        subset_all,
        cfg=cfg,
        variant="rf_T_3targets",
        context="zone_row",
        zones=zones,
    )
    subset = filter_dates(subset_all, dates).sort_values(["delivery_datetime_model", "zone"])
    needed = list(targets)
    subset = subset.dropna(subset=needed)

    X = subset[numeric_features + categorical_features].copy()
    Y = subset[needed].to_numpy(dtype=float)
    meta = subset[["delivery_datetime_model", "delivery_date", "hour", "zone"]].copy()
    output_mapping = build_output_mapping("rf_T_3targets", target=None, zone=None, zones=zones, targets=targets)

    X.attrs["lag_features"] = [f for f in numeric_features + categorical_features if is_lag_feature(f)]
    X.attrs["cross_zone_features"] = [
        f for f in numeric_features + categorical_features if is_cross_zone_lag(f, zones) or is_current_weather_wide(f, zones)
    ]
    return DatasetBundle(X, Y, meta, numeric_features, categorical_features, output_mapping, True)


def build_supervised_dataset_HT(
    panel: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    zone: str,
    targets: Sequence[str],
    cfg: Config,
) -> DatasetBundle:
    """Optional rf_HT_24h_3targets: one model per zone; output target x 24-hour profile.

    Disabled by default. Included as an extension hook.
    """
    subset_all = panel[panel["zone"] == zone].copy()
    numeric_base, categorical_base, feature_meta = select_row_level_features(
        subset_all,
        cfg=cfg,
        variant="rf_HT_24h_3targets",
        context="zone_row",
        zones=cfg.ZONES,
    )
    subset = filter_dates(subset_all, dates)

    rows: List[Dict[str, Any]] = []
    y_rows: List[List[float]] = []
    meta_rows: List[Dict[str, Any]] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for date, day in subset.groupby("delivery_date", sort=True):
        if not valid_complete_day(day):
            continue
        day = day.sort_values("hour")
        y_vec: List[float] = []
        ok = True
        for t in targets:
            vals = day[t].to_numpy(dtype=float)
            if len(vals) != 24 or np.isnan(vals).any():
                ok = False
                break
            y_vec.extend(vals.tolist())
        if not ok:
            continue
        x_row, num_flat, cat_flat = flatten_daily_features(day, numeric_base, categorical_base)
        rows.append(x_row)
        y_rows.append(y_vec)
        meta_rows.append({"delivery_date": normalize_timestamp(date), "zone": zone})
        numeric_features = num_flat
        categorical_features = cat_flat

    X = pd.DataFrame(rows)
    Y = np.asarray(y_rows, dtype=float) if y_rows else np.empty((0, len(targets) * 24), dtype=float)
    meta = pd.DataFrame(meta_rows)
    output_mapping = build_output_mapping("rf_HT_24h_3targets", target=None, zone=zone, zones=[zone], targets=targets)

    X.attrs["lag_features"] = [f for f in numeric_features + categorical_features if is_lag_feature(f)]
    X.attrs["cross_zone_features"] = [f for f in numeric_features + categorical_features if is_cross_zone_lag(f, cfg.ZONES)]
    X.attrs["base_lag_features"] = feature_meta["lag_features"]
    X.attrs["base_cross_zone_features"] = feature_meta["cross_zone_features"]

    return DatasetBundle(X, Y, meta, numeric_features, categorical_features, output_mapping, True)


def build_dataset_for_task(
    panel: pd.DataFrame,
    task: TaskSpec,
    dates: Sequence[pd.Timestamp],
    cfg: Config,
) -> DatasetBundle:
    if task.variant == "rf_H_24h":
        if task.target is None or task.zone is None:
            raise ValueError("rf_H_24h requires task.target and task.zone")
        return build_supervised_dataset_H(panel, dates, task.target, task.zone, cfg)

    if task.variant == "rf_Z_7zones":
        if task.target is None:
            raise ValueError("rf_Z_7zones requires task.target")
        return build_supervised_dataset_Z(panel, dates, task.target, task.zones, cfg)

    if task.variant == "rf_HZ_24h_7zones":
        if task.target is None:
            raise ValueError("rf_HZ_24h_7zones requires task.target")
        return build_supervised_dataset_HZ(panel, dates, task.target, task.zones, cfg)

    if task.variant == "rf_T_3targets":
        return build_supervised_dataset_T(panel, dates, task.targets, task.zones, cfg)

    if task.variant == "rf_HT_24h_3targets":
        if task.zone is None:
            raise ValueError("rf_HT_24h_3targets requires task.zone")
        return build_supervised_dataset_HT(panel, dates, task.zone, task.targets, cfg)

    raise ValueError(f"Unknown variant: {task.variant}")


# =============================================================================
# 10. TRAINING WINDOW AND TASK GENERATION
# =============================================================================


def get_training_dates(
    panel: pd.DataFrame,
    recalibration_date: pd.Timestamp,
    window_strategy: str,
    cfg: Config,
) -> Tuple[pd.Timestamp, pd.Timestamp, List[pd.Timestamp]]:
    recalibration_date = pd.Timestamp(recalibration_date).normalize()
    train_end = recalibration_date

    if window_strategy == "expanding":
        train_start = pd.Timestamp(cfg.INITIAL_TRAIN_START_DATE).normalize()
    elif re.fullmatch(r"rolling_\d+m", window_strategy):
        n_months = rolling_months_from_strategy(window_strategy)
        train_start = train_end - pd.DateOffset(months=n_months)
        min_start = pd.Timestamp(cfg.INITIAL_TRAIN_START_DATE).normalize()
        train_start = max(train_start, min_start)
    else:
        raise ValueError(f"Unknown window_strategy: {window_strategy}")

    dates = sorted(panel.loc[(panel["delivery_date"] >= train_start) & (panel["delivery_date"] < train_end), "delivery_date"].dropna().unique())
    dates = [pd.Timestamp(d).normalize() for d in dates]
    return train_start, train_end, dates


def filter_eval_dates_for_run(eval_long: pd.DataFrame, run: RunSpec, cfg: Config) -> List[pd.Timestamp]:
    subset = eval_long[eval_long["split"] == run.split].copy()
    if run.split == "validation":
        subset = subset[subset["delivery_date"].dt.year == cfg.VALIDATION_YEAR]
    elif run.split == "test":
        subset = subset[subset["delivery_date"].dt.year == cfg.TEST_YEAR]

    dates = sorted(pd.to_datetime(subset["delivery_date"].dropna().unique()))
    dates = [pd.Timestamp(d).normalize() for d in dates]

    if run.max_months is not None:
        months = sorted(pd.PeriodIndex(dates, freq="M").unique())[: run.max_months]
        dates = [d for d in dates if pd.Period(d, freq="M") in months]

    if run.max_forecast_dates is not None:
        dates = dates[: run.max_forecast_dates]

    return dates


def group_dates_by_month(dates: Sequence[pd.Timestamp]) -> Dict[pd.Period, Tuple[pd.Timestamp, ...]]:
    grouped: Dict[pd.Period, List[pd.Timestamp]] = {}
    for d in sorted(pd.Timestamp(x).normalize() for x in dates):
        grouped.setdefault(pd.Period(d, freq="M"), []).append(d)
    return {month: tuple(vals) for month, vals in grouped.items()}


def build_run_specs(cfg: Config) -> List[RunSpec]:
    if cfg.RUN_SMOKE_TEST:
        return [
            RunSpec(
                mode_name="smoke",
                split="validation",
                variants=cfg.SMOKE_VARIANTS,
                window_strategies=("rolling_24m",),
                targets=cfg.SMOKE_TARGETS,
                zones=cfg.SMOKE_ZONES,
                n_estimators=cfg.SMOKE_N_ESTIMATORS,
                max_months=cfg.SMOKE_MAX_VALIDATION_MONTHS,
                max_forecast_dates=cfg.SMOKE_MAX_FORECAST_DATES,
            )
        ]

    runs: List[RunSpec] = []
    if cfg.RUN_FAST_VALIDATION:
        runs.append(
            RunSpec(
                mode_name="fast_validation",
                split="validation",
                variants=cfg.VARIANTS_FAST_VALIDATION,
                window_strategies=cfg.WINDOW_STRATEGIES_FAST_VALIDATION,
                targets=cfg.FAST_TARGETS,
                zones=cfg.FAST_ZONES,
                n_estimators=cfg.FAST_N_ESTIMATORS,
                max_months=cfg.FAST_MAX_VALIDATION_MONTHS,
                max_forecast_dates=cfg.FAST_MAX_FORECAST_DATES,
            )
        )
    if cfg.RUN_FULL_VALIDATION:
        runs.append(
            RunSpec(
                mode_name="full_validation",
                split="validation",
                variants=cfg.VARIANTS_FULL_VALIDATION,
                window_strategies=cfg.WINDOW_STRATEGIES_FULL_VALIDATION,
                targets=cfg.FULL_TARGETS,
                zones=cfg.FULL_ZONES,
                n_estimators=cfg.RF_N_ESTIMATORS,
                max_months=None,
                max_forecast_dates=None,
            )
        )
    if cfg.RUN_FINAL_TEST:
        runs.append(
            RunSpec(
                mode_name="final_test",
                split="test",
                variants=cfg.VARIANTS_FINAL_TEST,
                window_strategies=cfg.WINDOW_STRATEGIES_FINAL_TEST,
                targets=cfg.FINAL_TARGETS,
                zones=cfg.FINAL_ZONES,
                n_estimators=cfg.RF_N_ESTIMATORS,
                max_months=None,
                max_forecast_dates=None,
            )
        )

    if not runs:
        raise ValueError("No execution mode enabled. Set one of RUN_SMOKE_TEST/RUN_FAST_VALIDATION/RUN_FULL_VALIDATION/RUN_FINAL_TEST to True.")
    return runs


def validate_run_specs(runs: Sequence[RunSpec], cfg: Config) -> None:
    for run in runs:
        unknown_variants = [v for v in run.variants if v not in VALID_VARIANTS]
        if unknown_variants:
            raise ValueError(f"Unknown variants in run {run.mode_name}: {unknown_variants}")
        unknown_targets = [t for t in run.targets if t not in cfg.TARGETS]
        if unknown_targets:
            raise ValueError(f"Unknown targets in run {run.mode_name}: {unknown_targets}")
        unknown_zones = [z for z in run.zones if z not in cfg.ZONES]
        if unknown_zones:
            raise ValueError(f"Unknown zones in run {run.mode_name}: {unknown_zones}")
        unknown_windows = [w for w in run.window_strategies if not is_valid_window_strategy(w)]
        if unknown_windows:
            raise ValueError(
                f"Unknown window strategies in run {run.mode_name}: {unknown_windows}. "
                "Use 'expanding' or strings such as 'rolling_12m', 'rolling_24m'."
            )
        if run.split not in {"validation", "test"}:
            raise ValueError(f"Invalid split: {run.split}")
        if "rf_T_3targets" in run.variants and tuple(run.targets) != tuple(cfg.TARGETS):
            warnings.warn(
                "rf_T_3targets is designed for [price, purchases, sales]. "
                "The current run uses a subset or reordered target list. "
                "For thesis-comparable results, use all three targets in the standard order.",
                RuntimeWarning,
            )

    if cfg.RUN_FINAL_TEST and cfg.USE_TARGET_SPECIFIC_FINAL_SELECTION:
        for target in cfg.FINAL_TARGETS:
            if target not in cfg.SELECTED_VARIANT_BY_TARGET:
                raise ValueError(f"Missing selected final RF variant for target: {target}")
            selected_variant = cfg.SELECTED_VARIANT_BY_TARGET[target]
            if selected_variant not in {"rf_H_24h", "rf_Z_7zones", "rf_HZ_24h_7zones"}:
                raise ValueError(
                    f"Target-specific final selection currently supports rf_H_24h, rf_Z_7zones "
                    f"or rf_HZ_24h_7zones. Got {selected_variant} for {target}."
                )
            selected_window = cfg.SELECTED_WINDOW_STRATEGY_BY_TARGET.get(target, cfg.WINDOW_STRATEGIES_FINAL_TEST[0])
            if not is_valid_window_strategy(selected_window):
                raise ValueError(f"Invalid selected final window for {target}: {selected_window}")


def generate_tasks(eval_long: pd.DataFrame, runs: Sequence[RunSpec], cfg: Config, logger: logging.Logger) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []

    for run in runs:
        dates = filter_eval_dates_for_run(eval_long, run, cfg)
        if not dates:
            logger.warning("No evaluation dates found for run %s (%s).", run.mode_name, run.split)
            continue

        month_groups = group_dates_by_month(dates)
        logger.info(
            "Run %s | split=%s | dates=%s to %s | months=%s | variants=%s | targets=%s | zones=%s | n_estimators=%s",
            run.mode_name,
            run.split,
            min(dates).date(),
            max(dates).date(),
            len(month_groups),
            run.variants,
            run.targets,
            run.zones,
            run.n_estimators,
        )

        sig = config_signature(cfg, n_estimators=run.n_estimators)
        active_window_count = len(run.window_strategies)

        if run.mode_name == "final_test" and cfg.USE_TARGET_SPECIFIC_FINAL_SELECTION:
            for month, forecast_dates in month_groups.items():
                recalibration_date = pd.Timestamp(month.start_time).normalize()
                for target in run.targets:
                    variant = cfg.SELECTED_VARIANT_BY_TARGET[target]
                    window_strategy = cfg.SELECTED_WINDOW_STRATEGY_BY_TARGET.get(
                        target,
                        cfg.WINDOW_STRATEGIES_FINAL_TEST[0],
                    )
                    model_name = cfg.FINAL_SELECTED_MODEL_NAME

                    if variant == "rf_H_24h":
                        for zone in run.zones:
                            task_id = make_task_id(
                                run.mode_name,
                                run.split,
                                variant,
                                window_strategy,
                                target,
                                zone,
                                [zone],
                                recalibration_date,
                                sig,
                            )
                            tasks.append(
                                TaskSpec(
                                    task_id=task_id,
                                    mode_name=run.mode_name,
                                    split=run.split,
                                    variant=variant,
                                    window_strategy=window_strategy,
                                    target=target,
                                    zone=zone,
                                    zones=(zone,),
                                    targets=(target,),
                                    recalibration_date=recalibration_date,
                                    forecast_dates=forecast_dates,
                                    n_estimators=run.n_estimators,
                                    model_name=model_name,
                                    config_signature=sig,
                                )
                            )

                    elif variant in {"rf_Z_7zones", "rf_HZ_24h_7zones"}:
                        task_id = make_task_id(
                            run.mode_name,
                            run.split,
                            variant,
                            window_strategy,
                            target,
                            None,
                            run.zones,
                            recalibration_date,
                            sig,
                        )
                        tasks.append(
                            TaskSpec(
                                task_id=task_id,
                                mode_name=run.mode_name,
                                split=run.split,
                                variant=variant,
                                window_strategy=window_strategy,
                                target=target,
                                zone=None,
                                zones=tuple(run.zones),
                                targets=(target,),
                                recalibration_date=recalibration_date,
                                forecast_dates=forecast_dates,
                                n_estimators=run.n_estimators,
                                model_name=model_name,
                                config_signature=sig,
                            )
                        )

                    else:
                        raise ValueError(f"Unsupported final selected variant for target {target}: {variant}")
            continue

        for window_strategy in run.window_strategies:
            for variant in run.variants:
                for month, forecast_dates in month_groups.items():
                    recalibration_date = pd.Timestamp(month.start_time).normalize()
                    model_name = get_model_name(cfg, variant, window_strategy, active_window_count)

                    if variant == "rf_H_24h":
                        for target in run.targets:
                            for zone in run.zones:
                                task_id = make_task_id(
                                    run.mode_name,
                                    run.split,
                                    variant,
                                    window_strategy,
                                    target,
                                    zone,
                                    [zone],
                                    recalibration_date,
                                    sig,
                                )
                                tasks.append(
                                    TaskSpec(
                                        task_id=task_id,
                                        mode_name=run.mode_name,
                                        split=run.split,
                                        variant=variant,
                                        window_strategy=window_strategy,
                                        target=target,
                                        zone=zone,
                                        zones=(zone,),
                                        targets=(target,),
                                        recalibration_date=recalibration_date,
                                        forecast_dates=forecast_dates,
                                        n_estimators=run.n_estimators,
                                        model_name=model_name,
                                        config_signature=sig,
                                    )
                                )

                    elif variant in {"rf_Z_7zones", "rf_HZ_24h_7zones"}:
                        for target in run.targets:
                            task_id = make_task_id(
                                run.mode_name,
                                run.split,
                                variant,
                                window_strategy,
                                target,
                                None,
                                run.zones,
                                recalibration_date,
                                sig,
                            )
                            tasks.append(
                                TaskSpec(
                                    task_id=task_id,
                                    mode_name=run.mode_name,
                                    split=run.split,
                                    variant=variant,
                                    window_strategy=window_strategy,
                                    target=target,
                                    zone=None,
                                    zones=tuple(run.zones),
                                    targets=(target,),
                                    recalibration_date=recalibration_date,
                                    forecast_dates=forecast_dates,
                                    n_estimators=run.n_estimators,
                                    model_name=model_name,
                                    config_signature=sig,
                                )
                            )

                    elif variant == "rf_T_3targets":
                        task_id = make_task_id(
                            run.mode_name,
                            run.split,
                            variant,
                            window_strategy,
                            None,
                            None,
                            run.zones,
                            recalibration_date,
                            sig,
                        )
                        tasks.append(
                            TaskSpec(
                                task_id=task_id,
                                mode_name=run.mode_name,
                                split=run.split,
                                variant=variant,
                                window_strategy=window_strategy,
                                target=None,
                                zone=None,
                                zones=tuple(run.zones),
                                targets=tuple(run.targets),
                                recalibration_date=recalibration_date,
                                forecast_dates=forecast_dates,
                                n_estimators=run.n_estimators,
                                model_name=model_name,
                                config_signature=sig,
                            )
                        )

                    elif variant == "rf_HT_24h_3targets":
                        if not cfg.RUN_OPTIONAL_RF_HT:
                            logger.info("Skipping optional variant %s because RUN_OPTIONAL_RF_HT=False.", variant)
                            continue
                        for zone in run.zones:
                            task_id = make_task_id(
                                run.mode_name,
                                run.split,
                                variant,
                                window_strategy,
                                None,
                                zone,
                                [zone],
                                recalibration_date,
                                sig,
                            )
                            tasks.append(
                                TaskSpec(
                                    task_id=task_id,
                                    mode_name=run.mode_name,
                                    split=run.split,
                                    variant=variant,
                                    window_strategy=window_strategy,
                                    target=None,
                                    zone=zone,
                                    zones=(zone,),
                                    targets=tuple(run.targets),
                                    recalibration_date=recalibration_date,
                                    forecast_dates=forecast_dates,
                                    n_estimators=run.n_estimators,
                                    model_name=model_name,
                                    config_signature=sig,
                                )
                            )

    logger.info("Generated %s task(s).", len(tasks))
    return tasks


# =============================================================================
# 11. MODEL FITTING, PREPROCESSING AND IMPORTANCE
# =============================================================================


def make_one_hot_encoder(cfg: Config) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=cfg.OHE_SPARSE_OUTPUT)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=cfg.OHE_SPARSE_OUTPUT)


def build_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str], cfg: Config) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(cfg)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, list(numeric_features)))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, list(categorical_features)))

    if not transformers:
        raise ValueError("No features selected for preprocessing.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_rf_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    task: TaskSpec,
    cfg: Config,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features, cfg)
    rf = RandomForestRegressor(
        n_estimators=task.n_estimators,
        criterion=cfg.RF_CRITERION,
        max_depth=cfg.RF_MAX_DEPTH,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
        max_features=cfg.RF_MAX_FEATURES,
        bootstrap=cfg.RF_BOOTSTRAP,
        n_jobs=cfg.RF_N_JOBS,
        random_state=cfg.RF_RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("rf", rf)])


def get_transformed_feature_names(pipeline: Pipeline, raw_features: Sequence[str]) -> List[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:  # noqa: BLE001
        # Fallback: this should rarely be needed, but keeps the script robust.
        return [f"feature_{i}" for i in range(len(raw_features))]


def strip_transformer_prefix(transformed_name: str) -> str:
    if "__" in transformed_name:
        return transformed_name.split("__", 1)[1]
    return transformed_name


def map_transformed_to_raw(transformed_name: str, raw_features: Sequence[str]) -> str:
    stripped = strip_transformer_prefix(transformed_name)
    for raw in sorted(raw_features, key=len, reverse=True):
        if stripped == raw or stripped.startswith(f"{raw}_"):
            return raw
    return stripped


def base_feature_name(raw_feature: str) -> str:
    # Remove daily flattening suffix.
    raw = re.sub(r"_h\d{2}$", "", raw_feature)
    return raw


def extract_impurity_importance(
    pipeline: Pipeline,
    raw_features: Sequence[str],
    task: TaskSpec,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rf: RandomForestRegressor = pipeline.named_steps["rf"]
    transformed_names = get_transformed_feature_names(pipeline, raw_features)
    importances = getattr(rf, "feature_importances_", None)

    if importances is None:
        detailed = pd.DataFrame()
        aggregated = pd.DataFrame()
        return detailed, aggregated

    n = min(len(transformed_names), len(importances))
    detailed = pd.DataFrame(
        {
            "task_id": task.task_id,
            "mode": task.mode_name,
            "split": task.split,
            "variant": task.variant,
            "window_strategy": task.window_strategy,
            "target": task.target,
            "zone": task.zone,
            "recalibration_date": task.recalibration_date,
            "transformed_feature": transformed_names[:n],
            "importance": importances[:n],
        }
    )
    detailed["raw_feature"] = detailed["transformed_feature"].apply(lambda x: map_transformed_to_raw(x, raw_features))
    detailed["base_feature"] = detailed["raw_feature"].apply(base_feature_name)

    aggregated = (
        detailed.groupby(
            [
                "task_id",
                "mode",
                "split",
                "variant",
                "window_strategy",
                "target",
                "zone",
                "recalibration_date",
                "raw_feature",
                "base_feature",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(importance=("importance", "sum"))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return detailed, aggregated


def predict_with_optional_target_scaler(
    pipeline: Pipeline,
    X: pd.DataFrame,
    target_scaler: Optional[StandardScaler],
) -> np.ndarray:
    pred = pipeline.predict(X)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred)
    return finite_or_nan(pred)


def compute_manual_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    Y_original: np.ndarray,
    task: TaskSpec,
    cfg: Config,
    target_scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    """Optional permutation importance using global negative MAE logic.

    This function is deliberately simple and disabled by default because it can
    be expensive for rolling multi-output forests.
    """
    if X.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(cfg.PERMUTATION_RANDOM_STATE)
    if len(X) > cfg.PERMUTATION_MAX_ROWS:
        sample_idx = rng.choice(len(X), size=cfg.PERMUTATION_MAX_ROWS, replace=False)
        X_eval = X.iloc[sample_idx].reset_index(drop=True)
        Y_eval = Y_original[sample_idx]
    else:
        X_eval = X.reset_index(drop=True)
        Y_eval = Y_original

    baseline_pred = predict_with_optional_target_scaler(pipeline, X_eval, target_scaler)
    baseline_mae = mean_absolute_error(Y_eval, baseline_pred)

    rows: List[Dict[str, Any]] = []
    for feature in X_eval.columns:
        impacts = []
        for _ in range(cfg.PERMUTATION_N_REPEATS):
            X_perm = X_eval.copy()
            X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
            perm_pred = predict_with_optional_target_scaler(pipeline, X_perm, target_scaler)
            perm_mae = mean_absolute_error(Y_eval, perm_pred)
            impacts.append(perm_mae - baseline_mae)
        rows.append(
            {
                "task_id": task.task_id,
                "mode": task.mode_name,
                "split": task.split,
                "variant": task.variant,
                "window_strategy": task.window_strategy,
                "target": task.target,
                "zone": task.zone,
                "recalibration_date": task.recalibration_date,
                "feature": feature,
                "base_feature": base_feature_name(feature),
                "baseline_mae": baseline_mae,
                "mean_mae_increase": float(np.mean(impacts)),
                "std_mae_increase": float(np.std(impacts, ddof=1)) if len(impacts) > 1 else 0.0,
                "n_repeats": cfg.PERMUTATION_N_REPEATS,
                "n_rows": len(X_eval),
                "output_group": "global",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mae_increase", ascending=False)


# =============================================================================
# 12. PREDICTION RESHAPING AND ALIGNMENT
# =============================================================================


def panel_truth_lookup(panel: pd.DataFrame, targets: Sequence[str], zones: Sequence[str], dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    subset = filter_dates(panel[panel["zone"].isin(zones)], dates)
    cols = ["delivery_datetime_model", "delivery_date", "hour", "zone"] + [t for t in targets if t in subset.columns]
    long = subset[cols].melt(
        id_vars=["delivery_datetime_model", "delivery_date", "hour", "zone"],
        value_vars=[t for t in targets if t in subset.columns],
        var_name="target",
        value_name="y_true_panel",
    )
    return long


def reshape_predictions_to_long(
    task: TaskSpec,
    pred: np.ndarray,
    pred_bundle: DatasetBundle,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Convert any output structure back to target-zone-hour long format."""
    pred_array: np.ndarray = np.asarray(pred, dtype=float)
    if pred_array.ndim == 1:
        pred_array = pred_array.reshape(-1, 1)
    rows: List[Dict[str, Any]] = []

    if task.variant == "rf_H_24h":
        assert task.target is not None and task.zone is not None
        meta = pred_bundle.meta.reset_index(drop=True)
        for row_pos, (_, meta_row) in enumerate(meta.iterrows()):
            date = pd.Timestamp(meta_row["delivery_date"]).normalize()
            for h in range(1, 25):
                rows.append(
                    {
                        "target": task.target,
                        "zone": task.zone,
                        "delivery_date": date,
                        "hour": h,
                        "horizon": h,
                        "y_pred": safe_float(pred_array[row_pos, h - 1]),
                    }
                )

    elif task.variant == "rf_Z_7zones":
        assert task.target is not None
        meta = pred_bundle.meta.reset_index(drop=True)
        for row_pos, (_, meta_row) in enumerate(meta.iterrows()):
            dt = pd.Timestamp(meta_row["delivery_datetime_model"])
            date = pd.Timestamp(meta_row["delivery_date"]).normalize()
            h = int(meta_row["hour"])
            for j, z in enumerate(task.zones):
                rows.append(
                    {
                        "target": task.target,
                        "zone": z,
                        "delivery_datetime_model": dt,
                        "delivery_date": date,
                        "hour": h,
                        "horizon": h,
                        "y_pred": safe_float(pred_array[row_pos, j]),
                    }
                )

    elif task.variant == "rf_HZ_24h_7zones":
        assert task.target is not None
        meta = pred_bundle.meta.reset_index(drop=True)
        for row_pos, (_, meta_row) in enumerate(meta.iterrows()):
            date = pd.Timestamp(meta_row["delivery_date"]).normalize()
            idx = 0
            for z in task.zones:
                for h in range(1, 25):
                    rows.append(
                        {
                            "target": task.target,
                            "zone": z,
                            "delivery_date": date,
                            "hour": h,
                            "horizon": h,
                            "y_pred": safe_float(pred_array[row_pos, idx]),
                        }
                    )
                    idx += 1

    elif task.variant == "rf_T_3targets":
        meta = pred_bundle.meta.reset_index(drop=True)
        for row_pos, (_, meta_row) in enumerate(meta.iterrows()):
            dt = pd.Timestamp(meta_row["delivery_datetime_model"])
            date = pd.Timestamp(meta_row["delivery_date"]).normalize()
            h = int(meta_row["hour"])
            z = str(meta_row["zone"])
            for j, target in enumerate(task.targets):
                rows.append(
                    {
                        "target": target,
                        "zone": z,
                        "delivery_datetime_model": dt,
                        "delivery_date": date,
                        "hour": h,
                        "horizon": h,
                        "y_pred": safe_float(pred_array[row_pos, j]),
                    }
                )

    elif task.variant == "rf_HT_24h_3targets":
        assert task.zone is not None
        meta = pred_bundle.meta.reset_index(drop=True)
        for row_pos, (_, meta_row) in enumerate(meta.iterrows()):
            date = pd.Timestamp(meta_row["delivery_date"]).normalize()
            idx = 0
            for target in task.targets:
                for h in range(1, 25):
                    rows.append(
                        {
                            "target": target,
                            "zone": task.zone,
                            "delivery_date": date,
                            "hour": h,
                            "horizon": h,
                            "y_pred": safe_float(pred_array[row_pos, idx]),
                        }
                    )
                    idx += 1

    else:
        raise ValueError(f"Unknown variant: {task.variant}")

    long = pd.DataFrame(rows)
    if long.empty:
        return long

    # Add delivery_datetime_model if the variant only produced date/hour.
    if "delivery_datetime_model" not in long.columns or long["delivery_datetime_model"].isna().any():
        key_panel = panel[
            ["delivery_date", "hour", "zone", "delivery_datetime_model"]
        ].drop_duplicates()
        long = long.merge(key_panel, on=["delivery_date", "hour", "zone"], how="left", suffixes=("", "_panel"))
        if "delivery_datetime_model_panel" in long.columns:
            long["delivery_datetime_model"] = long["delivery_datetime_model"].where(
                long["delivery_datetime_model"].notna(),
                long["delivery_datetime_model_panel"],
            )
            long = long.drop(columns=["delivery_datetime_model_panel"])

    return long


def align_predictions_to_eval_index(
    pred_long: pd.DataFrame,
    eval_long: pd.DataFrame,
    task: TaskSpec,
) -> pd.DataFrame:
    """Align task predictions to the common evaluation index exactly."""
    if pred_long.empty:
        return pd.DataFrame(columns=OFFICIAL_PREDICTION_COLUMNS)

    date_set = set(task.forecast_dates)
    eval_sub = eval_long[
        (eval_long["split"] == task.split)
        & (eval_long["delivery_date"].isin(date_set))
        & (eval_long["zone"].isin(task.zones))
        & (eval_long["target"].isin(task.targets))
    ].copy()

    if task.target is not None:
        eval_sub = eval_sub[eval_sub["target"] == task.target]
    if task.zone is not None:
        eval_sub = eval_sub[eval_sub["zone"] == task.zone]

    merge_keys = ["target", "zone", "delivery_datetime_model"]
    pred_keep = pred_long[merge_keys + ["y_pred"]].drop_duplicates(merge_keys)
    out = eval_sub.merge(pred_keep, on=merge_keys, how="left")
    out.insert(0, "model", task.model_name)

    out = out[OFFICIAL_PREDICTION_COLUMNS].copy()
    out["hour"] = out["hour"].astype(int)
    out["horizon"] = out["horizon"].astype(int)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    return out


# =============================================================================
# 13. FIT/PREDICT ONE RECALIBRATION TASK
# =============================================================================


def run_data_leakage_checks(
    train_dates: Sequence[pd.Timestamp],
    task: TaskSpec,
    feature_names: Sequence[str],
    logger: logging.Logger,
) -> None:
    if not train_dates:
        raise ValueError(f"No training dates for task {task.task_id}")

    max_train_date = max(pd.Timestamp(d).normalize() for d in train_dates)
    min_forecast_date = min(task.forecast_dates)
    if max_train_date >= task.recalibration_date:
        raise ValueError(
            f"Leakage check failed for {task.task_id}: max training date {max_train_date.date()} "
            f"is not before recalibration date {task.recalibration_date.date()}."
        )
    if task.recalibration_date > min_forecast_date:
        logger.warning(
            "Task %s has recalibration date %s after first forecast date %s. Check date definitions.",
            task.task_id,
            task.recalibration_date.date(),
            min_forecast_date.date(),
        )

    unsafe_used = [f for f in feature_names if f in UNSAFE_SAME_DAY_MARKET_COLUMNS or f == "mti_f"]
    if unsafe_used:
        raise ValueError(
            f"Leakage check failed for {task.task_id}: unsafe same-day market columns used as features: {unsafe_used}"
        )


def feature_metadata_dict(
    task: TaskSpec,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    train_bundle: DatasetBundle,
    pred_bundle: DatasetBundle,
    pipeline: Optional[Pipeline],
    elapsed_seconds: float,
    cfg: Config,
) -> Dict[str, Any]:
    raw_features = list(train_bundle.numeric_features) + list(train_bundle.categorical_features)
    transformed_count: Optional[int] = None
    if pipeline is not None:
        try:
            transformed_count = len(get_transformed_feature_names(pipeline, raw_features))
        except Exception:  # noqa: BLE001
            transformed_count = None

    return {
        "task_id": task.task_id,
        "mode": task.mode_name,
        "split": task.split,
        "variant": task.variant,
        "model_name": task.model_name,
        "window_strategy": task.window_strategy,
        "target": task.target,
        "zone": task.zone,
        "zones": list(task.zones),
        "targets": list(task.targets),
        "recalibration_date": str(task.recalibration_date.date()),
        "training_start_date": str(train_start.date()),
        "training_end_exclusive": str(train_end.date()),
        "training_last_observed_date": str((train_end - pd.Timedelta(days=1)).date()),
        "forecast_start_date": str(min(task.forecast_dates).date()),
        "forecast_end_date": str(max(task.forecast_dates).date()),
        "n_training_rows": int(train_bundle.X.shape[0]),
        "n_prediction_rows": int(pred_bundle.X.shape[0]),
        "n_raw_features": int(len(raw_features)),
        "n_numeric_features": int(len(train_bundle.numeric_features)),
        "n_categorical_features": int(len(train_bundle.categorical_features)),
        "n_transformed_features": transformed_count,
        "output_dimension": int(train_bundle.Y.shape[1]) if train_bundle.Y.ndim == 2 else 1,
        "numeric_features": list(train_bundle.numeric_features),
        "categorical_features": list(train_bundle.categorical_features),
        "lag_features": list(train_bundle.X.attrs.get("lag_features", [])),
        "cross_zone_features": list(train_bundle.X.attrs.get("cross_zone_features", [])),
        "use_weather_for_delivery_day": cfg.USE_WEATHER_FOR_DELIVERY_DAY,
        "use_cross_zone_lags": cfg.USE_CROSS_ZONE_LAGS,
        "target_scaling_used": bool(train_bundle.target_scaling_required),
        "n_estimators": task.n_estimators,
        "max_features": cfg.RF_MAX_FEATURES,
        "min_samples_leaf": cfg.RF_MIN_SAMPLES_LEAF,
        "max_depth": cfg.RF_MAX_DEPTH,
        "elapsed_seconds": elapsed_seconds,
        "config_signature": task.config_signature,
    }


def fit_predict_rf_for_recalibration(
    task: TaskSpec,
    panel: pd.DataFrame,
    eval_long: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
    task_number: int,
    total_tasks: int,
) -> pd.DataFrame:
    paths = get_checkpoint_paths(cfg, task.task_id)

    if (
        cfg.RESUME_FROM_CHECKPOINTS
        and paths["prediction"].exists()
        and not cfg.OVERWRITE_EXISTING_CHECKPOINTS
    ):
        logger.info(
            "[%s/%s] Skipping existing checkpoint: %s",
            task_number,
            total_tasks,
            task.task_id,
        )
        return pd.read_parquet(paths["prediction"])

    logger.info(
        "[%s/%s] Running task=%s | variant=%s | split=%s | target=%s | zone=%s | recalibration=%s",
        task_number,
        total_tasks,
        task.task_id,
        task.variant,
        task.split,
        task.target,
        task.zone,
        task.recalibration_date.date(),
    )

    start_task = time.perf_counter()
    train_start, train_end, train_dates = get_training_dates(panel, task.recalibration_date, task.window_strategy, cfg)

    logger.info(
        "Training window for %s: [%s, %s) | n_train_dates=%s | forecast dates=%s to %s",
        task.task_id,
        train_start.date(),
        train_end.date(),
        len(train_dates),
        min(task.forecast_dates).date(),
        max(task.forecast_dates).date(),
    )

    try:
        train_bundle = build_dataset_for_task(panel, task, train_dates, cfg)
        pred_bundle = build_dataset_for_task(panel, task, task.forecast_dates, cfg)

        raw_features = list(train_bundle.numeric_features) + list(train_bundle.categorical_features)
        run_data_leakage_checks(train_dates, task, raw_features, logger)

        if train_bundle.X.shape[0] < cfg.MIN_TRAIN_ROWS:
            raise ValueError(
                f"Too few training rows for {task.task_id}: {train_bundle.X.shape[0]} < {cfg.MIN_TRAIN_ROWS}"
            )
        if pred_bundle.X.shape[0] < cfg.MIN_PREDICTION_ROWS:
            raise ValueError(f"No prediction rows for {task.task_id}.")
        if train_bundle.Y.ndim != 2 or train_bundle.Y.shape[1] == 0:
            raise ValueError(f"Invalid Y shape for {task.task_id}: {train_bundle.Y.shape}")

        # Ensure prediction set has the same raw feature columns as training.
        pred_bundle.X = pred_bundle.X.reindex(columns=train_bundle.X.columns)

        logger.info(
            "Task %s | X_train=%s | Y_train=%s | X_pred=%s | output_dim=%s | raw_features=%s",
            task.task_id,
            train_bundle.X.shape,
            train_bundle.Y.shape,
            pred_bundle.X.shape,
            train_bundle.Y.shape[1],
            len(raw_features),
        )

        y_train_model = train_bundle.Y
        target_scaler: Optional[StandardScaler] = None
        if train_bundle.target_scaling_required:
            target_scaler = StandardScaler()
            y_train_model = target_scaler.fit_transform(train_bundle.Y)

        pipeline = build_rf_pipeline(train_bundle.numeric_features, train_bundle.categorical_features, task, cfg)
        pipeline.fit(train_bundle.X, y_train_model)

        pred = predict_with_optional_target_scaler(pipeline, pred_bundle.X, target_scaler)
        pred_long_raw = reshape_predictions_to_long(task, pred, pred_bundle, panel)
        pred_long = align_predictions_to_eval_index(pred_long_raw, eval_long, task)

        elapsed = time.perf_counter() - start_task
        missing_pred = int(pred_long["y_pred"].isna().sum()) if not pred_long.empty else 0

        # Save model analysis checkpoints.
        if cfg.SAVE_IMPURITY_IMPORTANCE:
            detailed, aggregated = extract_impurity_importance(pipeline, raw_features, task)
            if not detailed.empty:
                atomic_write_parquet(detailed, paths["importance_detailed"], engine=cfg.PARQUET_ENGINE)
            if not aggregated.empty:
                atomic_write_parquet(aggregated, paths["importance_aggregated"], engine=cfg.PARQUET_ENGINE)

        if cfg.RUN_PERMUTATION_IMPORTANCE:
            perm = compute_manual_permutation_importance(
                pipeline,
                train_bundle.X,
                train_bundle.Y,
                task,
                cfg,
                target_scaler,
            )
            if not perm.empty:
                atomic_write_parquet(perm, paths["permutation"], engine=cfg.PARQUET_ENGINE)

        metadata = feature_metadata_dict(
            task,
            train_start,
            train_end,
            train_bundle,
            pred_bundle,
            pipeline,
            elapsed,
            cfg,
        )
        metadata["status"] = "success"
        metadata["n_predictions"] = int(len(pred_long))
        metadata["missing_predictions"] = missing_pred

        atomic_write_json(metadata, paths["feature_metadata"])
        atomic_write_json(metadata, paths["log"])
        atomic_write_parquet(train_bundle.output_mapping.assign(task_id=task.task_id), paths["output_mapping"], engine=cfg.PARQUET_ENGINE)

        if cfg.SAVE_FITTED_MODELS:
            ensure_dir(paths["model"].parent)
            tmp_model_path = paths["model"].with_name(f"{paths['model'].stem}.tmp{paths['model'].suffix}")
            joblib.dump(
                {
                    "pipeline": pipeline,
                    "target_scaler": target_scaler,
                    "task": task,
                    "metadata": metadata,
                },
                tmp_model_path,
            )
            os.replace(tmp_model_path, paths["model"])

        if cfg.SAVE_CHECKPOINTS:
            atomic_write_parquet(pred_long, paths["prediction"], engine=cfg.PARQUET_ENGINE)

        logger.info(
            "Finished task=%s | predictions=%s | missing_y_pred=%s | elapsed=%.2fs",
            task.task_id,
            len(pred_long),
            missing_pred,
            elapsed,
        )
        return pred_long

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start_task
        error_log = {
            "task_id": task.task_id,
            "mode": task.mode_name,
            "split": task.split,
            "variant": task.variant,
            "window_strategy": task.window_strategy,
            "target": task.target,
            "zone": task.zone,
            "zones": list(task.zones),
            "targets": list(task.targets),
            "recalibration_date": str(task.recalibration_date.date()),
            "status": "failed",
            "error": repr(exc),
            "elapsed_seconds": elapsed,
            "config_signature": task.config_signature,
        }
        atomic_write_json(error_log, paths["log"])
        logger.exception("Task failed: %s", task.task_id)
        return pd.DataFrame(columns=OFFICIAL_PREDICTION_COLUMNS)

    finally:
        if cfg.GARBAGE_COLLECT_AFTER_TASK:
            gc.collect()


# =============================================================================
# 14. COLLECTING CHECKPOINTS AND OUTPUTS
# =============================================================================


def collect_checkpoint_predictions(tasks: Sequence[TaskSpec], cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for task in tasks:
        path = get_checkpoint_paths(cfg, task.task_id)["prediction"]
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(task.task_id)
    if missing:
        logger.warning("Missing prediction checkpoints for %s task(s).", len(missing))
    if not frames:
        return pd.DataFrame(columns=OFFICIAL_PREDICTION_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out


def collect_checkpoint_table(
    tasks: Sequence[TaskSpec],
    cfg: Config,
    key: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for task in tasks:
        path = get_checkpoint_paths(cfg, task.task_id)[key]
        if path.exists():
            try:
                if path.suffix == ".parquet":
                    frames.append(pd.read_parquet(path))
                elif path.suffix == ".csv":
                    frames.append(pd.read_csv(path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read checkpoint table %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_json_logs(tasks: Sequence[TaskSpec], cfg: Config, key: str, logger: logging.Logger) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        path = get_checkpoint_paths(cfg, task.task_id)[key]
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read JSON log %s: %s", path, exc)
    return pd.DataFrame(rows)


def verify_official_prediction_columns(pred: pd.DataFrame) -> None:
    missing = [col for col in OFFICIAL_PREDICTION_COLUMNS if col not in pred.columns]
    if missing:
        raise ValueError(f"Final prediction output is missing official columns: {missing}")


def quality_checks(pred: pd.DataFrame, eval_long: pd.DataFrame, cfg: Config, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    verify_official_prediction_columns(pred)

    pred = pred.copy()
    for col in ["forecast_date", "delivery_date"]:
        pred[col] = to_timestamp_date(pred[col])
    pred["delivery_datetime_model"] = pd.to_datetime(pred["delivery_datetime_model"])
    pred["hour"] = pred["hour"].astype(int)
    pred["horizon"] = pred["horizon"].astype(int)

    # Remove exact duplicate official rows, preserving the first prediction.
    duplicate_key = ["model", "target", "zone", "split", "delivery_datetime_model"]
    duplicate_mask = pred.duplicated(duplicate_key, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        logger.warning("Removing %s duplicate prediction rows by %s.", duplicate_count, duplicate_key)
        pred.drop_duplicates(duplicate_key, keep="first", inplace=True)

    unexpected_horizons = sorted(set(pred["horizon"].dropna().astype(int)) - set(range(1, 25)))
    if unexpected_horizons:
        logger.warning("Unexpected horizons found: %s", unexpected_horizons)

    counts = (
        pred.groupby(["model", "target", "zone", "split"], dropna=False)
        .size()
        .reset_index(name="n_predictions")
        .sort_values(["model", "target", "zone", "split"])
    )

    missing_counts = (
        pred.groupby(["model", "target", "zone", "split"], dropna=False)["y_pred"]
        .apply(lambda s: int(s.isna().sum()))
        .reset_index(name="missing_y_pred")
    )

    duplicate_summary = pd.DataFrame(
        {
            "duplicate_key": [", ".join(duplicate_key)],
            "duplicate_predictions_removed": [duplicate_count],
        }
    )

    horizon_distribution = (
        pred.groupby(["model", "split", "horizon"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["model", "split", "horizon"])
    )

    date_ranges = (
        pred.groupby(["model", "split"], dropna=False)
        .agg(
            min_delivery_date=("delivery_date", "min"),
            max_delivery_date=("delivery_date", "max"),
            min_forecast_date=("forecast_date", "min"),
            max_forecast_date=("forecast_date", "max"),
        )
        .reset_index()
    )

    # Missing expected combinations within the subset each model attempts.
    missing_combo_rows: List[Dict[str, Any]] = []
    for model_name, model_df in pred.groupby("model"):
        attempted_targets = sorted(model_df["target"].unique())
        attempted_zones = sorted(model_df["zone"].unique())
        attempted_splits = sorted(model_df["split"].unique())
        attempted_dates = sorted(model_df["delivery_date"].unique())
        expected = eval_long[
            (eval_long["target"].isin(attempted_targets))
            & (eval_long["zone"].isin(attempted_zones))
            & (eval_long["split"].isin(attempted_splits))
            & (eval_long["delivery_date"].isin(attempted_dates))
        ]
        expected_keys = expected[["target", "zone", "split", "delivery_datetime_model"]].drop_duplicates()
        got_keys = model_df[["target", "zone", "split", "delivery_datetime_model"]].drop_duplicates()
        merged = expected_keys.merge(got_keys, on=["target", "zone", "split", "delivery_datetime_model"], how="left", indicator=True)
        n_missing = int((merged["_merge"] == "left_only").sum())
        missing_combo_rows.append({"model": model_name, "missing_expected_rows": n_missing})
    missing_combinations = pd.DataFrame(missing_combo_rows)

    logger.info("Prediction count summary:\n%s", counts.to_string(index=False))
    logger.info("Missing y_pred summary:\n%s", missing_counts.to_string(index=False))
    logger.info("Date ranges:\n%s", date_ranges.to_string(index=False))

    return {
        "pred_clean": pred,
        "prediction_counts": counts,
        "missing_predictions": missing_counts,
        "duplicate_predictions": duplicate_summary,
        "horizon_distribution": horizon_distribution,
        "date_ranges": date_ranges,
        "missing_expected_combinations": missing_combinations,
    }


def save_quality_outputs(checks: Dict[str, pd.DataFrame], cfg: Config) -> None:
    out_dir = cfg.EXPERIMENT_DIR / "quality_checks"
    ensure_dir(out_dir)
    for name, df in checks.items():
        if name == "pred_clean":
            continue
        atomic_write_csv(df, out_dir / f"rf_{name}.csv")


def save_model_analysis_outputs(tasks: Sequence[TaskSpec], cfg: Config, logger: logging.Logger) -> None:
    analysis_dir = cfg.EXPERIMENT_DIR
    ensure_dir(analysis_dir)

    detailed = collect_checkpoint_table(tasks, cfg, "importance_detailed", logger)
    if not detailed.empty:
        atomic_write_csv(detailed, analysis_dir / "rf_feature_importance_detailed.csv")

    aggregated = collect_checkpoint_table(tasks, cfg, "importance_aggregated", logger)
    if not aggregated.empty:
        atomic_write_csv(aggregated, analysis_dir / "rf_feature_importance_aggregated.csv")

    permutation = collect_checkpoint_table(tasks, cfg, "permutation", logger)
    if not permutation.empty:
        atomic_write_csv(permutation, analysis_dir / "rf_permutation_importance.csv")

    feature_meta = collect_json_logs(tasks, cfg, "feature_metadata", logger)
    if not feature_meta.empty:
        # Keep list-like columns as JSON strings for readable CSV output.
        for col in ["zones", "targets", "numeric_features", "categorical_features", "lag_features", "cross_zone_features"]:
            if col in feature_meta.columns:
                feature_meta[col] = feature_meta[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        atomic_write_csv(feature_meta, analysis_dir / "rf_feature_metadata.csv")
        # Alias requested in the prompt.
        atomic_write_csv(feature_meta, analysis_dir / "rf_feature_log.csv")

    output_mapping = collect_checkpoint_table(tasks, cfg, "output_mapping", logger)
    if not output_mapping.empty:
        output_mapping = output_mapping.drop_duplicates()
        atomic_write_csv(output_mapping, analysis_dir / "rf_output_mapping.csv")

    run_log = collect_json_logs(tasks, cfg, "log", logger)
    if not run_log.empty:
        for col in ["zones", "targets", "numeric_features", "categorical_features", "lag_features", "cross_zone_features"]:
            if col in run_log.columns:
                run_log[col] = run_log[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        atomic_write_csv(run_log, analysis_dir / "rf_run_log.csv")
        # Recalibration log and computation times are useful subsets of the same information.
        recalib_cols = [
            c
            for c in [
                "task_id",
                "mode",
                "split",
                "variant",
                "model_name",
                "window_strategy",
                "target",
                "zone",
                "recalibration_date",
                "training_start_date",
                "training_end_exclusive",
                "forecast_start_date",
                "forecast_end_date",
                "n_training_rows",
                "n_prediction_rows",
                "output_dimension",
                "status",
                "error",
            ]
            if c in run_log.columns
        ]
        if recalib_cols:
            atomic_write_csv(run_log[recalib_cols], analysis_dir / "rf_recalibration_log.csv")

        time_cols = [
            c
            for c in [
                "task_id",
                "mode",
                "split",
                "variant",
                "model_name",
                "window_strategy",
                "target",
                "zone",
                "recalibration_date",
                "n_training_rows",
                "output_dimension",
                "elapsed_seconds",
                "status",
            ]
            if c in run_log.columns
        ]
        if time_cols:
            atomic_write_csv(run_log[time_cols], analysis_dir / "rf_computation_times.csv")

        structure_cols = [
            c
            for c in [
                "model_name",
                "variant",
                "window_strategy",
                "split",
                "target",
                "zone",
                "output_dimension",
                "n_raw_features",
                "n_transformed_features",
                "n_training_rows",
                "n_predictions",
                "missing_predictions",
                "elapsed_seconds",
                "status",
            ]
            if c in run_log.columns
        ]
        if structure_cols:
            structure_results = run_log[structure_cols].copy()
            atomic_write_csv(structure_results, analysis_dir / "rf_output_structure_results.csv")


# =============================================================================
# 15. QUICK DIAGNOSTIC METRICS
# =============================================================================


def load_weekly_naive_predictions(cfg: Config, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if cfg.WEEKLY_NAIVE_PARQUET_PATH.exists():
        logger.info("Reading weekly naive predictions from parquet: %s", cfg.WEEKLY_NAIVE_PARQUET_PATH)
        return pd.read_parquet(cfg.WEEKLY_NAIVE_PARQUET_PATH)
    if cfg.WEEKLY_NAIVE_RDS_PATH.exists():
        logger.info("Reading weekly naive predictions from RDS: %s", cfg.WEEKLY_NAIVE_RDS_PATH)
        return read_rds(cfg.WEEKLY_NAIVE_RDS_PATH)
    logger.info("Weekly naive predictions not found; skipping quick rMAE diagnostics.")
    return None


def compute_quick_metrics(pred: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()

    df = pred.dropna(subset=["y_true", "y_pred"]).copy()
    if df.empty:
        return pd.DataFrame()

    metrics = (
        df.groupby(["model", "target", "zone", "split"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
                    "RMSE": math.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                    "Bias": float(np.mean(g["y_pred"] - g["y_true"])),
                }
            )
        )
        .reset_index()
    )

    naive = load_weekly_naive_predictions(cfg, logger)
    if naive is not None and not naive.empty:
        naive = naive.copy()
        if "delivery_datetime_model" in naive.columns:
            naive["delivery_datetime_model"] = pd.to_datetime(naive["delivery_datetime_model"])
        required = ["target", "zone", "split", "delivery_datetime_model", "y_true", "y_pred"]
        if all(col in naive.columns for col in required):
            naive_clean = naive.dropna(subset=["y_true", "y_pred"]).copy()
            naive_mae = (
                naive_clean.groupby(["target", "zone", "split"], dropna=False)
                .apply(lambda g: mean_absolute_error(g["y_true"], g["y_pred"]))
                .reset_index(name="naive_weekly_MAE")
            )
            metrics = metrics.merge(naive_mae, on=["target", "zone", "split"], how="left")
            metrics["rMAE_vs_weekly_naive"] = metrics["MAE"] / metrics["naive_weekly_MAE"]
        else:
            logger.warning("Weekly naive file exists but does not contain required columns for quick rMAE: %s", required)

    atomic_write_csv(metrics, cfg.EXPERIMENT_DIR / "rf_quick_validation_metrics.csv")
    logger.info("Saved quick diagnostic metrics: %s", cfg.EXPERIMENT_DIR / "rf_quick_validation_metrics.csv")
    return metrics


# =============================================================================
# 16. SAVE FINAL OUTPUTS
# =============================================================================


def save_outputs(pred: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    if pred.empty and cfg.FAIL_ON_EMPTY_FINAL_OUTPUT:
        raise ValueError("Final prediction output is empty. Check task logs in experiments/random_forest_multioutput/checkpoints/logs/.")

    verify_official_prediction_columns(pred)

    # Final ordering for stable R compatibility.
    pred = pred[OFFICIAL_PREDICTION_COLUMNS].copy()
    pred["forecast_date"] = pd.to_datetime(pred["forecast_date"]).dt.date
    pred["delivery_date"] = pd.to_datetime(pred["delivery_date"]).dt.date
    pred = pred.sort_values(["model", "split", "target", "zone", "delivery_datetime_model"]).reset_index(drop=True)

    ensure_dir(cfg.PREDICTIONS_DIR)
    atomic_write_parquet(pred, cfg.FINAL_PREDICTION_PARQUET, engine=cfg.PARQUET_ENGINE)
    logger.info("Saved final parquet predictions: %s | rows=%s", cfg.FINAL_PREDICTION_PARQUET, len(pred))

    if cfg.TRY_WRITE_RDS_OUTPUT:
        maybe_write_rds(pred, cfg.FINAL_PREDICTION_RDS, logger)




def check_runtime_dependencies(cfg: Config, logger: logging.Logger) -> None:
    """Fail early with clear messages for required runtime dependencies."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to write parquet outputs compatible with the R evaluation pipeline. "
            "Install it with `pip install pyarrow` or `conda install pyarrow`."
        ) from exc


# =============================================================================
# 17. MAIN EXECUTION
# =============================================================================


def main() -> None:
    cfg = CFG
    logger = setup_logging(cfg)
    start_all = time.perf_counter()

    ensure_dir(cfg.EXPERIMENT_DIR)
    ensure_dir(cfg.PREDICTIONS_DIR)

    logger.info("Starting Random Forest multi-output script.")
    check_runtime_dependencies(cfg, logger)
    logger.info("Configuration snapshot saved to %s", cfg.EXPERIMENT_DIR / "rf_config.json")
    atomic_write_json({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, cfg.EXPERIMENT_DIR / "rf_config.json")

    runs = build_run_specs(cfg)
    validate_run_specs(runs, cfg)
    logger.info("Active execution modes: %s", [r.mode_name for r in runs])

    panel_raw, eval_raw = load_inputs(cfg, logger)
    panel = prepare_panel(panel_raw, cfg, logger)
    panel = create_lag_features(panel, cfg, logger)
    eval_long = prepare_eval_index(eval_raw, panel, cfg, logger)

    # Keep only relevant validation/test rows to reduce memory.
    eval_long = eval_long[eval_long["split"].isin({"validation", "test"})].copy()

    tasks = generate_tasks(eval_long, runs, cfg, logger)
    if not tasks:
        raise ValueError("No tasks were generated. Check execution modes and evaluation index dates.")

    all_predictions: List[pd.DataFrame] = []
    total_tasks = len(tasks)
    for i, task in enumerate(tasks, start=1):
        pred_task = fit_predict_rf_for_recalibration(
            task=task,
            panel=panel,
            eval_long=eval_long,
            cfg=cfg,
            logger=logger,
            task_number=i,
            total_tasks=total_tasks,
        )
        if not pred_task.empty:
            all_predictions.append(pred_task)

    # Final aggregation from checkpoints, not only memory, so resumed runs are complete.
    pred_combined = collect_checkpoint_predictions(tasks, cfg, logger)
    if pred_combined.empty and all_predictions:
        pred_combined = pd.concat(all_predictions, ignore_index=True)

    checks = quality_checks(pred_combined, eval_long, cfg, logger)
    pred_clean = checks["pred_clean"]
    save_quality_outputs(checks, cfg)

    save_outputs(pred_clean, cfg, logger)
    save_model_analysis_outputs(tasks, cfg, logger)

    if cfg.COMPUTE_QUICK_METRICS:
        compute_quick_metrics(pred_clean, cfg, logger)

    elapsed_all = time.perf_counter() - start_all
    logger.info("Random Forest multi-output script completed in %.2f seconds.", elapsed_all)


if __name__ == "__main__":
    main()
