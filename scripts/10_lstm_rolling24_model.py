#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_lstm_rolling24_model.py

"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset



# =============================================================================
# 1. CONFIGURATION
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PROJECT_ROOT = Path.cwd()


@dataclass
class LSTMStrategy:
    """High-level LSTM strategy used in validation and final testing."""

    strategy_id: str
    window_type: str  # "expanding" or "rolling"
    lookback_hours: int
    window_months: Optional[int]
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int


@dataclass
class Config:
    """Central configuration for the LSTM experiment."""

    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------
    PROJECT_ROOT: Path = _DEFAULT_PROJECT_ROOT

    PANEL_PARQUET_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/processed/gme_model_panel_weather_hourly.parquet"
    PANEL_RDS_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/processed/gme_model_panel_weather_hourly.rds"

    EVAL_INDEX_PARQUET_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/evaluation/eval_index_hourly.parquet"
    EVAL_INDEX_RDS_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/evaluation/eval_index_hourly.rds"

    WEEKLY_NAIVE_PARQUET_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_naive_week_before.parquet"
    WEEKLY_NAIVE_RDS_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_naive_week_before.rds"

    PREDICTIONS_DIR: Path = _DEFAULT_PROJECT_ROOT / "data/predictions"
    FINAL_PREDICTION_PARQUET: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_lstm_rolling24.parquet"
    FINAL_PREDICTION_RDS: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_lstm_rolling24.rds"

    EXPERIMENT_DIR: Path = _DEFAULT_PROJECT_ROOT / "experiments/lstm_rolling24_norecal"

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------
    # If several are True, RUN_SMOKE_TEST has priority and runs alone.
    RUN_SMOKE_TEST: bool = False
    RUN_FAST_VALIDATION: bool = False
    RUN_FULL_VALIDATION: bool = False
    # Default: run only the 2025 test with the compact selected rolling-24m strategies.
    RUN_FINAL_TEST: bool = True

    # ------------------------------------------------------------------
    # Checkpoints / resume
    # ------------------------------------------------------------------
    SAVE_CHECKPOINTS: bool = True
    RESUME_FROM_CHECKPOINTS: bool = True
    OVERWRITE_EXISTING_CHECKPOINTS: bool = False
    SAVE_FITTED_MODELS: bool = False

    # ------------------------------------------------------------------
    # Core problem definition
    # ------------------------------------------------------------------
    MODEL_NAME_FINAL: str = "lstm_rolling24_selected"
    MODEL_NAME_SMOKE: str = "lstm_smoke"
    TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    ZONES: Tuple[str, ...] = ("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")

    VALIDATION_YEAR: int = 2024
    TEST_YEAR: int = 2025
    INITIAL_TRAIN_START_DATE: str = "2021-01-01"
    RECALIBRATION_FREQUENCY: str = "none"  # currently implemented as monthly blocks

    INNER_VALIDATION_DAYS: int = 30
    MIN_TRAIN_SAMPLES: int = 60
    MIN_INNER_TRAIN_SAMPLES: int = 30
    MIN_INNER_VALIDATION_SAMPLES: int = 7

    # ------------------------------------------------------------------
    # Deep learning defaults
    # ------------------------------------------------------------------
    LOOKBACK_OPTIONS: Tuple[int, ...] = (168, 336)
    HIDDEN_SIZE_OPTIONS: Tuple[int, ...] = (32, 64)
    NUM_LAYERS_OPTIONS: Tuple[int, ...] = (1, 2)
    LEARNING_RATE_OPTIONS: Tuple[float, ...] = (1e-3, 5e-4)
    BATCH_SIZE_OPTIONS: Tuple[int, ...] = (32, 64)
    # Dropout is not searched independently: it is set to 0.0 for one-layer
    # LSTMs and 0.2 for two-layer LSTMs.
    DROPOUT_ONE_LAYER: float = 0.0
    DROPOUT_TWO_LAYERS: float = 0.2

    MAX_EPOCHS: int = 50
    PATIENCE: int = 7
    MIN_DELTA: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    GRADIENT_CLIP_NORM: float = 1.0

    # Device: CPU is safest/reproducible. Set ALLOW_CUDA/MPS True if desired.
    ALLOW_CUDA: bool = False
    ALLOW_MPS: bool = False
    RANDOM_SEED: int = 123
    NUM_WORKERS: int = 0

    # ------------------------------------------------------------------
    # Smoke / fast limits
    # ------------------------------------------------------------------
    SMOKE_TARGETS: Tuple[str, ...] = ("price",)
    SMOKE_ZONES: Tuple[str, ...] = ("NORD", "CSUD")
    SMOKE_MAX_FORECAST_DATES: int = 14

    FAST_TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    FAST_ZONES: Tuple[str, ...] = ("NORD", "CSUD", "SICI")
    FAST_MAX_VALIDATION_MONTHS: Optional[int] = 2
    FAST_MAX_FORECAST_DATES_PER_SERIES: Optional[int] = 31
    FAST_MAX_EPOCHS: int = 8

    # ------------------------------------------------------------------
    # Final strategy selection
    # ------------------------------------------------------------------
    # Selected from the previous expanding-window validation, but re-estimated
    # here with a 24-month rolling training window to reduce the influence of
    # the 2022 electricity-price crisis in the 2025 final test.
    SELECTED_STRATEGY_BY_TARGET = {
        "price": "lstm_rolling24_168h_h64_l2_lr5em04_b32",
        "purchases": "lstm_rolling24_336h_h64_l1_lr5em04_b32",
        "sales": "lstm_rolling24_336h_h64_l2_lr5em04_b32",
    }
    SELECTED_STRATEGY_BY_TARGET_ZONE: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Feature blocks
    # ------------------------------------------------------------------
    # These are part of the LSTM model design, not optional feature-selection flags.
    OWN_ZONE_VARIABLES: Tuple[str, ...] = (
        "price",
        "purchases",
        "sales",
        "hhi",
        "rsi",
        "mti",
        "temperature_2m",
        "wind_speed_100m",
        "shortwave_radiation",
    )
    COMMON_MARKET_VARIABLES: Tuple[str, ...] = (
        "pun",
        "purchases_italy",
        "sales_italy",
        "unsold_italy",
        "purchases_external_total",
        "sales_external_total",
        "purchases_external_n_active_areas",
        "sales_external_n_active_areas",
    )
    CROSS_ZONE_VARIABLES: Tuple[str, ...] = (
        "price",
        "purchases",
        "sales",
        "hhi",
        "rsi",
        "mti",
    )
    WEATHER_VARIABLES: Tuple[str, ...] = (
        "temperature_2m",
        "wind_speed_100m",
        "shortwave_radiation",
    )

    REQUIRED_PREDICTION_COLUMNS: Tuple[str, ...] = (
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
    )


# =============================================================================
# 2. LOGGING AND BASIC UTILITIES
# =============================================================================


def safe_filename(value: str) -> str:
    """Return a filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config) -> logging.Logger:
    ensure_dir(cfg.EXPERIMENT_DIR / "logs")
    log_file = cfg.EXPERIMENT_DIR / "logs" / "lstm_run.log"

    logger = logging.getLogger("lstm_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Config) -> torch.device:
    if cfg.ALLOW_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.ALLOW_MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def serialize_config(cfg: Config) -> Dict[str, Any]:
    out = asdict(cfg)
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out


# =============================================================================
# 3. I/O FUNCTIONS
# =============================================================================


def read_rds_with_pyreadr(path: Path) -> pd.DataFrame:
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Cannot read {path}. pyreadr is not installed and parquet fallback was not available."
        ) from exc

    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"pyreadr could not read any object from {path}")
    return next(iter(result.values()))


def load_table_prefer_parquet(parquet_path: Path, rds_path: Path, table_name: str) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if rds_path.exists():
        return read_rds_with_pyreadr(rds_path)
    raise FileNotFoundError(
        f"Missing {table_name}. Expected either {parquet_path} or {rds_path}."
    )


def load_panel(cfg: Config) -> pd.DataFrame:
    panel = load_table_prefer_parquet(
        cfg.PANEL_PARQUET_PATH,
        cfg.PANEL_RDS_PATH,
        "processed panel",
    )
    return panel


def load_eval_index(cfg: Config) -> pd.DataFrame:
    eval_index = load_table_prefer_parquet(
        cfg.EVAL_INDEX_PARQUET_PATH,
        cfg.EVAL_INDEX_RDS_PATH,
        "evaluation index",
    )
    return eval_index


def load_weekly_naive(cfg: Config, logger: logging.Logger) -> Optional[pd.DataFrame]:
    try:
        return load_table_prefer_parquet(
            cfg.WEEKLY_NAIVE_PARQUET_PATH,
            cfg.WEEKLY_NAIVE_RDS_PATH,
            "weekly naive predictions",
        )
    except Exception as exc:
        logger.warning(
            "Weekly naive predictions were not loaded. Quick rMAE will be skipped. "
            "Official rMAE can still be computed by 05_evaluation_metrics.R. Error: %s",
            exc,
        )
        return None


def save_dataframe_outputs(df: pd.DataFrame, parquet_path: Path, rds_path: Optional[Path], logger: logging.Logger) -> None:
    ensure_dir(parquet_path.parent)
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet: %s", parquet_path)

    if rds_path is None:
        return
    try:
        import pyreadr  # type: ignore

        ensure_dir(rds_path.parent)
        pyreadr.write_rds(str(rds_path), df)
        logger.info("Saved RDS: %s", rds_path)
    except Exception as exc:
        logger.warning("Could not save RDS output %s: %s", rds_path, exc)


# =============================================================================
# 4. DATA PREPARATION
# =============================================================================


def require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def normalize_datetime_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def prepare_panel(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clean dtypes and create calendar variables used by the LSTM."""
    require_columns(
        panel,
        ["datetime_model", "date", "hour", "zone", "price", "purchases", "sales"],
        "panel",
    )

    panel = panel.copy()
    panel["datetime_model"] = normalize_datetime_column(panel["datetime_model"])
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.date
    panel["date"] = pd.to_datetime(panel["date"])
    panel["hour"] = panel["hour"].astype(int)
    panel["zone"] = panel["zone"].astype(str)

    if "mti" in panel.columns:
        panel["mti"] = panel["mti"].astype("object").where(panel["mti"].notna(), np.nan)

    # Calendar variables. Hour is 1,...,24 in the normalized local delivery grid.
    panel["weekday"] = panel["date"].dt.weekday + 1  # Monday=1, Sunday=7
    panel["month"] = panel["date"].dt.month
    panel["is_weekend"] = panel["weekday"].isin([6, 7]).astype(float)

    panel["sin_hour"] = np.sin(2.0 * np.pi * (panel["hour"].astype(float) - 1.0) / 24.0)
    panel["cos_hour"] = np.cos(2.0 * np.pi * (panel["hour"].astype(float) - 1.0) / 24.0)
    panel["sin_weekday"] = np.sin(2.0 * np.pi * (panel["weekday"].astype(float) - 1.0) / 7.0)
    panel["cos_weekday"] = np.cos(2.0 * np.pi * (panel["weekday"].astype(float) - 1.0) / 7.0)
    panel["sin_month"] = np.sin(2.0 * np.pi * (panel["month"].astype(float) - 1.0) / 12.0)
    panel["cos_month"] = np.cos(2.0 * np.pi * (panel["month"].astype(float) - 1.0) / 12.0)

    # If a holiday column is present, keep it as numeric if possible. Otherwise ignore it later.
    if "holiday" in panel.columns:
        panel["holiday"] = pd.to_numeric(panel["holiday"], errors="coerce").fillna(0.0)

    # Keep only physical zones used in the thesis.
    panel = panel[panel["zone"].isin(cfg.ZONES)].copy()
    panel = panel.sort_values(["zone", "datetime_model"]).reset_index(drop=True)
    return panel


def prepare_eval_index(eval_index: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        eval_index,
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
        ],
        "eval_index",
    )
    out = eval_index.copy()
    out["target"] = out["target"].astype(str)
    out["zone"] = out["zone"].astype(str)
    out["split"] = out["split"].astype(str)
    out["forecast_date"] = pd.to_datetime(out["forecast_date"], errors="coerce")
    out["delivery_datetime_model"] = normalize_datetime_column(out["delivery_datetime_model"])
    out["delivery_date"] = pd.to_datetime(out["delivery_date"], errors="coerce")
    out["hour"] = out["hour"].astype(int)
    out["horizon"] = out["horizon"].astype(int)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    return out


def available_columns(columns: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [col for col in columns if col in df.columns]


def build_historical_feature_frame(panel: pd.DataFrame, target_zone: str, cfg: Config) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Build the raw historical feature matrix for one target-zone task.

    Rows are hourly timestamps. Columns include own-zone variables, common market
    variables, calendar variables and cross-zone variables from other zones.
    """
    calendar_cols = [
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_month",
        "cos_month",
        "is_weekend",
    ]
    if "holiday" in panel.columns:
        calendar_cols.append("holiday")

    own_vars = available_columns(cfg.OWN_ZONE_VARIABLES, panel)
    common_vars = available_columns(cfg.COMMON_MARKET_VARIABLES, panel)
    cross_vars = available_columns(cfg.CROSS_ZONE_VARIABLES, panel)

    zone_df = panel[panel["zone"] == target_zone].copy()
    if zone_df.empty:
        raise ValueError(f"No panel rows found for zone {target_zone}")

    base_cols = ["datetime_model"] + own_vars + common_vars + calendar_cols
    base_cols = list(dict.fromkeys(base_cols))
    hist = zone_df[base_cols].copy()
    rename_map: Dict[str, str] = {}
    for col in own_vars:
        rename_map[col] = f"own_{col}"
    for col in common_vars:
        rename_map[col] = f"common_{col}"
    for col in calendar_cols:
        rename_map[col] = f"cal_{col}"
    hist = hist.rename(columns=rename_map).set_index("datetime_model").sort_index()

    other_zones = [zone for zone in cfg.ZONES if zone != target_zone]
    cross_df = panel[panel["zone"].isin(other_zones)][["datetime_model", "zone"] + cross_vars].copy()

    for var in cross_vars:
        wide = cross_df.pivot(index="datetime_model", columns="zone", values=var)
        wide = wide.reindex(columns=other_zones)
        wide.columns = [f"cross_{zone}_{var}" for zone in wide.columns]
        hist = hist.join(wide, how="left")

    hist = hist.sort_index()
    hist = hist[~hist.index.duplicated(keep="first")]

    categorical_cols = [col for col in hist.columns if col.endswith("_mti") or col == "own_mti" or "_mti" in col]
    categorical_cols = [col for col in categorical_cols if col in hist.columns]

    numeric_cols = [col for col in hist.columns if col not in categorical_cols]
    for col in numeric_cols:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    for col in categorical_cols:
        hist[col] = hist[col].astype("object").where(hist[col].notna(), np.nan)

    return hist, numeric_cols, categorical_cols


def get_future_covariate_columns(panel: pd.DataFrame, cfg: Config) -> List[str]:
    cols = [
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_month",
        "cos_month",
        "is_weekend",
    ]
    if "holiday" in panel.columns:
        cols.append("holiday")
    cols += available_columns(cfg.WEATHER_VARIABLES, panel)
    return list(dict.fromkeys(cols))


@dataclass
class RawSample:
    delivery_date: pd.Timestamp
    forecast_date: pd.Timestamp
    history_positions: np.ndarray
    future_covariates: Dict[str, float]
    y: np.ndarray


@dataclass
class SampleCollection:
    samples: List[RawSample]
    future_covariate_columns: List[str]

    def dates(self) -> List[pd.Timestamp]:
        return [s.delivery_date for s in self.samples]


def build_future_covariates_for_date(
    zone_panel: pd.DataFrame,
    delivery_date: pd.Timestamp,
    future_cols: Sequence[str],
) -> Optional[Dict[str, float]]:
    day_rows = zone_panel[zone_panel["date"] == delivery_date].sort_values("hour")
    if len(day_rows) != 24 or set(day_rows["hour"].astype(int)) != set(range(1, 25)):
        return None

    out: Dict[str, float] = {}
    for _, row in day_rows.iterrows():
        hour = int(row["hour"])
        for col in future_cols:
            out[f"future_{col}_h{hour:02d}"] = row.get(col, np.nan)
    return out


def build_daily_samples_for_target_zone(
    panel: pd.DataFrame,
    hist_features: pd.DataFrame,
    target: str,
    zone: str,
    lookback_hours: int,
    delivery_dates: Sequence[pd.Timestamp],
    cfg: Config,
) -> SampleCollection:
    """Build raw daily samples for one target-zone and a given lookback.

    The historical window for delivery day d ends at hour 24 of forecast_date=d-1.
    The target vector contains the 24 hourly values of delivery day d.
    """
    require_columns(panel, [target], "panel")

    zone_panel = panel[panel["zone"] == zone].copy()
    zone_panel = zone_panel.sort_values(["date", "hour"])
    future_cols_base = get_future_covariate_columns(zone_panel, cfg)

    target_pivot = zone_panel.pivot(index="date", columns="hour", values=target)
    target_pivot = target_pivot.reindex(columns=range(1, 25))

    hist_index = pd.Index(hist_features.index)
    timestamp_to_pos = {ts: i for i, ts in enumerate(hist_index)}

    samples: List[RawSample] = []
    expected_delta = pd.Timedelta(hours=1)

    for raw_date in delivery_dates:
        delivery_date = pd.Timestamp(raw_date).normalize()
        forecast_date = delivery_date - pd.Timedelta(days=1)
        hist_end = forecast_date + pd.Timedelta(hours=23)
        hist_start = hist_end - pd.Timedelta(hours=lookback_hours - 1)
        timestamps = pd.date_range(hist_start, hist_end, freq="h")

        if len(timestamps) != lookback_hours:
            continue
        if any(ts not in timestamp_to_pos for ts in timestamps):
            continue

        if delivery_date not in target_pivot.index:
            continue
        y_vec = target_pivot.loc[delivery_date].to_numpy(dtype=float)
        if y_vec.shape[0] != 24 or np.isnan(y_vec).all():
            continue

        future_cov = build_future_covariates_for_date(zone_panel, delivery_date, future_cols_base)
        if future_cov is None:
            continue

        positions = np.array([timestamp_to_pos[ts] for ts in timestamps], dtype=np.int64)
        samples.append(
            RawSample(
                delivery_date=delivery_date,
                forecast_date=forecast_date,
                history_positions=positions,
                future_covariates=future_cov,
                y=y_vec.astype(float),
            )
        )

    # Use stable future-column order even when some dates are missing.
    future_flat_cols = []
    for hour in range(1, 25):
        for col in future_cols_base:
            future_flat_cols.append(f"future_{col}_h{hour:02d}")

    return SampleCollection(samples=samples, future_covariate_columns=future_flat_cols)


# =============================================================================
# 5. LEAKAGE-SAFE PREPROCESSORS
# =============================================================================


class SequencePreprocessor:
    """Preprocess historical sequence features with training-only statistics."""

    def __init__(self, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> None:
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_scaler = StandardScaler()
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.onehot: Optional[OneHotEncoder] = None
        self.output_feature_names_: List[str] = []

    @staticmethod
    def _make_onehot() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, df: pd.DataFrame) -> "SequencePreprocessor":
        parts: List[str] = []
        if self.numeric_cols:
            x_num = df[self.numeric_cols].copy()
            x_num_imp = self.numeric_imputer.fit_transform(x_num)
            self.numeric_scaler.fit(x_num_imp)
            parts.extend(self.numeric_cols)

        if self.categorical_cols:
            x_cat = df[self.categorical_cols].copy().astype("object")
            x_cat_imp = self.categorical_imputer.fit_transform(x_cat)
            self.onehot = self._make_onehot()
            self.onehot.fit(x_cat_imp)
            try:
                cat_names = self.onehot.get_feature_names_out(self.categorical_cols).tolist()
            except Exception:
                cat_names = [f"cat_{i}" for i in range(self.onehot.transform(x_cat_imp[:1]).shape[1])]
            parts.extend(cat_names)

        self.output_feature_names_ = parts
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arrays: List[np.ndarray] = []
        if self.numeric_cols:
            x_num = df[self.numeric_cols].copy()
            x_num_imp = self.numeric_imputer.transform(x_num)
            x_num_scaled = self.numeric_scaler.transform(x_num_imp)
            arrays.append(np.asarray(x_num_scaled, dtype=np.float32))

        if self.categorical_cols:
            if self.onehot is None:
                raise RuntimeError("SequencePreprocessor has not been fitted.")
            x_cat = df[self.categorical_cols].copy().astype("object")
            x_cat_imp = self.categorical_imputer.transform(x_cat)
            x_cat_ohe = self.onehot.transform(x_cat_imp)
            arrays.append(np.asarray(x_cat_ohe, dtype=np.float32))

        if not arrays:
            raise ValueError("No sequence features available after preprocessing.")
        return np.concatenate(arrays, axis=1)


class FutureCovariatePreprocessor:
    """Median-impute and scale flattened future-known covariates."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "FutureCovariatePreprocessor":
        x = df.reindex(columns=self.columns)
        x_imp = self.imputer.fit_transform(x)
        self.scaler.fit(x_imp)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df.reindex(columns=self.columns)
        x_imp = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imp)
        return np.asarray(x_scaled, dtype=np.float32)


class TargetScaler:
    """Scale multi-output 24h targets using one scaler fitted to all hourly values."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.scaler.fit(y.reshape(-1, 1))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        shape = y.shape
        return self.scaler.transform(y.reshape(-1, 1)).reshape(shape).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        shape = y_scaled.shape
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(shape)


def samples_to_future_frame(samples: Sequence[RawSample], columns: Sequence[str]) -> pd.DataFrame:
    rows = [sample.future_covariates for sample in samples]
    return pd.DataFrame(rows).reindex(columns=list(columns))


def samples_to_y(samples: Sequence[RawSample]) -> np.ndarray:
    return np.vstack([sample.y for sample in samples]).astype(float)


def transform_sample_collection(
    samples: Sequence[RawSample],
    hist_matrix: np.ndarray,
    future_preprocessor: FutureCovariatePreprocessor,
    future_columns: Sequence[str],
    target_scaler: Optional[TargetScaler] = None,
    include_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x_seq = np.stack([hist_matrix[s.history_positions, :] for s in samples]).astype(np.float32)
    future_df = samples_to_future_frame(samples, future_columns)
    x_future = future_preprocessor.transform(future_df).astype(np.float32)

    y_out: Optional[np.ndarray] = None
    if include_y:
        y_raw = samples_to_y(samples)
        y_out = target_scaler.transform(y_raw) if target_scaler is not None else y_raw.astype(np.float32)
    return x_seq, x_future, y_out


# =============================================================================
# 6. PYTORCH DATASET AND MODEL
# =============================================================================


class DailySequenceDataset(Dataset):
    """Dataset for LSTM day-ahead direct multi-output forecasting."""

    def __init__(self, x_seq: np.ndarray, x_future: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.x_future = torch.as_tensor(x_future, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_seq.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.y is None:
            return self.x_seq[idx], self.x_future[idx], torch.empty(0)
        return self.x_seq[idx], self.x_future[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    """Encoder LSTM: historical sequence -> final hidden state -> dense 24h head."""

    def __init__(
        self,
        input_size: int,
        future_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int = 24,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        combined_size = hidden_size + future_size
        mid_size = max(hidden_size, 32)
        self.head = nn.Sequential(
            nn.Linear(combined_size, mid_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, output_size),
        )

    def forward(self, x_seq: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        combined = torch.cat([h_last, x_future], dim=1)
        return self.head(combined)


@dataclass
class TrainingResult:
    model: LSTMForecaster
    history: pd.DataFrame
    epochs_trained: int
    best_validation_loss: float
    fit_seconds: float


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    abs_errors: List[float] = []
    with torch.no_grad():
        for x_seq, x_future, y in loader:
            x_seq = x_seq.to(device)
            x_future = x_future.to(device)
            y = y.to(device)
            pred = model(x_seq, x_future)
            loss = loss_fn(pred, y)
            losses.append(float(loss.item()))
            abs_errors.append(float(torch.mean(torch.abs(pred - y)).item()))
    return float(np.mean(losses)), float(np.mean(abs_errors))


def train_lstm_model(
    x_train_seq: np.ndarray,
    x_train_future: np.ndarray,
    y_train: np.ndarray,
    x_val_seq: np.ndarray,
    x_val_future: np.ndarray,
    y_val: np.ndarray,
    strategy: LSTMStrategy,
    cfg: Config,
    device: torch.device,
) -> TrainingResult:
    """Train one LSTM model with early stopping on inner validation loss."""
    start = time.time()
    train_dataset = DailySequenceDataset(x_train_seq, x_train_future, y_train)
    val_dataset = DailySequenceDataset(x_val_seq, x_val_future, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=strategy.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=strategy.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.NUM_WORKERS,
    )

    model = LSTMForecaster(
        input_size=x_train_seq.shape[2],
        future_size=x_train_future.shape[1],
        hidden_size=strategy.hidden_size,
        num_layers=strategy.num_layers,
        dropout=strategy.dropout,
        output_size=24,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=strategy.learning_rate,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    rows: List[Dict[str, Any]] = []

    for epoch in range(1, strategy.max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for x_seq, x_future, y in train_loader:
            x_seq = x_seq.to(device)
            x_future = x_future.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_seq, x_future)
            loss = loss_fn(pred, y)
            loss.backward()
            if cfg.GRADIENT_CLIP_NORM is not None and cfg.GRADIENT_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.inf
        val_loss, val_mae_scaled = evaluate_loss(model, val_loader, device, loss_fn)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_mae_scaled": val_mae_scaled,
            }
        )

        if val_loss < best_val_loss - cfg.MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    fit_seconds = time.time() - start
    history = pd.DataFrame(rows)
    return TrainingResult(
        model=model,
        history=history,
        epochs_trained=int(best_epoch if best_epoch > 0 else len(rows)),
        best_validation_loss=float(best_val_loss),
        fit_seconds=float(fit_seconds),
    )


def predict_lstm_model(
    model: LSTMForecaster,
    x_seq: np.ndarray,
    x_future: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    start = time.time()
    dataset = DailySequenceDataset(x_seq, x_future, y=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    preds: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_seq_batch, x_future_batch, _ in loader:
            x_seq_batch = x_seq_batch.to(device)
            x_future_batch = x_future_batch.to(device)
            pred = model(x_seq_batch, x_future_batch)
            preds.append(pred.detach().cpu().numpy())
    pred_scaled = np.vstack(preds) if preds else np.empty((0, 24), dtype=np.float32)
    return pred_scaled, float(time.time() - start)


# =============================================================================
# 7. STRATEGIES AND MODE SELECTION
# =============================================================================

def make_default_strategies(cfg: Config) -> Dict[str, LSTMStrategy]:
    """Create a compact 24-month rolling LSTM specification.

    This script is a robustness/extension run, not the original wide
    hyperparameter search. It keeps the best architectures found in the
    expanding-window validation and changes only the training-window design
    from expanding to rolling 24 months.

    The goal is to test whether excluding the 2022 crisis from the 2025
    training window improves the final test performance, while avoiding another
    large hyperparameter search.
    """
    strategies = [
        LSTMStrategy(
            strategy_id="lstm_rolling24_168h_h64_l2_lr5em04_b32",
            window_type="rolling",
            lookback_hours=168,
            window_months=24,
            hidden_size=64,
            num_layers=2,
            dropout=cfg.DROPOUT_TWO_LAYERS,
            learning_rate=5e-4,
            batch_size=32,
            max_epochs=cfg.MAX_EPOCHS,
        ),
        LSTMStrategy(
            strategy_id="lstm_rolling24_336h_h64_l1_lr5em04_b32",
            window_type="rolling",
            lookback_hours=336,
            window_months=24,
            hidden_size=64,
            num_layers=1,
            dropout=cfg.DROPOUT_ONE_LAYER,
            learning_rate=5e-4,
            batch_size=32,
            max_epochs=cfg.MAX_EPOCHS,
        ),
        LSTMStrategy(
            strategy_id="lstm_rolling24_336h_h64_l2_lr5em04_b32",
            window_type="rolling",
            lookback_hours=336,
            window_months=24,
            hidden_size=64,
            num_layers=2,
            dropout=cfg.DROPOUT_TWO_LAYERS,
            learning_rate=5e-4,
            batch_size=32,
            max_epochs=cfg.MAX_EPOCHS,
        ),
    ]
    return {s.strategy_id: s for s in strategies}


def make_smoke_strategy() -> LSTMStrategy:
    return LSTMStrategy(
        strategy_id="lstm_rolling24_smoke_168h",
        window_type="rolling",
        lookback_hours=168,
        window_months=24,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=16,
        max_epochs=2,
    )


def apply_fast_overrides(strategy: LSTMStrategy, cfg: Config) -> LSTMStrategy:
    """Reduce epochs for fast validation but preserve the selected architecture."""
    return LSTMStrategy(
        strategy_id=strategy.strategy_id,
        window_type=strategy.window_type,
        lookback_hours=strategy.lookback_hours,
        window_months=strategy.window_months,
        hidden_size=strategy.hidden_size,
        num_layers=strategy.num_layers,
        dropout=strategy.dropout,
        learning_rate=strategy.learning_rate,
        batch_size=strategy.batch_size,
        max_epochs=cfg.FAST_MAX_EPOCHS,
    )

def resolve_modes(cfg: Config) -> List[str]:
    if cfg.RUN_SMOKE_TEST:
        return ["smoke"]
    modes: List[str] = []
    if cfg.RUN_FAST_VALIDATION:
        modes.append("fast_validation")
    if cfg.RUN_FULL_VALIDATION:
        modes.append("full_validation")
    if cfg.RUN_FINAL_TEST:
        modes.append("final_test")
    if not modes:
        raise ValueError("No execution mode is enabled.")
    return modes


def get_mode_targets_zones(cfg: Config, mode: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if mode == "smoke":
        return cfg.SMOKE_TARGETS, cfg.SMOKE_ZONES
    if mode == "fast_validation":
        return cfg.FAST_TARGETS, cfg.FAST_ZONES
    return cfg.TARGETS, cfg.ZONES


def get_mode_split(mode: str) -> str:
    if mode in {"smoke", "fast_validation", "full_validation"}:
        return "validation"
    if mode == "final_test":
        return "test"
    raise ValueError(f"Unsupported mode: {mode}")


def get_mode_strategies(
    mode: str,
    cfg: Config,
    strategy_dict: Dict[str, LSTMStrategy],
    selected_by_target: Optional[Dict[str, str]] = None,
) -> List[LSTMStrategy]:
    if mode == "smoke":
        return [make_smoke_strategy()]
    if mode == "fast_validation":
        selected_ids = [
            "lstm_rolling24_168h_h64_l2_lr5em04_b32",
            "lstm_rolling24_336h_h64_l1_lr5em04_b32",
            "lstm_rolling24_336h_h64_l2_lr5em04_b32",
        ]
        return [apply_fast_overrides(strategy_dict[sid], cfg) for sid in selected_ids]
    if mode == "full_validation":
        return list(strategy_dict.values())
    if mode == "final_test":
        ids = set((selected_by_target or cfg.SELECTED_STRATEGY_BY_TARGET).values())
        ids.update(cfg.SELECTED_STRATEGY_BY_TARGET_ZONE.values())
        return [strategy_dict[sid] for sid in sorted(ids) if sid in strategy_dict]
    raise ValueError(f"Unsupported mode: {mode}")


# =============================================================================
# 8. RECALIBRATION BLOCKS AND CHECKPOINTS
# =============================================================================


def limit_eval_dates_for_mode(eval_tz: pd.DataFrame, mode: str, cfg: Config) -> pd.DataFrame:
    out = eval_tz.copy().sort_values(["forecast_date", "delivery_date", "hour"])
    if mode == "smoke":
        first_dates = out["forecast_date"].drop_duplicates().sort_values().head(cfg.SMOKE_MAX_FORECAST_DATES)
        out = out[out["forecast_date"].isin(first_dates)].copy()
    elif mode == "fast_validation":
        if cfg.FAST_MAX_VALIDATION_MONTHS is not None:
            months = out["forecast_date"].dt.to_period("M").drop_duplicates().sort_values().head(cfg.FAST_MAX_VALIDATION_MONTHS)
            out = out[out["forecast_date"].dt.to_period("M").isin(months)].copy()
        if cfg.FAST_MAX_FORECAST_DATES_PER_SERIES is not None:
            dates = out["forecast_date"].drop_duplicates().sort_values().head(cfg.FAST_MAX_FORECAST_DATES_PER_SERIES)
            out = out[out["forecast_date"].isin(dates)].copy()
    return out


def make_recalibration_blocks(eval_tz: pd.DataFrame, cfg: Config) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
    """Create recalibration blocks according to the configured frequency.

    If RECALIBRATION_FREQUENCY = "none", the model is trained once at the
    beginning of the evaluation period and then used to predict the whole
    validation/test block.

    If RECALIBRATION_FREQUENCY = "monthly", the model is re-trained once
    per forecast month.
    """
    blocks: List[Tuple[pd.Timestamp, pd.DataFrame]] = []
    if eval_tz.empty:
        return blocks

    temp = eval_tz.copy().sort_values(["forecast_date", "delivery_date", "hour"])
    freq = str(cfg.RECALIBRATION_FREQUENCY).lower()

    if freq in {"none", "no", "no_recalibration", "fixed"}:
        recalibration_date = pd.Timestamp(temp["forecast_date"].min()).normalize()
        blocks.append((recalibration_date, temp))
        return blocks

    if freq == "monthly":
        temp["forecast_month"] = temp["forecast_date"].dt.to_period("M")
        for _, block in temp.groupby("forecast_month", sort=True):
            block = block.drop(columns=["forecast_month"]).sort_values(["forecast_date", "hour"])
            recalibration_date = pd.Timestamp(block["forecast_date"].min()).normalize()
            blocks.append((recalibration_date, block))
        return blocks

    raise ValueError(f"Unsupported RECALIBRATION_FREQUENCY: {cfg.RECALIBRATION_FREQUENCY}")

def get_train_window(recalibration_date: pd.Timestamp, strategy: LSTMStrategy, cfg: Config) -> Tuple[pd.Timestamp, pd.Timestamp]:
    train_end = pd.Timestamp(recalibration_date).normalize()
    if strategy.window_type == "expanding":
        train_start = pd.Timestamp(cfg.INITIAL_TRAIN_START_DATE).normalize()
    elif strategy.window_type == "rolling":
        if strategy.window_months is None:
            raise ValueError(f"Rolling strategy {strategy.strategy_id} requires window_months.")
        train_start = train_end - pd.DateOffset(months=int(strategy.window_months)) + pd.Timedelta(days=1)
        train_start = pd.Timestamp(train_start).normalize()
    else:
        raise ValueError(f"Unknown window_type: {strategy.window_type}")
    return train_start, train_end


def checkpoint_path(cfg: Config, split: str, strategy_id: str, target: str, zone: str, recalibration_date: pd.Timestamp) -> Path:
    fname = "__".join(
        [
            safe_filename(split),
            safe_filename(strategy_id),
            safe_filename(target),
            safe_filename(zone),
            safe_filename(pd.Timestamp(recalibration_date).strftime("%Y-%m-%d")),
        ]
    ) + ".parquet"
    return cfg.EXPERIMENT_DIR / "checkpoints" / fname


def model_artifact_path(cfg: Config, split: str, strategy_id: str, target: str, zone: str, recalibration_date: pd.Timestamp) -> Path:
    fname = "__".join(
        [
            safe_filename(split),
            safe_filename(strategy_id),
            safe_filename(target),
            safe_filename(zone),
            safe_filename(pd.Timestamp(recalibration_date).strftime("%Y-%m-%d")),
        ]
    ) + ".pt"
    return cfg.EXPERIMENT_DIR / "model_artifacts" / fname


def training_curve_path(cfg: Config, split: str, strategy_id: str, target: str, zone: str, recalibration_date: pd.Timestamp) -> Path:
    fname = "__".join(
        [
            safe_filename(split),
            safe_filename(strategy_id),
            safe_filename(target),
            safe_filename(zone),
            safe_filename(pd.Timestamp(recalibration_date).strftime("%Y-%m-%d")),
        ]
    ) + ".csv"
    return cfg.EXPERIMENT_DIR / "training_curves" / fname


# =============================================================================
# 9. CORE FORECASTING ROUTINE
# =============================================================================


def select_samples_by_date(
    samples: Sequence[RawSample],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    exact_dates: Optional[Sequence[pd.Timestamp]] = None,
) -> List[RawSample]:
    if exact_dates is not None:
        date_set = {pd.Timestamp(d).normalize() for d in exact_dates}
        return [s for s in samples if s.delivery_date.normalize() in date_set]

    out: List[RawSample] = []
    for s in samples:
        d = s.delivery_date.normalize()
        if start_date is not None and d < pd.Timestamp(start_date).normalize():
            continue
        if end_date is not None and d > pd.Timestamp(end_date).normalize():
            continue
        out.append(s)
    return out


def split_inner_validation(
    train_samples: Sequence[RawSample],
    inner_validation_days: int,
) -> Tuple[List[RawSample], List[RawSample]]:
    samples = sorted(train_samples, key=lambda s: s.delivery_date)
    if not samples:
        return [], []
    dates = sorted({s.delivery_date.normalize() for s in samples})
    val_dates = set(dates[-inner_validation_days:])
    inner_train = [s for s in samples if s.delivery_date.normalize() not in val_dates]
    inner_val = [s for s in samples if s.delivery_date.normalize() in val_dates]
    return inner_train, inner_val


def make_prediction_long_from_samples(
    pred_samples: Sequence[RawSample],
    pred_values: np.ndarray,
    eval_block: pd.DataFrame,
    target: str,
    zone: str,
    split: str,
    model_name: str,
    strategy: LSTMStrategy,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    recalibration_date: pd.Timestamp,
    epochs_trained: int,
    best_validation_loss: float,
    fit_seconds: float,
    predict_seconds: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(pred_samples):
        for hour in range(1, 25):
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "zone": zone,
                    "split": split,
                    "forecast_date": sample.forecast_date,
                    "delivery_date": sample.delivery_date,
                    "hour": hour,
                    "horizon": hour,
                    "y_pred": float(pred_values[sample_idx, hour - 1]),
                    "strategy_id": strategy.strategy_id,
                    "lookback_hours": strategy.lookback_hours,
                    "window_type": strategy.window_type,
                    "train_start_date": train_start,
                    "train_end_date": train_end,
                    "recalibration_date": recalibration_date,
                    "hidden_size": strategy.hidden_size,
                    "num_layers": strategy.num_layers,
                    "dropout": strategy.dropout,
                    "learning_rate": strategy.learning_rate,
                    "batch_size": strategy.batch_size,
                    "epochs_trained": epochs_trained,
                    "best_validation_loss": best_validation_loss,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )

    pred = pd.DataFrame(rows)
    if pred.empty:
        return pred

    join_cols = ["target", "zone", "split", "forecast_date", "delivery_date", "hour", "horizon"]
    eval_join = eval_block[join_cols + ["delivery_datetime_model", "y_true"]].copy()
    eval_join["delivery_date"] = pd.to_datetime(eval_join["delivery_date"])
    eval_join["forecast_date"] = pd.to_datetime(eval_join["forecast_date"])

    pred["delivery_date"] = pd.to_datetime(pred["delivery_date"])
    pred["forecast_date"] = pd.to_datetime(pred["forecast_date"])

    merged = pred.merge(eval_join, on=join_cols, how="left", validate="one_to_one")
    if merged["delivery_datetime_model"].isna().any():
        missing = int(merged["delivery_datetime_model"].isna().sum())
        raise ValueError(f"Prediction/eval_index alignment failed: {missing} rows did not match eval_index.")

    ordered_cols = [
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
        "strategy_id",
        "lookback_hours",
        "window_type",
        "train_start_date",
        "train_end_date",
        "recalibration_date",
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "batch_size",
        "epochs_trained",
        "best_validation_loss",
        "fit_seconds",
        "predict_seconds",
    ]
    return merged[ordered_cols].sort_values(["target", "zone", "delivery_datetime_model"])


def run_one_recalibration_block(
    panel: pd.DataFrame,
    hist_features: pd.DataFrame,
    hist_numeric_cols: List[str],
    hist_categorical_cols: List[str],
    all_samples: SampleCollection,
    eval_block: pd.DataFrame,
    target: str,
    zone: str,
    split: str,
    strategy: LSTMStrategy,
    recalibration_date: pd.Timestamp,
    model_name: str,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit/predict one target-zone-strategy-recalibration block."""
    total_start = time.time()
    log_row: Dict[str, Any] = {
        "split": split,
        "target": target,
        "zone": zone,
        "strategy_id": strategy.strategy_id,
        "recalibration_date": recalibration_date,
        "lookback_hours": strategy.lookback_hours,
        "hidden_size": strategy.hidden_size,
        "num_layers": strategy.num_layers,
        "dropout": strategy.dropout,
        "learning_rate": strategy.learning_rate,
        "batch_size": strategy.batch_size,
        "device_used": str(device),
        "status": "started",
        "error_message": "",
    }

    ckpt = checkpoint_path(cfg, split, strategy.strategy_id, target, zone, recalibration_date)
    if (
        cfg.SAVE_CHECKPOINTS
        and cfg.RESUME_FROM_CHECKPOINTS
        and ckpt.exists()
        and not cfg.OVERWRITE_EXISTING_CHECKPOINTS
    ):
        pred_ckpt = pd.read_parquet(ckpt)
        log_row.update(
            {
                "status": "loaded_checkpoint",
                "n_predictions": len(pred_ckpt),
                "total_seconds": 0.0,
            }
        )
        return pred_ckpt, log_row

    try:
        train_start, train_end = get_train_window(recalibration_date, strategy, cfg)
        log_row["train_start_date"] = train_start
        log_row["train_end_date"] = train_end

        train_samples_all = select_samples_by_date(
            all_samples.samples,
            start_date=train_start,
            end_date=train_end,
        )
        inner_train_samples, inner_val_samples = split_inner_validation(
            train_samples_all,
            cfg.INNER_VALIDATION_DAYS,
        )

        log_row["n_train_samples"] = len(inner_train_samples)
        log_row["n_inner_validation_samples"] = len(inner_val_samples)

        if len(train_samples_all) < cfg.MIN_TRAIN_SAMPLES:
            raise ValueError(f"Not enough total training samples: {len(train_samples_all)}")
        if len(inner_train_samples) < cfg.MIN_INNER_TRAIN_SAMPLES:
            raise ValueError(f"Not enough inner training samples: {len(inner_train_samples)}")
        if len(inner_val_samples) < cfg.MIN_INNER_VALIDATION_SAMPLES:
            raise ValueError(f"Not enough inner validation samples: {len(inner_val_samples)}")

        pred_delivery_dates = eval_block["delivery_date"].drop_duplicates().sort_values().tolist()
        pred_samples = select_samples_by_date(
            all_samples.samples,
            exact_dates=pred_delivery_dates,
        )
        if len(pred_samples) == 0:
            raise ValueError("No prediction samples could be constructed for this block.")

        # Fit historical preprocessor only on rows available up to train_end.
        hist_fit_end = train_end + pd.Timedelta(hours=23)
        hist_fit_start = train_start - pd.Timedelta(hours=strategy.lookback_hours)
        hist_fit_df = hist_features.loc[(hist_features.index >= hist_fit_start) & (hist_features.index <= hist_fit_end)]
        if hist_fit_df.empty:
            raise ValueError("No historical rows available to fit sequence preprocessor.")

        seq_preprocessor = SequencePreprocessor(hist_numeric_cols, hist_categorical_cols).fit(hist_fit_df)
        transformed_hist = seq_preprocessor.transform(hist_features)
        log_row["n_features"] = transformed_hist.shape[1]

        future_preprocessor = FutureCovariatePreprocessor(all_samples.future_covariate_columns)
        future_preprocessor.fit(samples_to_future_frame(inner_train_samples, all_samples.future_covariate_columns))

        target_scaler = TargetScaler().fit(samples_to_y(inner_train_samples))

        x_train_seq, x_train_future, y_train = transform_sample_collection(
            inner_train_samples,
            transformed_hist,
            future_preprocessor,
            all_samples.future_covariate_columns,
            target_scaler=target_scaler,
            include_y=True,
        )
        x_val_seq, x_val_future, y_val = transform_sample_collection(
            inner_val_samples,
            transformed_hist,
            future_preprocessor,
            all_samples.future_covariate_columns,
            target_scaler=target_scaler,
            include_y=True,
        )

        training_result = train_lstm_model(
            x_train_seq,
            x_train_future,
            y_train,
            x_val_seq,
            x_val_future,
            y_val,
            strategy,
            cfg,
            device,
        )

        curve = training_result.history.copy()
        curve.insert(0, "split", split)
        curve.insert(1, "target", target)
        curve.insert(2, "zone", zone)
        curve.insert(3, "strategy_id", strategy.strategy_id)
        curve.insert(4, "recalibration_date", recalibration_date)
        curve_path = training_curve_path(cfg, split, strategy.strategy_id, target, zone, recalibration_date)
        ensure_dir(curve_path.parent)
        curve.to_csv(curve_path, index=False)

        if cfg.SAVE_FITTED_MODELS:
            artifact = model_artifact_path(cfg, split, strategy.strategy_id, target, zone, recalibration_date)
            ensure_dir(artifact.parent)
            torch.save(
                {
                    "model_state_dict": training_result.model.state_dict(),
                    "strategy": asdict(strategy),
                    "input_size": x_train_seq.shape[2],
                    "future_size": x_train_future.shape[1],
                    "sequence_feature_names": seq_preprocessor.output_feature_names_,
                    "future_columns": all_samples.future_covariate_columns,
                },
                artifact,
            )

        x_pred_seq, x_pred_future, _ = transform_sample_collection(
            pred_samples,
            transformed_hist,
            future_preprocessor,
            all_samples.future_covariate_columns,
            target_scaler=None,
            include_y=False,
        )
        pred_scaled, predict_seconds = predict_lstm_model(
            training_result.model,
            x_pred_seq,
            x_pred_future,
            strategy.batch_size,
            device,
        )
        pred_original = target_scaler.inverse_transform(pred_scaled)

        pred_long = make_prediction_long_from_samples(
            pred_samples=pred_samples,
            pred_values=pred_original,
            eval_block=eval_block,
            target=target,
            zone=zone,
            split=split,
            model_name=model_name,
            strategy=strategy,
            train_start=train_start,
            train_end=train_end,
            recalibration_date=recalibration_date,
            epochs_trained=training_result.epochs_trained,
            best_validation_loss=training_result.best_validation_loss,
            fit_seconds=training_result.fit_seconds,
            predict_seconds=predict_seconds,
        )

        if cfg.SAVE_CHECKPOINTS:
            ensure_dir(ckpt.parent)
            pred_long.to_parquet(ckpt, index=False)

        log_row.update(
            {
                "epochs_trained": training_result.epochs_trained,
                "best_validation_loss": training_result.best_validation_loss,
                "fit_seconds": training_result.fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": time.time() - total_start,
                "n_predictions": len(pred_long),
                "status": "ok",
            }
        )

        del transformed_hist, x_train_seq, x_train_future, y_train, x_val_seq, x_val_future, y_val
        gc.collect()
        return pred_long, log_row

    except Exception as exc:
        log_row.update(
            {
                "status": "error",
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "total_seconds": time.time() - total_start,
            }
        )
        logger.error(
            "Failed block | split=%s strategy=%s target=%s zone=%s recalibration=%s | %s",
            split,
            strategy.strategy_id,
            target,
            zone,
            recalibration_date,
            exc,
        )
        return pd.DataFrame(), log_row


def run_strategy_for_target_zone(
    panel: pd.DataFrame,
    eval_index: pd.DataFrame,
    target: str,
    zone: str,
    split: str,
    mode: str,
    strategy: LSTMStrategy,
    model_name: str,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all monthly recalibration blocks for one strategy-target-zone."""
    logger.info(
        "Running | mode=%s split=%s strategy=%s target=%s zone=%s",
        mode,
        split,
        strategy.strategy_id,
        target,
        zone,
    )

    eval_tz = eval_index[
        (eval_index["target"] == target)
        & (eval_index["zone"] == zone)
        & (eval_index["split"] == split)
    ].copy()
    eval_tz = limit_eval_dates_for_mode(eval_tz, mode, cfg)

    if eval_tz.empty:
        logger.warning("No eval_index rows for split=%s target=%s zone=%s", split, target, zone)
        return pd.DataFrame(), pd.DataFrame()

    hist_features, hist_numeric_cols, hist_categorical_cols = build_historical_feature_frame(panel, zone, cfg)

    # Build samples for all dates needed for training and prediction under this strategy.
    min_eval_date = pd.Timestamp(eval_tz["delivery_date"].min()).normalize()
    max_eval_date = pd.Timestamp(eval_tz["delivery_date"].max()).normalize()

    # Include all potentially useful training dates from the initial start until the max eval date.
    start_date = pd.Timestamp(cfg.INITIAL_TRAIN_START_DATE).normalize()
    all_dates = pd.date_range(start_date, max_eval_date, freq="D")

    all_samples = build_daily_samples_for_target_zone(
        panel=panel,
        hist_features=hist_features,
        target=target,
        zone=zone,
        lookback_hours=strategy.lookback_hours,
        delivery_dates=all_dates,
        cfg=cfg,
    )

    if not all_samples.samples:
        logger.warning("No daily samples built for target=%s zone=%s strategy=%s", target, zone, strategy.strategy_id)
        return pd.DataFrame(), pd.DataFrame()

    blocks = make_recalibration_blocks(eval_tz, cfg)
    pred_parts: List[pd.DataFrame] = []
    log_rows: List[Dict[str, Any]] = []

    for recalibration_date, block in blocks:
        pred_block, log_row = run_one_recalibration_block(
            panel=panel,
            hist_features=hist_features,
            hist_numeric_cols=hist_numeric_cols,
            hist_categorical_cols=hist_categorical_cols,
            all_samples=all_samples,
            eval_block=block,
            target=target,
            zone=zone,
            split=split,
            strategy=strategy,
            recalibration_date=recalibration_date,
            model_name=model_name,
            cfg=cfg,
            device=device,
            logger=logger,
        )
        if not pred_block.empty:
            pred_parts.append(pred_block)
        log_rows.append(log_row)

    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    logs_df = pd.DataFrame(log_rows)

    if split == "validation" and not pred_df.empty:
        candidate_dir = cfg.EXPERIMENT_DIR / "validation_candidate_predictions"
        ensure_dir(candidate_dir)
        candidate_file = candidate_dir / f"pred_{safe_filename(strategy.strategy_id)}__{safe_filename(target)}__{safe_filename(zone)}.parquet"
        pred_df.to_parquet(candidate_file, index=False)

    return pred_df, logs_df


# =============================================================================
# 10. QUICK METRICS AND STRATEGY SELECTION
# =============================================================================


def compute_quick_metrics(pred: pd.DataFrame, naive: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    df = pred.copy()
    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
    df["squared_error"] = (df["y_true"] - df["y_pred"]) ** 2

    group_cols = ["model", "strategy_id", "target", "zone", "split"]
    metrics = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("abs_error", "size"),
            MAE=("abs_error", "mean"),
            RMSE=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
        )
        .reset_index()
    )

    if naive is None or naive.empty:
        metrics["rMAE"] = np.nan
        return metrics

    naive_df = naive.copy()
    required = ["target", "zone", "split", "delivery_datetime_model", "y_true", "y_pred"]
    if any(col not in naive_df.columns for col in required):
        metrics["rMAE"] = np.nan
        return metrics

    naive_df["delivery_datetime_model"] = normalize_datetime_column(naive_df["delivery_datetime_model"])
    naive_df = naive_df[naive_df["model"].astype(str) == "naive_week_before"].copy()
    naive_df["naive_abs_error"] = (naive_df["y_true"] - naive_df["y_pred"]).abs()

    key_cols = ["target", "zone", "split", "delivery_datetime_model"]
    joined = df.merge(
        naive_df[key_cols + ["naive_abs_error"]],
        on=key_cols,
        how="left",
    )
    rmae = (
        joined.groupby(group_cols, dropna=False)
        .agg(naive_MAE=("naive_abs_error", "mean"))
        .reset_index()
    )
    metrics = metrics.merge(rmae, on=group_cols, how="left")
    metrics["rMAE"] = metrics["MAE"] / metrics["naive_MAE"]
    return metrics


def select_best_strategy_by_target(metrics: pd.DataFrame, cfg: Config, logger: logging.Logger) -> Dict[str, str]:
    if metrics.empty:
        logger.warning("No validation metrics available for strategy selection. Using configured selected strategies.")
        return cfg.SELECTED_STRATEGY_BY_TARGET.copy()

    val = metrics[metrics["split"] == "validation"].copy()
    if val.empty:
        return cfg.SELECTED_STRATEGY_BY_TARGET.copy()

    score_col = "rMAE" if val["rMAE"].notna().any() else "MAE"
    target_scores = (
        val.groupby(["target", "strategy_id"], dropna=False)
        .agg(score=(score_col, "mean"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), n=("n", "sum"))
        .reset_index()
        .sort_values(["target", "score", "MAE"])
    )

    selected: Dict[str, str] = {}
    for target, group in target_scores.groupby("target", sort=True):
        selected[str(target)] = str(group.iloc[0]["strategy_id"])

    selected_dir = cfg.EXPERIMENT_DIR / "selected_strategy"
    ensure_dir(selected_dir)
    target_scores.to_csv(selected_dir / "lstm_validation_strategy_results.csv", index=False)
    pd.DataFrame(
        [{"target": k, "selected_strategy_id": v, "selection_metric": score_col} for k, v in selected.items()]
    ).to_csv(selected_dir / "lstm_selected_strategy_by_target.csv", index=False)

    logger.info("Selected strategies by target using %s: %s", score_col, selected)
    return selected


def select_best_strategy_by_target_zone(metrics: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    if metrics.empty:
        return {}
    val = metrics[metrics["split"] == "validation"].copy()
    if val.empty:
        return {}
    score_col = "rMAE" if val["rMAE"].notna().any() else "MAE"
    scores = (
        val.groupby(["target", "zone", "strategy_id"], dropna=False)
        .agg(score=(score_col, "mean"), MAE=("MAE", "mean"), n=("n", "sum"))
        .reset_index()
        .sort_values(["target", "zone", "score", "MAE"])
    )
    selected: Dict[str, str] = {}
    for (target, zone), group in scores.groupby(["target", "zone"], sort=True):
        selected[f"{target}__{zone}"] = str(group.iloc[0]["strategy_id"])

    selected_dir = cfg.EXPERIMENT_DIR / "selected_strategy"
    ensure_dir(selected_dir)
    scores.to_csv(selected_dir / "lstm_validation_strategy_results_by_target_zone.csv", index=False)
    pd.DataFrame(
        [
            {"target_zone": k, "selected_strategy_id": v, "selection_metric": score_col}
            for k, v in selected.items()
        ]
    ).to_csv(selected_dir / "lstm_selected_strategy_by_target_zone.csv", index=False)
    return selected


# =============================================================================
# 11. QUALITY CHECKS
# =============================================================================


def validate_prediction_output(pred: pd.DataFrame, cfg: Config) -> None:
    if pred.empty:
        raise ValueError("Prediction output is empty.")

    missing_cols = [col for col in cfg.REQUIRED_PREDICTION_COLUMNS if col not in pred.columns]
    if missing_cols:
        raise ValueError(f"Prediction output is missing required columns: {missing_cols}")

    if pred["y_pred"].isna().any():
        n_missing = int(pred["y_pred"].isna().sum())
        raise ValueError(f"Prediction output contains {n_missing} missing y_pred values.")

    duplicated = pred.duplicated(["model", "target", "zone", "delivery_datetime_model"]).sum()
    if duplicated > 0:
        raise ValueError(f"Prediction output contains {duplicated} duplicated model-target-zone-datetime rows.")

    bad_horizons = sorted(set(pred["horizon"].dropna().astype(int)) - set(range(1, 25)))
    if bad_horizons:
        raise ValueError(f"Unexpected horizons in prediction output: {bad_horizons}")

    # Every target-zone-delivery day should contain the complete 24-hour profile.
    counts = (
        pred.groupby(["model", "target", "zone", "delivery_date"])["hour"]
        .nunique()
        .reset_index(name="n_hours")
    )
    incomplete = counts[counts["n_hours"] != 24]
    if not incomplete.empty:
        raise ValueError(
            "Some model-target-zone-delivery_date combinations do not have 24 hourly predictions. "
            f"First rows: {incomplete.head().to_dict(orient='records')}"
        )


def print_quality_summary(pred: pd.DataFrame, logger: logging.Logger) -> None:
    logger.info("================ LSTM PREDICTION QUALITY SUMMARY ================")
    logger.info("Models: %s", sorted(pred["model"].astype(str).unique().tolist()))
    counts = (
        pred.groupby(["model", "split", "target", "zone"])
        .size()
        .reset_index(name="n_predictions")
        .sort_values(["model", "split", "target", "zone"])
    )
    logger.info("Prediction counts by model/split/target/zone:\n%s", counts.to_string(index=False))
    missing = pred.groupby(["model", "split", "target"])["y_pred"].apply(lambda x: int(x.isna().sum()))
    logger.info("Missing y_pred counts:\n%s", missing.to_string())
    horizon_counts = pred["horizon"].value_counts().sort_index()
    logger.info("Horizon distribution:\n%s", horizon_counts.to_string())


# =============================================================================
# 12. MAIN ORCHESTRATION
# =============================================================================


def should_run_strategy_for_final(
    target: str,
    zone: str,
    strategy_id: str,
    selected_by_target: Dict[str, str],
    cfg: Config,
) -> bool:
    tz_key = f"{target}__{zone}"
    if tz_key in cfg.SELECTED_STRATEGY_BY_TARGET_ZONE:
        return strategy_id == cfg.SELECTED_STRATEGY_BY_TARGET_ZONE[tz_key]
    return strategy_id == selected_by_target.get(target, cfg.SELECTED_STRATEGY_BY_TARGET.get(target))


def run_mode(
    mode: str,
    panel: pd.DataFrame,
    eval_index: pd.DataFrame,
    cfg: Config,
    strategy_dict: Dict[str, LSTMStrategy],
    selected_by_target: Dict[str, str],
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = get_mode_split(mode)
    targets, zones = get_mode_targets_zones(cfg, mode)
    strategies = get_mode_strategies(mode, cfg, strategy_dict, selected_by_target)

    pred_parts: List[pd.DataFrame] = []
    log_parts: List[pd.DataFrame] = []

    for strategy in strategies:
        for target in targets:
            for zone in zones:
                if mode == "final_test" and not should_run_strategy_for_final(target, zone, strategy.strategy_id, selected_by_target, cfg):
                    continue

                if mode in {"smoke", "final_test"}:
                    model_name = cfg.MODEL_NAME_SMOKE if mode == "smoke" else cfg.MODEL_NAME_FINAL
                else:
                    model_name = f"lstm__{strategy.strategy_id}"

                pred_df, logs_df = run_strategy_for_target_zone(
                    panel=panel,
                    eval_index=eval_index,
                    target=target,
                    zone=zone,
                    split=split,
                    mode=mode,
                    strategy=strategy,
                    model_name=model_name,
                    cfg=cfg,
                    device=device,
                    logger=logger,
                )
                if not pred_df.empty:
                    pred_parts.append(pred_df)
                if not logs_df.empty:
                    log_parts.append(logs_df)

                gc.collect()

    pred_out = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    logs_out = pd.concat(log_parts, ignore_index=True) if log_parts else pd.DataFrame()
    return pred_out, logs_out


def save_logs_and_metrics(
    all_logs: pd.DataFrame,
    all_predictions: pd.DataFrame,
    naive: Optional[pd.DataFrame],
    cfg: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    logs_dir = cfg.EXPERIMENT_DIR / "logs"
    ensure_dir(logs_dir)

    if not all_logs.empty:
        all_logs.to_csv(logs_dir / "lstm_training_logs.csv", index=False)
        logger.info("Saved training logs: %s", logs_dir / "lstm_training_logs.csv")

    metrics = compute_quick_metrics(all_predictions, naive)
    if not metrics.empty:
        metrics.to_csv(logs_dir / "lstm_quick_metrics.csv", index=False)
        logger.info("Saved quick metrics: %s", logs_dir / "lstm_quick_metrics.csv")

    return metrics


def main() -> None:
    cfg = Config()

    # Create folders before logging.
    for folder in [
        cfg.PREDICTIONS_DIR,
        cfg.EXPERIMENT_DIR,
        cfg.EXPERIMENT_DIR / "logs",
        cfg.EXPERIMENT_DIR / "checkpoints",
        cfg.EXPERIMENT_DIR / "training_curves",
        cfg.EXPERIMENT_DIR / "validation_candidate_predictions",
        cfg.EXPERIMENT_DIR / "selected_strategy",
        cfg.EXPERIMENT_DIR / "model_artifacts",
    ]:
        ensure_dir(folder)

    logger = setup_logging(cfg)
    set_global_seed(cfg.RANDOM_SEED)
    device = get_device(cfg)

    logger.info("Starting 10_lstm_rolling24_model.py")
    logger.info("Project root: %s", cfg.PROJECT_ROOT)
    logger.info("Device: %s", device)
    logger.info("Enabled modes: %s", resolve_modes(cfg))

    with open(cfg.EXPERIMENT_DIR / "logs" / "lstm_config.json", "w", encoding="utf-8") as f:
        json.dump(serialize_config(cfg), f, indent=2, default=str)

    panel = prepare_panel(load_panel(cfg), cfg)
    eval_index = prepare_eval_index(load_eval_index(cfg))
    naive = load_weekly_naive(cfg, logger)

    strategy_dict = make_default_strategies(cfg)
    modes = resolve_modes(cfg)

    all_predictions: List[pd.DataFrame] = []
    all_logs: List[pd.DataFrame] = []
    selected_by_target = cfg.SELECTED_STRATEGY_BY_TARGET.copy()

    # If validation is run before final in the same execution, use validation results
    # to update target-level strategy selection.
    for mode in modes:
        pred_mode, logs_mode = run_mode(
            mode=mode,
            panel=panel,
            eval_index=eval_index,
            cfg=cfg,
            strategy_dict=strategy_dict,
            selected_by_target=selected_by_target,
            device=device,
            logger=logger,
        )

        if not pred_mode.empty:
            all_predictions.append(pred_mode)
        if not logs_mode.empty:
            all_logs.append(logs_mode)

        if mode in {"fast_validation", "full_validation"} and not pred_mode.empty:
            current_predictions = pd.concat(all_predictions, ignore_index=True)
            current_logs = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
            current_metrics = save_logs_and_metrics(current_logs, current_predictions, naive, cfg, logger)
            selected_by_target = select_best_strategy_by_target(current_metrics, cfg, logger)
            select_best_strategy_by_target_zone(current_metrics, cfg)

    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_logs = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()

    if final_predictions.empty:
        logger.warning("No predictions were generated. Check logs for skipped/error tasks.")
        if not final_logs.empty:
            save_logs_and_metrics(final_logs, final_predictions, naive, cfg, logger)
        return

    # Coerce dates/timestamps to stable parquet-friendly dtypes.
    for col in ["forecast_date", "delivery_date", "train_start_date", "train_end_date", "recalibration_date"]:
        if col in final_predictions.columns:
            final_predictions[col] = pd.to_datetime(final_predictions[col], errors="coerce")
    if "delivery_datetime_model" in final_predictions.columns:
        final_predictions["delivery_datetime_model"] = normalize_datetime_column(final_predictions["delivery_datetime_model"])

    validate_prediction_output(final_predictions, cfg)
    print_quality_summary(final_predictions, logger)

    save_logs_and_metrics(final_logs, final_predictions, naive, cfg, logger)
    save_dataframe_outputs(final_predictions, cfg.FINAL_PREDICTION_PARQUET, cfg.FINAL_PREDICTION_RDS, logger)

    logger.info("Finished successfully.")
    logger.info("Main prediction output: %s", cfg.FINAL_PREDICTION_PARQUET)


if __name__ == "__main__":
    main()
