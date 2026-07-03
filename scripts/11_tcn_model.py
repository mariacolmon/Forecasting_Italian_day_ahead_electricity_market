#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_tcn_model.py

TCN

"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.nn.utils.parametrizations import weight_norm as apply_weight_norm
except ImportError:
    from torch.nn.utils import weight_norm as apply_weight_norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass
class TCNStrategy:
    strategy_id: str
    window_type: str
    lookback_hours: int
    window_months: Optional[int]
    num_channels: List[int]
    kernel_size: int
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    weight_norm: bool

    @property
    def dilations(self) -> List[int]:
        return [2 ** i for i in range(len(self.num_channels))]

    @property
    def receptive_field(self) -> int:
        # Two causal dilated convolutions per residual block.
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def as_log_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["dilations"] = self.dilations
        payload["receptive_field"] = self.receptive_field
        return payload


@dataclass
class Config:
    """Central configuration for the TCN experiment."""

    # Execution modes. If smoke test is True, it runs alone. If runtime
    # calibration is True, it runs alone unless ALLOW_COMBINED_RUNTIME_MODE=True.
    RUN_SMOKE_TEST: bool = False
    RUN_RUNTIME_CALIBRATION: bool = False
    RUN_FAST_VALIDATION: bool = False
    RUN_FULL_VALIDATION: bool = False
    RUN_FINAL_TEST: bool = True
    ALLOW_COMBINED_RUNTIME_MODE: bool = False

    # Deep learning defaults. CPU-safe and reproducible by default.
    MAX_EPOCHS: int = 50
    PATIENCE: int = 7
    MIN_DELTA: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    GRADIENT_CLIP_NORM: float = 1.0
    LOSS_FUNCTION: str = "mse"
    WEIGHT_NORM: bool = True
    RANDOM_SEED: int = 123
    NUM_WORKERS: int = 0
    ALLOW_CUDA: bool = False
    ALLOW_MPS: bool = False

    # Main modelling dimensions.
    TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    ZONES: Tuple[str, ...] = ("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")
    RECALIBRATION_FREQUENCY: str = "none"
    MODEL_NAME: str = "tcn"
    CANDIDATE_MODEL_PREFIX: str = "tcn__"

    # Validation/test years.
    VALIDATION_YEAR: int = 2024
    TEST_YEAR: int = 2025
    INNER_VALIDATION_DAYS: int = 30
    MIN_TRAIN_SAMPLES: int = 60
    MIN_INNER_TRAIN_SAMPLES: int = 30
    MIN_INNER_VALIDATION_SAMPLES: int = 7

    # Smoke test.
    SMOKE_TARGETS: Tuple[str, ...] = ("price",)
    SMOKE_ZONES: Tuple[str, ...] = ("NORD", "CSUD")
    SMOKE_MAX_FORECAST_DATES: int = 14
    SMOKE_MAX_EPOCHS: int = 2

    # Runtime calibration. These are representative strategy-target-zone fits.
    RUNTIME_CALIBRATION_TASKS: Tuple[Tuple[str, str, str], ...] = (
        ("tcn_168_c32_k3_do10_lr1em03_b64", "price", "NORD"),
        ("tcn_168_c64_k3_do20_lr5em04_b64", "price", "CSUD"),
        ("tcn_336_c32_k5_do20_lr1em03_b64", "purchases", "NORD"),
        ("tcn_336_c64_k3_do20_lr5em04_b64", "purchases", "SICI"),
        ("tcn_168_c32_k5_do20_lr1em03_b64", "sales", "CSUD"),
        ("tcn_336_c64_k5_do20_lr5em04_b64", "sales", "SARD"),
    )

    # Fast validation. It is a structural/output check, not a runtime estimator.
    FAST_TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    FAST_ZONES: Tuple[str, ...] = ("NORD", "CSUD", "SICI")
    FAST_MAX_VALIDATION_MONTHS: int = 2
    FAST_MAX_FORECAST_DATES_PER_SERIES: int = 31
    FAST_MAX_EPOCHS: int = 8
    FAST_STRATEGY_IDS: Tuple[str, ...] = (
        "tcn_168_c32_k3_do10_lr1em03_b64",
        "tcn_168_c64_k3_do20_lr5em04_b64",
        "tcn_336_c32_k3_do10_lr1em03_b64",
        "tcn_336_c64_k5_do20_lr5em04_b64",
    )

    # Manual fallback if final test is run alone before full validation results
    # are available. Replace these after inspecting validation results.
    SELECTED_STRATEGY_BY_TARGET: Dict[str, str] = field(
        default_factory=lambda: {
            "price": "tcn_168_c32_k3_do10_lr1em03_b64",
            "purchases": "tcn_336_c32_k3_do10_lr1em03_b64",
            "sales": "tcn_336_c32_k3_do10_lr1em03_b64",
        }
    )

    # Checkpointing and resume.
    SAVE_CHECKPOINTS: bool = True
    RESUME_FROM_CHECKPOINTS: bool = False
    OVERWRITE_EXISTING_CHECKPOINTS: bool = False
    SAVE_FITTED_MODELS: bool = False

    # Data paths. The script is intended to be launched from project root.
    PROJECT_ROOT: Path = field(default_factory=lambda: find_project_root())
    PANEL_PARQUET_REL: str = "data/processed/gme_model_panel_weather_hourly.parquet"
    PANEL_RDS_REL: str = "data/processed/gme_model_panel_weather_hourly.rds"
    EVAL_PARQUET_REL: str = "data/evaluation/eval_index_hourly.parquet"
    EVAL_RDS_REL: str = "data/evaluation/eval_index_hourly.rds"
    NAIVE_PARQUET_REL: str = "data/predictions/pred_naive_week_before.parquet"
    NAIVE_RDS_REL: str = "data/predictions/pred_naive_week_before.rds"
    PRED_DIR_REL: str = "data/predictions"
    FINAL_PRED_PARQUET_REL: str = "data/predictions/pred_tcn.parquet"
    FINAL_PRED_RDS_REL: str = "data/predictions/pred_tcn.rds"
    EXPERIMENT_DIR_REL: str = "experiments/tcn"

    # Feature groups.
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
    FUTURE_WEATHER_VARIABLES: Tuple[str, ...] = (
        "temperature_2m",
        "wind_speed_100m",
        "shortwave_radiation",
    )
    CALENDAR_VARIABLES: Tuple[str, ...] = (
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_month",
        "cos_month",
        "is_weekend",
    )
    REQUIRED_PANEL_COLUMNS: Tuple[str, ...] = (
        "datetime_model",
        "date",
        "hour",
        "zone",
        "price",
        "purchases",
        "sales",
    )
    REQUIRED_EVAL_COLUMNS: Tuple[str, ...] = (
        "target",
        "zone",
        "split",
        "forecast_date",
        "delivery_datetime_model",
        "delivery_date",
        "hour",
        "horizon",
        "y_true",
    )
    OFFICIAL_PREDICTION_COLUMNS: Tuple[str, ...] = (
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
        "kernel_size",
        "num_channels",
        "dilations",
        "receptive_field",
        "dropout",
        "learning_rate",
        "batch_size",
        "weight_norm",
        "weight_decay",
        "epochs_trained",
        "best_validation_loss",
        "fit_seconds",
        "predict_seconds",
    )

    # Miscellaneous.
    TRY_WRITE_RDS_OUTPUT: bool = True
    PARQUET_ENGINE: str = "pyarrow"
    FUTURE_HIDDEN_DIM: int = 64
    OUTPUT_HORIZON: int = 24
    RUNTIME_WARNING_HOURS: float = 9.0


# =============================================================================
# 2. PROJECT ROOT, STRATEGY GRID AND GLOBAL CONFIG
# =============================================================================


def find_project_root() -> Path:
    """Return a plausible project root.

    The preferred launch pattern is `python 11_tcn_model.py` from the project
    root. If the script is placed inside a scripts/ folder and launched from
    there, this function walks upwards until it finds a `data` directory.
    """
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "data").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def make_strategy_grid(max_epochs: int, weight_norm: bool) -> List[TCNStrategy]:
    base = [
        ("tcn_168_c32_k3_do10_lr1em03_b64", 168, [32, 32, 32, 32, 32, 32, 32], 3, 0.10, 1e-3, 64),
        ("tcn_168_c32_k3_do20_lr1em03_b64", 168, [32, 32, 32, 32, 32, 32, 32], 3, 0.20, 1e-3, 64),
        ("tcn_168_c64_k3_do10_lr5em04_b64", 168, [64, 64, 64, 64, 64, 64, 64], 3, 0.10, 5e-4, 64),
        ("tcn_168_c64_k3_do20_lr5em04_b64", 168, [64, 64, 64, 64, 64, 64, 64], 3, 0.20, 5e-4, 64),
        ("tcn_168_c32_k5_do10_lr1em03_b64", 168, [32, 32, 32, 32, 32, 32], 5, 0.10, 1e-3, 64),
        ("tcn_168_c32_k5_do20_lr1em03_b64", 168, [32, 32, 32, 32, 32, 32], 5, 0.20, 1e-3, 64),
        ("tcn_168_c64_k5_do10_lr5em04_b64", 168, [64, 64, 64, 64, 64, 64], 5, 0.10, 5e-4, 64),
        ("tcn_168_c64_k5_do20_lr5em04_b64", 168, [64, 64, 64, 64, 64, 64], 5, 0.20, 5e-4, 64),
        ("tcn_336_c32_k3_do10_lr1em03_b64", 336, [32, 32, 32, 32, 32, 32, 32, 32], 3, 0.10, 1e-3, 64),
        ("tcn_336_c32_k3_do20_lr1em03_b64", 336, [32, 32, 32, 32, 32, 32, 32, 32], 3, 0.20, 1e-3, 64),
        ("tcn_336_c64_k3_do10_lr5em04_b64", 336, [64, 64, 64, 64, 64, 64, 64, 64], 3, 0.10, 5e-4, 64),
        ("tcn_336_c64_k3_do20_lr5em04_b64", 336, [64, 64, 64, 64, 64, 64, 64, 64], 3, 0.20, 5e-4, 64),
        ("tcn_336_c32_k5_do10_lr1em03_b64", 336, [32, 32, 32, 32, 32, 32, 32], 5, 0.10, 1e-3, 64),
        ("tcn_336_c32_k5_do20_lr1em03_b64", 336, [32, 32, 32, 32, 32, 32, 32], 5, 0.20, 1e-3, 64),
        ("tcn_336_c64_k5_do10_lr5em04_b64", 336, [64, 64, 64, 64, 64, 64, 64], 5, 0.10, 5e-4, 64),
        ("tcn_336_c64_k5_do20_lr5em04_b64", 336, [64, 64, 64, 64, 64, 64, 64], 5, 0.20, 5e-4, 64),
    ]
    strategies = [
        TCNStrategy(
            strategy_id=sid,
            window_type="expanding",
            lookback_hours=lookback,
            window_months=None,
            num_channels=list(channels),
            kernel_size=kernel,
            dropout=dropout,
            learning_rate=lr,
            batch_size=batch,
            max_epochs=max_epochs,
            weight_norm=weight_norm,
        )
        for sid, lookback, channels, kernel, dropout, lr, batch in base
    ]
    for strategy in strategies:
        if strategy.receptive_field < 168:
            raise ValueError(
                f"TCN strategy {strategy.strategy_id} has receptive field "
                f"{strategy.receptive_field}, below the required minimum 168."
            )
    return strategies


CFG = Config()
TCN_STRATEGIES = make_strategy_grid(CFG.MAX_EPOCHS, CFG.WEIGHT_NORM)


# =============================================================================
# 3. LOGGING AND GENERAL UTILITIES
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path, engine: str = "pyarrow") -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, engine=engine)
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def serialize_config(cfg: Config) -> Dict[str, Any]:
    out = asdict(cfg)
    out["PROJECT_ROOT"] = str(cfg.PROJECT_ROOT)
    out["TCN_STRATEGIES"] = [strategy.as_log_dict() for strategy in TCN_STRATEGIES]
    return out


def setup_experiment_dirs(cfg: Config) -> Dict[str, Path]:
    root = cfg.PROJECT_ROOT / cfg.EXPERIMENT_DIR_REL
    paths = {
        "root": root,
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
        "training_curves": root / "training_curves",
        "validation_candidate_predictions": root / "validation_candidate_predictions",
        "selected_strategy": root / "selected_strategy",
        "model_artifacts": root / "model_artifacts",
        "learning_curve_figures": root / "figures" / "learning_curves",
    }
    for path in paths.values():
        ensure_dir(path)
    ensure_dir(cfg.PROJECT_ROOT / cfg.PRED_DIR_REL)
    return paths


def setup_logging(paths: Dict[str, Path]) -> logging.Logger:
    logger = logging.getLogger("tcn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(paths["logs"] / "tcn_run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def choose_device(cfg: Config, logger: logging.Logger) -> torch.device:
    if cfg.ALLOW_CUDA and torch.cuda.is_available():
        logger.info("Using CUDA device.")
        return torch.device("cuda")
    if cfg.ALLOW_MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS device.")
        return torch.device("mps")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def check_runtime_dependencies() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for parquet outputs. Install it with "
            "`pip install pyarrow` or `conda install pyarrow`."
        ) from exc


def safe_id(*parts: Any) -> str:
    raw = "__".join(str(p) for p in parts)
    raw = re.sub(r"[^A-Za-z0-9_\-.]+", "_", raw)
    return raw.strip("_")


def short_hash(payload: Any, length: int = 12) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def root_path(cfg: Config, rel: str) -> Path:
    return cfg.PROJECT_ROOT / rel




def to_datetime_naive_utc(values: Any) -> Any:
    """Convert datetimes to timezone-naive timestamps on the modelling grid.

    The upstream R pipeline stores `datetime_model` as a regular UTC-based
    artificial grid. For Python indexing we remove any timezone metadata so that
    pandas date ranges and panel timestamps match exactly.
    """
    return pd.to_datetime(values, utc=True).dt.tz_convert(None) if hasattr(values, "dt") or isinstance(values, pd.Series) else pd.to_datetime(values, utc=True).tz_convert(None)

def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def maybe_write_rds(df: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    try:
        import pyreadr
    except ImportError:
        logger.warning("pyreadr is not installed; skipping optional RDS output: %s", path)
        return
    ensure_dir(path.parent)
    try:
        pyreadr.write_rds(str(path), df)
        logger.info("Saved optional RDS output: %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write RDS output %s: %s", path, exc)


# =============================================================================
# 4. INPUT / OUTPUT FUNCTIONS
# =============================================================================


def read_dataframe_with_fallback(parquet_path: Path, rds_path: Path, label: str, logger: logging.Logger) -> pd.DataFrame:
    if parquet_path.exists():
        logger.info("Reading %s from parquet: %s", label, parquet_path)
        return pd.read_parquet(parquet_path)
    if rds_path.exists():
        logger.info("Reading %s from RDS: %s", label, rds_path)
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError(
                f"{label} parquet file was not found and reading the RDS fallback requires pyreadr. "
                "Install it with `pip install pyreadr` or create the parquet input upstream."
            ) from exc
        result = pyreadr.read_r(str(rds_path))
        if not result:
            raise ValueError(f"No objects found in RDS file: {rds_path}")
        return next(iter(result.values()))
    raise FileNotFoundError(f"Neither parquet nor RDS file found for {label}: {parquet_path}, {rds_path}")


def load_inputs(cfg: Config, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = read_dataframe_with_fallback(
        root_path(cfg, cfg.PANEL_PARQUET_REL),
        root_path(cfg, cfg.PANEL_RDS_REL),
        "hourly model panel",
        logger,
    )
    eval_index = read_dataframe_with_fallback(
        root_path(cfg, cfg.EVAL_PARQUET_REL),
        root_path(cfg, cfg.EVAL_RDS_REL),
        "common evaluation index",
        logger,
    )
    naive_week = read_dataframe_with_fallback(
        root_path(cfg, cfg.NAIVE_PARQUET_REL),
        root_path(cfg, cfg.NAIVE_RDS_REL),
        "weekly naive predictions",
        logger,
    )
    return panel, eval_index, naive_week


# =============================================================================
# 5. DATA PREPARATION AND FEATURE CONSTRUCTION
# =============================================================================


def prepare_panel(panel: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    panel = panel.copy()
    require_columns(panel, cfg.REQUIRED_PANEL_COLUMNS, "panel")
    panel["datetime_model"] = to_datetime_naive_utc(panel["datetime_model"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel["date_ts"] = pd.to_datetime(panel["date"])
    panel["hour"] = panel["hour"].astype(int)
    panel["zone"] = panel["zone"].astype(str)

    unexpected_hours = sorted(set(panel["hour"].dropna().astype(int)) - set(range(1, 25)))
    if unexpected_hours:
        raise ValueError(f"Unexpected hourly grid values found. Expected 1..24, got {unexpected_hours}")

    panel = add_calendar_features(panel)

    if "holiday" in panel.columns:
        panel["holiday"] = panel["holiday"].fillna(0)
        if not pd.api.types.is_numeric_dtype(panel["holiday"]):
            panel["holiday"] = panel["holiday"].astype(str).str.upper().isin({"TRUE", "T", "1", "YES"}).astype(int)
        else:
            panel["holiday"] = panel["holiday"].astype(float)
        logger.info("Optional holiday column found and included as a known covariate.")
    else:
        logger.info("Optional holiday column not found; skipping holiday covariate.")

    missing_optional = [
        col for col in list(cfg.OWN_ZONE_VARIABLES) + list(cfg.COMMON_MARKET_VARIABLES)
        if col not in panel.columns
    ]
    if missing_optional:
        logger.info("Optional panel columns not found and skipped: %s", sorted(set(missing_optional)))

    return panel.sort_values(["zone", "datetime_model"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour0 = df["hour"].astype(float) - 1.0
    date_ts = pd.to_datetime(df["date"])
    weekday = date_ts.dt.weekday.astype(float)
    month0 = date_ts.dt.month.astype(float) - 1.0
    df["sin_hour"] = np.sin(2 * np.pi * hour0 / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour0 / 24.0)
    df["sin_weekday"] = np.sin(2 * np.pi * weekday / 7.0)
    df["cos_weekday"] = np.cos(2 * np.pi * weekday / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * month0 / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * month0 / 12.0)
    df["is_weekend"] = (weekday >= 5).astype(int)
    return df


def build_zone_feature_frame(
    panel: pd.DataFrame,
    target_zone: str,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Build a time-indexed feature frame for one target zone.

    Cross-zone features are taken only from zones different from target_zone.
    The function fails if a target-zone cross feature is accidentally created.
    """
    zone_panel = panel.loc[panel["zone"] == target_zone].copy()
    if zone_panel.empty:
        raise ValueError(f"No panel rows found for target zone {target_zone}")

    zone_panel = zone_panel.drop_duplicates("datetime_model", keep="first").set_index("datetime_model").sort_index()

    hist_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in cfg.OWN_ZONE_VARIABLES:
        if col in zone_panel.columns:
            hist_cols.append(col)
            if col == "mti":
                categorical_cols.append(col)

    for col in cfg.COMMON_MARKET_VARIABLES:
        if col in zone_panel.columns and col not in hist_cols:
            hist_cols.append(col)

    for col in cfg.CALENDAR_VARIABLES:
        if col in zone_panel.columns and col not in hist_cols:
            hist_cols.append(col)
    if "holiday" in zone_panel.columns and "holiday" not in hist_cols:
        hist_cols.append("holiday")

    features = zone_panel[hist_cols].copy()

    cross_cols_created: List[str] = []
    available_cross_vars = [var for var in cfg.CROSS_ZONE_VARIABLES if var in panel.columns]
    other_zones = [zone for zone in cfg.ZONES if zone != target_zone]
    for var in available_cross_vars:
        wide = (
            panel.loc[panel["zone"].isin(other_zones), ["datetime_model", "zone", var]]
            .drop_duplicates(["datetime_model", "zone"], keep="first")
            .pivot(index="datetime_model", columns="zone", values=var)
            .sort_index()
        )
        for zone in other_zones:
            if zone in wide.columns:
                new_col = f"cross_{zone}_{var}"
                features[new_col] = wide[zone]
                cross_cols_created.append(new_col)
                hist_cols.append(new_col)
                if var == "mti":
                    categorical_cols.append(new_col)

    bad_cross = [col for col in cross_cols_created if col.startswith(f"cross_{target_zone}_")]
    if bad_cross:
        raise ValueError(
            f"Cross-zone leakage detected for target zone {target_zone}: {bad_cross}. "
            "The target zone must not be duplicated as a cross-zone predictor."
        )

    future_cols = [col for col in cfg.CALENDAR_VARIABLES if col in zone_panel.columns]
    if "holiday" in zone_panel.columns:
        future_cols.append("holiday")
    for col in cfg.FUTURE_WEATHER_VARIABLES:
        if col in zone_panel.columns and col not in future_cols:
            # Delivery-day weather is used as a day-ahead forecast proxy, not as target information.
            future_cols.append(col)

    # Future covariates must be available in the same feature frame.
    for col in future_cols:
        if col not in features.columns:
            features[col] = zone_panel[col]

    numeric_hist_cols = [col for col in hist_cols if col not in categorical_cols]
    logger.debug(
        "Feature frame for %s: %d hist numeric, %d hist categorical, %d future columns.",
        target_zone,
        len(numeric_hist_cols),
        len(categorical_cols),
        len(future_cols),
    )
    return features.sort_index(), numeric_hist_cols, categorical_cols, future_cols


def prepare_eval_index(eval_index: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    eval_index = eval_index.copy()
    require_columns(eval_index, cfg.REQUIRED_EVAL_COLUMNS, "eval_index")
    eval_index["target"] = eval_index["target"].astype(str)
    eval_index["zone"] = eval_index["zone"].astype(str)
    eval_index["split"] = eval_index["split"].astype(str)
    eval_index["forecast_date"] = pd.to_datetime(eval_index["forecast_date"]).dt.date
    eval_index["delivery_date"] = pd.to_datetime(eval_index["delivery_date"]).dt.date
    eval_index["delivery_datetime_model"] = to_datetime_naive_utc(eval_index["delivery_datetime_model"])
    eval_index["hour"] = eval_index["hour"].astype(int)
    eval_index["horizon"] = eval_index["horizon"].astype(int)
    eval_index["y_true"] = pd.to_numeric(eval_index["y_true"], errors="coerce")
    return eval_index.sort_values(["target", "zone", "delivery_datetime_model"]).reset_index(drop=True)


def prepare_naive_week(naive_week: pd.DataFrame) -> pd.DataFrame:
    naive_week = naive_week.copy()
    needed = ["target", "zone", "split", "delivery_datetime_model", "y_pred"]
    require_columns(naive_week, needed, "naive_week")
    naive_week["target"] = naive_week["target"].astype(str)
    naive_week["zone"] = naive_week["zone"].astype(str)
    naive_week["split"] = naive_week["split"].astype(str)
    naive_week["delivery_datetime_model"] = to_datetime_naive_utc(naive_week["delivery_datetime_model"])
    return naive_week.rename(columns={"y_pred": "y_pred_naive_week"})[
        ["target", "zone", "split", "delivery_datetime_model", "y_pred_naive_week"]
    ]


# =============================================================================
# 6. SUPERVISED DAILY SAMPLE CONSTRUCTION
# =============================================================================


@dataclass
class RawSamples:
    dates: List[pd.Timestamp]
    hist_frames: List[pd.DataFrame]
    future_frames: List[pd.DataFrame]
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.dates)


def daily_target_matrix(panel: pd.DataFrame, target: str, zone: str) -> pd.DataFrame:
    zone_panel = panel.loc[panel["zone"] == zone, ["date", "hour", "datetime_model", target]].copy()
    zone_panel["delivery_date"] = pd.to_datetime(zone_panel["date"])
    wide = zone_panel.pivot_table(index="delivery_date", columns="hour", values=target, aggfunc="first")
    wide = wide.reindex(columns=list(range(1, 25)))
    wide.columns = [f"h{int(c):02d}" for c in wide.columns]
    return wide.sort_index()


def infer_all_daily_dates(panel: pd.DataFrame, target: str, zone: str) -> List[pd.Timestamp]:
    y_wide = daily_target_matrix(panel, target, zone)
    valid = y_wide.dropna(how="any")
    return list(valid.index)


def select_training_dates(
    all_dates: Sequence[pd.Timestamp],
    stage: str,
    cfg: Config,
) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    dates = sorted(pd.to_datetime(list(all_dates)))
    validation_start = pd.Timestamp(f"{cfg.VALIDATION_YEAR}-01-01")
    validation_end = pd.Timestamp(f"{cfg.VALIDATION_YEAR}-12-31")

    if stage in {"validation", "fast_validation", "runtime_calibration", "smoke"}:
        outer_dates = [d for d in dates if d < validation_start]
    elif stage == "final_test":
        outer_dates = [d for d in dates if d <= validation_end]
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    if len(outer_dates) <= cfg.INNER_VALIDATION_DAYS:
        raise ValueError(f"Not enough outer training dates for stage {stage}: {len(outer_dates)}")

    inner_valid_dates = outer_dates[-cfg.INNER_VALIDATION_DAYS :]
    train_dates = outer_dates[: -cfg.INNER_VALIDATION_DAYS]
    return train_dates, inner_valid_dates


def eval_dates_from_index(
    eval_index: pd.DataFrame,
    target: str,
    zone: str,
    split: str,
    max_dates: Optional[int] = None,
    max_months: Optional[int] = None,
) -> List[pd.Timestamp]:
    df = eval_index.loc[
        (eval_index["target"] == target) &
        (eval_index["zone"] == zone) &
        (eval_index["split"] == split)
    ].copy()
    if df.empty:
        return []
    df["delivery_date_ts"] = pd.to_datetime(df["delivery_date"])
    dates = sorted(df["delivery_date_ts"].drop_duplicates())
    if max_months is not None:
        months = sorted({(d.year, d.month) for d in dates})[:max_months]
        allowed = set(months)
        dates = [d for d in dates if (d.year, d.month) in allowed]
    if max_dates is not None:
        dates = dates[:max_dates]
    return dates


def build_raw_samples(
    panel: pd.DataFrame,
    feature_frame: pd.DataFrame,
    target: str,
    zone: str,
    delivery_dates: Sequence[pd.Timestamp],
    strategy: TCNStrategy,
    future_cols: Sequence[str],
) -> RawSamples:
    target_series = (
        panel.loc[panel["zone"] == zone, ["datetime_model", target]]
        .drop_duplicates("datetime_model", keep="first")
        .assign(datetime_model=lambda d: to_datetime_naive_utc(d["datetime_model"]))
        .set_index("datetime_model")
        .sort_index()[target]
    )

    dates: List[pd.Timestamp] = []
    hist_frames: List[pd.DataFrame] = []
    future_frames: List[pd.DataFrame] = []
    y_list: List[np.ndarray] = []

    for delivery_date in sorted(pd.to_datetime(list(delivery_dates))):
        start = pd.Timestamp(delivery_date).normalize()
        hist_index = pd.date_range(
            start=start - pd.Timedelta(hours=strategy.lookback_hours),
            periods=strategy.lookback_hours,
            freq="h",
        )
        future_index = pd.date_range(start=start, periods=24, freq="h")

        if not hist_index.isin(feature_frame.index).all():
            continue
        if not future_index.isin(feature_frame.index).all():
            continue
        if not future_index.isin(target_series.index).all():
            continue

        y_values = target_series.reindex(future_index).to_numpy(dtype=float)
        if np.isnan(y_values).any():
            continue

        hist = feature_frame.reindex(hist_index).copy()
        fut = feature_frame.reindex(future_index)[list(future_cols)].copy()
        dates.append(start)
        hist_frames.append(hist)
        future_frames.append(fut)
        y_list.append(y_values)

    if y_list:
        y = np.vstack(y_list).astype(np.float32)
    else:
        y = np.empty((0, 24), dtype=np.float32)
    return RawSamples(dates=dates, hist_frames=hist_frames, future_frames=future_frames, y=y)


# =============================================================================
# 7. LEAKAGE-SAFE PREPROCESSORS
# =============================================================================


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class FittedPreprocessors:
    hist_preprocessor: ColumnTransformer
    future_preprocessor: Pipeline
    target_scaler: StandardScaler
    hist_numeric_cols: List[str]
    hist_categorical_cols: List[str]
    future_cols: List[str]
    hist_feature_names: List[str]
    future_feature_names: List[str]


def fit_preprocessors(
    train_samples: RawSamples,
    hist_numeric_cols: Sequence[str],
    hist_categorical_cols: Sequence[str],
    future_cols: Sequence[str],
) -> FittedPreprocessors:
    if train_samples.n_samples == 0:
        raise ValueError("Cannot fit preprocessors with zero training samples.")

    hist_numeric_cols = list(hist_numeric_cols)
    hist_categorical_cols = list(hist_categorical_cols)
    future_cols = list(future_cols)

    hist_rows = pd.concat(train_samples.hist_frames, axis=0, ignore_index=True)
    for col in hist_categorical_cols:
        if col in hist_rows.columns:
            hist_rows[col] = hist_rows[col].astype("object").where(hist_rows[col].notna(), "MISSING").astype(str)
    for col in hist_numeric_cols:
        if col in hist_rows.columns:
            hist_rows[col] = pd.to_numeric(hist_rows[col], errors="coerce")

    transformers: List[Tuple[str, Any, List[str]]] = []
    if hist_numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            hist_numeric_cols,
        ))
    if hist_categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("onehot", make_onehot_encoder()),
            ]),
            hist_categorical_cols,
        ))

    hist_preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    hist_preprocessor.fit(hist_rows)

    future_rows = pd.concat(train_samples.future_frames, axis=0, ignore_index=True)
    for col in future_cols:
        future_rows[col] = pd.to_numeric(future_rows[col], errors="coerce")
    future_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    future_preprocessor.fit(future_rows[future_cols])

    target_scaler = StandardScaler()
    target_scaler.fit(train_samples.y.reshape(-1, 1))

    hist_feature_names = get_column_transformer_feature_names(hist_preprocessor)
    future_feature_names = list(future_cols)

    return FittedPreprocessors(
        hist_preprocessor=hist_preprocessor,
        future_preprocessor=future_preprocessor,
        target_scaler=target_scaler,
        hist_numeric_cols=hist_numeric_cols,
        hist_categorical_cols=hist_categorical_cols,
        future_cols=future_cols,
        hist_feature_names=hist_feature_names,
        future_feature_names=future_feature_names,
    )


def get_column_transformer_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: List[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names.extend(list(transformer.get_feature_names_out(cols)))
                    continue
                except Exception:
                    pass
            names.extend(list(cols))
        return names


def transform_samples(samples: RawSamples, prep: FittedPreprocessors) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples.n_samples == 0:
        return (
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 24), dtype=np.float32),
        )

    x_hist_list: List[np.ndarray] = []
    x_future_list: List[np.ndarray] = []

    for hist_df, future_df in zip(samples.hist_frames, samples.future_frames):
        hist = hist_df.copy()
        for col in prep.hist_categorical_cols:
            if col in hist.columns:
                hist[col] = hist[col].astype("object").where(hist[col].notna(), "MISSING").astype(str)
        for col in prep.hist_numeric_cols:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
        x_hist = prep.hist_preprocessor.transform(hist)
        x_hist_list.append(np.asarray(x_hist, dtype=np.float32))

        fut = future_df.copy()
        for col in prep.future_cols:
            fut[col] = pd.to_numeric(fut[col], errors="coerce")
        x_fut = prep.future_preprocessor.transform(fut[prep.future_cols])
        x_future_list.append(np.asarray(x_fut, dtype=np.float32).reshape(-1))

    x_hist_arr = np.stack(x_hist_list).astype(np.float32)
    x_future_arr = np.stack(x_future_list).astype(np.float32)
    y_scaled = prep.target_scaler.transform(samples.y.reshape(-1, 1)).reshape(samples.y.shape).astype(np.float32)
    return x_hist_arr, x_future_arr, y_scaled


def inverse_transform_y(y_scaled: np.ndarray, prep: FittedPreprocessors) -> np.ndarray:
    return prep.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)


# =============================================================================
# 8. PYTORCH DATASET AND TCN MODULES
# =============================================================================


class DailyTCNDataset(Dataset):
    def __init__(self, x_hist: np.ndarray, x_future: np.ndarray, y: np.ndarray):
        self.x_hist = torch.as_tensor(x_hist, dtype=torch.float32)
        self.x_future = torch.as_tensor(x_future, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_hist.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_hist[idx], self.x_future[idx], self.y[idx]


class CausalConv1d(nn.Module):
    """A causal 1D convolution implemented by explicit left padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, weight_norm: bool):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        if weight_norm:
            conv = apply_weight_norm(conv)
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float, weight_norm: bool):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, weight_norm)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, weight_norm)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        residual = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + residual)


class TCNEncoder(nn.Module):
    def __init__(self, input_size: int, num_channels: Sequence[int], kernel_size: int, dropout: float, weight_norm: bool):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout, weight_norm))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Natural input shape: (batch, sequence_length, features).
        # Conv1d expected shape: (batch, channels, sequence_length).
        x = x.permute(0, 2, 1)
        return self.network(x)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_hist_features: int,
        n_future_features: int,
        strategy: TCNStrategy,
        future_hidden_dim: int,
        output_horizon: int = 24,
    ):
        super().__init__()
        self.encoder = TCNEncoder(
            input_size=n_hist_features,
            num_channels=strategy.num_channels,
            kernel_size=strategy.kernel_size,
            dropout=strategy.dropout,
            weight_norm=strategy.weight_norm,
        )
        last_channels = strategy.num_channels[-1]
        self.future_branch = nn.Sequential(
            nn.Linear(n_future_features, future_hidden_dim),
            nn.ReLU(),
            nn.Dropout(strategy.dropout),
        )
        self.output_head = nn.Sequential(
            nn.Linear(last_channels + future_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(strategy.dropout),
            nn.Linear(128, output_horizon),
        )

    def forward(self, x_hist: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x_hist)
        last_hidden = encoded[:, :, -1]
        future_repr = self.future_branch(x_future)
        combined = torch.cat([last_hidden, future_repr], dim=1)
        return self.output_head(combined)


# =============================================================================
# 9. TRAINING AND PREDICTION FUNCTIONS
# =============================================================================


@dataclass
class TrainResult:
    model: TCNForecaster
    history: pd.DataFrame
    best_validation_loss: float
    epochs_trained: int
    fit_seconds: float


def train_tcn_model(
    x_train_hist: np.ndarray,
    x_train_future: np.ndarray,
    y_train: np.ndarray,
    x_val_hist: np.ndarray,
    x_val_future: np.ndarray,
    y_val: np.ndarray,
    strategy: TCNStrategy,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> TrainResult:
    start = time.perf_counter()
    model = TCNForecaster(
        n_hist_features=x_train_hist.shape[2],
        n_future_features=x_train_future.shape[1],
        strategy=strategy,
        future_hidden_dim=cfg.FUTURE_HIDDEN_DIM,
        output_horizon=cfg.OUTPUT_HORIZON,
    ).to(device)

    if cfg.LOSS_FUNCTION != "mse":
        raise ValueError(f"Unsupported loss function: {cfg.LOSS_FUNCTION}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=strategy.learning_rate, weight_decay=cfg.WEIGHT_DECAY)

    train_ds = DailyTCNDataset(x_train_hist, x_train_future, y_train)
    val_ds = DailyTCNDataset(x_val_hist, x_val_future, y_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=strategy.batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=strategy.batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        drop_last=False,
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    epochs_no_improve = 0
    rows: List[Dict[str, Any]] = []

    for epoch in range(1, strategy.max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for xb_hist, xb_future, yb in train_loader:
            xb_hist = xb_hist.to(device)
            xb_future = xb_future.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb_hist, xb_future)
            loss = criterion(pred, yb)
            loss.backward()
            if cfg.GRADIENT_CLIP_NORM is not None and cfg.GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_loss, val_mae = evaluate_scaled_loss(model, val_loader, criterion, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "inner_validation_loss": val_loss,
            "inner_validation_mae_scaled": val_mae,
        })

        if val_loss < best_loss - cfg.MIN_DELTA:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.PATIENCE:
            logger.debug("Early stopping at epoch %d for %s", epoch, strategy.strategy_id)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    fit_seconds = time.perf_counter() - start
    history = pd.DataFrame(rows)
    return TrainResult(
        model=model,
        history=history,
        best_validation_loss=float(best_loss),
        epochs_trained=int(len(history)),
        fit_seconds=float(fit_seconds),
    )


def evaluate_scaled_loss(
    model: TCNForecaster,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    abs_errors: List[float] = []
    with torch.no_grad():
        for xb_hist, xb_future, yb in loader:
            xb_hist = xb_hist.to(device)
            xb_future = xb_future.to(device)
            yb = yb.to(device)
            pred = model(xb_hist, xb_future)
            loss = criterion(pred, yb)
            losses.append(float(loss.detach().cpu().item()))
            abs_errors.append(float(torch.mean(torch.abs(pred - yb)).detach().cpu().item()))
    return float(np.mean(losses)), float(np.mean(abs_errors))


def predict_tcn(
    model: TCNForecaster,
    x_hist: np.ndarray,
    x_future: np.ndarray,
    strategy: TCNStrategy,
    cfg: Config,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    ds = DailyTCNDataset(x_hist, x_future, np.zeros((x_hist.shape[0], cfg.OUTPUT_HORIZON), dtype=np.float32))
    loader = DataLoader(ds, batch_size=strategy.batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)
    preds: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb_hist, xb_future, _ in loader:
            xb_hist = xb_hist.to(device)
            xb_future = xb_future.to(device)
            pred = model(xb_hist, xb_future).detach().cpu().numpy()
            preds.append(pred)
    out = np.vstack(preds) if preds else np.empty((0, cfg.OUTPUT_HORIZON), dtype=np.float32)
    return out, float(time.perf_counter() - start)


# =============================================================================
# 10. CHECKPOINTS, TRAINING CURVES AND PREDICTION ASSEMBLY
# =============================================================================


def task_signature(
    stage: str,
    split: str,
    strategy: TCNStrategy,
    target: str,
    zone: str,
    cfg: Config,
    eval_limit_key: str,
) -> str:
    payload = {
        "stage": stage,
        "split": split,
        "strategy": strategy.as_log_dict(),
        "target": target,
        "zone": zone,
        "inner_validation_days": cfg.INNER_VALIDATION_DAYS,
        "features": {
            "own": cfg.OWN_ZONE_VARIABLES,
            "common": cfg.COMMON_MARKET_VARIABLES,
            "cross": cfg.CROSS_ZONE_VARIABLES,
            "future_weather": cfg.FUTURE_WEATHER_VARIABLES,
        },
        "eval_limit_key": eval_limit_key,
        "script_version": "tcn_v1_validation_target_selection",
    }
    return short_hash(payload, length=16)


def checkpoint_paths(paths: Dict[str, Path], signature: str) -> Dict[str, Path]:
    return {
        "prediction": paths["checkpoints"] / "predictions" / f"{signature}.parquet",
        "log": paths["checkpoints"] / "logs" / f"{signature}.json",
        "curve": paths["training_curves"] / f"{signature}_curve.csv",
        "curve_fig": paths["learning_curve_figures"] / f"{signature}_curve.png",
        "model": paths["model_artifacts"] / f"{signature}_model.pt",
        "features": paths["model_artifacts"] / f"{signature}_features.json",
    }


def save_learning_curve(history: pd.DataFrame, path_csv: Path, path_png: Path) -> None:
    ensure_dir(path_csv.parent)
    ensure_dir(path_png.parent)
    atomic_write_csv(history, path_csv)
    if history.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["inner_validation_loss"], label="inner validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("TCN learning curve")
    plt.legend()
    plt.tight_layout()
    tmp = path_png.with_suffix(path_png.suffix + ".tmp.png")
    plt.savefig(tmp, dpi=150)
    plt.close()
    os.replace(tmp, path_png)


def prediction_rows_from_daily_matrix(
    pred_matrix: np.ndarray,
    samples: RawSamples,
    eval_index: pd.DataFrame,
    target: str,
    zone: str,
    split: str,
    model_name: str,
    strategy: TCNStrategy,
    train_result: TrainResult,
    predict_seconds: float,
    cfg: Config,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    eval_subset = eval_index.loc[
        (eval_index["target"] == target) &
        (eval_index["zone"] == zone) &
        (eval_index["split"] == split)
    ].copy()
    eval_subset["delivery_date_ts"] = pd.to_datetime(eval_subset["delivery_date"])
    eval_lookup = {
        (pd.Timestamp(row.delivery_date_ts).normalize(), int(row.hour)): row
        for row in eval_subset.itertuples(index=False)
    }

    for sample_idx, delivery_date in enumerate(samples.dates):
        for h in range(1, 25):
            key = (pd.Timestamp(delivery_date).normalize(), h)
            if key not in eval_lookup:
                continue
            erow = eval_lookup[key]
            rows.append({
                "model": model_name,
                "target": target,
                "zone": zone,
                "split": split,
                "forecast_date": erow.forecast_date,
                "delivery_datetime_model": erow.delivery_datetime_model,
                "delivery_date": erow.delivery_date,
                "hour": int(erow.hour),
                "horizon": int(erow.horizon),
                "y_true": float(erow.y_true),
                "y_pred": float(pred_matrix[sample_idx, h - 1]),
                "strategy_id": strategy.strategy_id,
                "lookback_hours": strategy.lookback_hours,
                "kernel_size": strategy.kernel_size,
                "num_channels": json.dumps(strategy.num_channels),
                "dilations": json.dumps(strategy.dilations),
                "receptive_field": strategy.receptive_field,
                "dropout": strategy.dropout,
                "learning_rate": strategy.learning_rate,
                "batch_size": strategy.batch_size,
                "weight_norm": strategy.weight_norm,
                "weight_decay": cfg.WEIGHT_DECAY,
                "epochs_trained": train_result.epochs_trained,
                "best_validation_loss": train_result.best_validation_loss,
                "fit_seconds": train_result.fit_seconds,
                "predict_seconds": predict_seconds,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 11. CORE FIT-PREDICT ROUTINE
# =============================================================================


def fit_predict_one_task(
    panel: pd.DataFrame,
    eval_index: pd.DataFrame,
    strategy: TCNStrategy,
    target: str,
    zone: str,
    split: str,
    stage: str,
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
    logger: logging.Logger,
    max_eval_dates: Optional[int] = None,
    max_eval_months: Optional[int] = None,
    max_epochs_override: Optional[int] = None,
    candidate_model: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if zone not in cfg.ZONES:
        raise ValueError(f"Unknown zone: {zone}")
    if target not in cfg.TARGETS:
        raise ValueError(f"Unknown target: {target}")

    active_strategy = TCNStrategy(**asdict(strategy))
    if max_epochs_override is not None:
        active_strategy.max_epochs = int(max_epochs_override)

    eval_limit_key = f"maxdates={max_eval_dates}|maxmonths={max_eval_months}|epochs={active_strategy.max_epochs}"
    sig = task_signature(stage, split, active_strategy, target, zone, cfg, eval_limit_key)
    ckpt = checkpoint_paths(paths, sig)

    if (
        cfg.SAVE_CHECKPOINTS
        and cfg.RESUME_FROM_CHECKPOINTS
        and not cfg.OVERWRITE_EXISTING_CHECKPOINTS
        and ckpt["prediction"].exists()
        and ckpt["log"].exists()
    ):
        logger.info("Loading checkpoint | stage=%s split=%s strategy=%s target=%s zone=%s", stage, split, active_strategy.strategy_id, target, zone)
        pred = pd.read_parquet(ckpt["prediction"])
        with ckpt["log"].open("r", encoding="utf-8") as f:
            log_payload = json.load(f)
        return pred, log_payload

    logger.info("Training TCN | stage=%s split=%s strategy=%s target=%s zone=%s", stage, split, active_strategy.strategy_id, target, zone)

    feature_frame, hist_numeric_cols, hist_categorical_cols, future_cols = build_zone_feature_frame(panel, zone, cfg, logger)
    all_dates = infer_all_daily_dates(panel, target, zone)
    train_dates, inner_val_dates = select_training_dates(all_dates, stage, cfg)
    eval_dates = eval_dates_from_index(
        eval_index,
        target=target,
        zone=zone,
        split=split,
        max_dates=max_eval_dates,
        max_months=max_eval_months,
    )

    if not eval_dates:
        raise ValueError(f"No evaluation dates for target={target}, zone={zone}, split={split}")

    train_raw = build_raw_samples(panel, feature_frame, target, zone, train_dates, active_strategy, future_cols)
    inner_raw = build_raw_samples(panel, feature_frame, target, zone, inner_val_dates, active_strategy, future_cols)
    eval_raw = build_raw_samples(panel, feature_frame, target, zone, eval_dates, active_strategy, future_cols)

    if train_raw.n_samples < cfg.MIN_INNER_TRAIN_SAMPLES:
        raise ValueError(f"Too few training samples for {target}-{zone}: {train_raw.n_samples}")
    if inner_raw.n_samples < cfg.MIN_INNER_VALIDATION_SAMPLES:
        raise ValueError(f"Too few inner validation samples for {target}-{zone}: {inner_raw.n_samples}")
    if train_raw.n_samples + inner_raw.n_samples < cfg.MIN_TRAIN_SAMPLES:
        raise ValueError(f"Too few total samples for {target}-{zone}: {train_raw.n_samples + inner_raw.n_samples}")
    if eval_raw.n_samples == 0:
        raise ValueError(f"No valid evaluation samples for {target}-{zone}-{split}")

    prep = fit_preprocessors(train_raw, hist_numeric_cols, hist_categorical_cols, future_cols)
    x_train_hist, x_train_future, y_train = transform_samples(train_raw, prep)
    x_inner_hist, x_inner_future, y_inner = transform_samples(inner_raw, prep)
    x_eval_hist, x_eval_future, _ = transform_samples(eval_raw, prep)

    train_result = train_tcn_model(
        x_train_hist=x_train_hist,
        x_train_future=x_train_future,
        y_train=y_train,
        x_val_hist=x_inner_hist,
        x_val_future=x_inner_future,
        y_val=y_inner,
        strategy=active_strategy,
        cfg=cfg,
        device=device,
        logger=logger,
    )

    y_pred_scaled, predict_seconds = predict_tcn(train_result.model, x_eval_hist, x_eval_future, active_strategy, cfg, device)
    y_pred = inverse_transform_y(y_pred_scaled, prep)

    model_name = f"{cfg.CANDIDATE_MODEL_PREFIX}{active_strategy.strategy_id}" if candidate_model else cfg.MODEL_NAME
    pred_df = prediction_rows_from_daily_matrix(
        pred_matrix=y_pred,
        samples=eval_raw,
        eval_index=eval_index,
        target=target,
        zone=zone,
        split=split,
        model_name=model_name,
        strategy=active_strategy,
        train_result=train_result,
        predict_seconds=predict_seconds,
        cfg=cfg,
    )

    total_seconds = train_result.fit_seconds + predict_seconds
    log_payload = {
        "status": "ok",
        "stage": stage,
        "split": split,
        "target": target,
        "zone": zone,
        "strategy_id": active_strategy.strategy_id,
        "fit_seconds": train_result.fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
        "epochs_trained": train_result.epochs_trained,
        "best_validation_loss": train_result.best_validation_loss,
        "n_train_samples": train_raw.n_samples,
        "n_inner_validation_samples": inner_raw.n_samples,
        "n_validation_prediction_days": eval_raw.n_samples if split == "validation" else 0,
        "n_test_prediction_days": eval_raw.n_samples if split == "test" else 0,
        "n_hist_features_raw_numeric": len(hist_numeric_cols),
        "n_hist_features_raw_categorical": len(hist_categorical_cols),
        "n_hist_features_encoded": int(x_train_hist.shape[2]),
        "n_future_features_flat": int(x_train_future.shape[1]),
        "n_features": int(x_train_hist.shape[2] + x_train_future.shape[1]),
        "hist_numeric_cols": hist_numeric_cols,
        "hist_categorical_cols": hist_categorical_cols,
        "future_cols": future_cols,
        **active_strategy.as_log_dict(),
    }

    save_learning_curve(train_result.history, ckpt["curve"], ckpt["curve_fig"])
    atomic_write_json({
        "hist_numeric_cols": hist_numeric_cols,
        "hist_categorical_cols": hist_categorical_cols,
        "future_cols": future_cols,
        "hist_feature_names_encoded": prep.hist_feature_names,
        "future_feature_names": prep.future_feature_names,
        "weather_future_note": "Delivery-day weather variables are treated as day-ahead weather forecast proxies.",
    }, ckpt["features"])

    if cfg.SAVE_FITTED_MODELS:
        ensure_dir(ckpt["model"].parent)
        torch.save({
            "model_state_dict": train_result.model.state_dict(),
            "strategy": active_strategy.as_log_dict(),
            "feature_metadata_path": str(ckpt["features"]),
        }, ckpt["model"])

    if cfg.SAVE_CHECKPOINTS:
        atomic_write_parquet(pred_df, ckpt["prediction"], engine=cfg.PARQUET_ENGINE)
        atomic_write_json(log_payload, ckpt["log"])

    del train_result
    gc.collect()
    return pred_df, log_payload


# =============================================================================
# 12. VALIDATION METRICS AND STRATEGY SELECTION
# =============================================================================


def compute_quick_metrics(pred: pd.DataFrame, naive_week: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pred.empty:
        empty_cols = ["strategy_id", "target", "zone", "split", "MAE", "RMSE", "rMAE", "n"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    pred = pred.copy()
    pred["delivery_datetime_model"] = to_datetime_naive_utc(pred["delivery_datetime_model"])
    merged = pred.merge(
        naive_week,
        on=["target", "zone", "split", "delivery_datetime_model"],
        how="left",
        validate="many_to_one",
    )
    merged["abs_error"] = (merged["y_true"] - merged["y_pred"]).abs()
    merged["sq_error"] = (merged["y_true"] - merged["y_pred"]) ** 2
    merged["abs_error_naive"] = (merged["y_true"] - merged["y_pred_naive_week"]).abs()

    by_tz = (
        merged.dropna(subset=["y_true", "y_pred"])
        .groupby(["strategy_id", "target", "zone", "split"], as_index=False)
        .agg(
            n=("abs_error", "size"),
            MAE=("abs_error", "mean"),
            RMSE=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            naive_MAE=("abs_error_naive", "mean"),
            fit_seconds=("fit_seconds", "max"),
            predict_seconds=("predict_seconds", "max"),
            epochs_trained=("epochs_trained", "max"),
            receptive_field=("receptive_field", "max"),
            lookback_hours=("lookback_hours", "max"),
            kernel_size=("kernel_size", "max"),
            dropout=("dropout", "max"),
            learning_rate=("learning_rate", "max"),
            batch_size=("batch_size", "max"),
        )
    )
    by_tz["rMAE"] = by_tz["MAE"] / by_tz["naive_MAE"]

    by_target = (
        by_tz.groupby(["strategy_id", "target", "split"], as_index=False)
        .agg(
            n=("n", "sum"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            rMAE=("rMAE", "mean"),
            naive_MAE=("naive_MAE", "mean"),
            fit_seconds=("fit_seconds", "sum"),
            predict_seconds=("predict_seconds", "sum"),
            epochs_trained=("epochs_trained", "mean"),
            receptive_field=("receptive_field", "max"),
            lookback_hours=("lookback_hours", "max"),
            kernel_size=("kernel_size", "max"),
            dropout=("dropout", "max"),
            learning_rate=("learning_rate", "max"),
            batch_size=("batch_size", "max"),
        )
        .sort_values(["target", "rMAE", "MAE"])
        .reset_index(drop=True)
    )
    return by_target, by_tz


def select_strategy_by_target(metrics_by_target: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for target in cfg.TARGETS:
        subset = metrics_by_target.loc[(metrics_by_target["target"] == target) & (metrics_by_target["split"] == "validation")].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["rMAE", "MAE", "RMSE"])
        best = subset.iloc[0].to_dict()
        rows.append(best)
    return pd.DataFrame(rows)


# =============================================================================
# 13. RUN MODES
# =============================================================================


def strategies_by_id() -> Dict[str, TCNStrategy]:
    return {strategy.strategy_id: strategy for strategy in TCN_STRATEGIES}


def run_validation_mode(
    panel: pd.DataFrame,
    eval_index: pd.DataFrame,
    naive_week: pd.DataFrame,
    strategies: Sequence[TCNStrategy],
    targets: Sequence[str],
    zones: Sequence[str],
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
    logger: logging.Logger,
    stage: str,
    max_eval_dates: Optional[int] = None,
    max_eval_months: Optional[int] = None,
    max_epochs_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_predictions: List[pd.DataFrame] = []
    all_logs: List[Dict[str, Any]] = []
    total_tasks = len(strategies) * len(targets) * len(zones)
    task_counter = 0

    for strategy in strategies:
        for target in targets:
            for zone in zones:
                task_counter += 1
                logger.info("Validation task %d/%d", task_counter, total_tasks)
                try:
                    pred_task, log_task = fit_predict_one_task(
                        panel=panel,
                        eval_index=eval_index,
                        strategy=strategy,
                        target=target,
                        zone=zone,
                        split="validation",
                        stage=stage,
                        cfg=cfg,
                        paths=paths,
                        device=device,
                        logger=logger,
                        max_eval_dates=max_eval_dates,
                        max_eval_months=max_eval_months,
                        max_epochs_override=max_epochs_override,
                        candidate_model=True,
                    )
                    all_predictions.append(pred_task)
                    all_logs.append(log_task)
                    pred_path = paths["validation_candidate_predictions"] / f"pred_{stage}_{strategy.strategy_id}_{target}_{zone}.parquet"
                    atomic_write_parquet(pred_task, pred_path, engine=cfg.PARQUET_ENGINE)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("TCN validation task failed: %s %s %s", strategy.strategy_id, target, zone)
                    all_logs.append({
                        "status": "failed",
                        "stage": stage,
                        "split": "validation",
                        "strategy_id": strategy.strategy_id,
                        "target": target,
                        "zone": zone,
                        "error": str(exc),
                    })

    pred = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    training_log = pd.DataFrame(all_logs)
    metrics_by_target, metrics_by_tz = compute_quick_metrics(pred, naive_week)
    selected = select_strategy_by_target(metrics_by_target, cfg)
    return pred, training_log, metrics_by_target, metrics_by_tz, selected


def run_runtime_calibration(
    panel: pd.DataFrame,
    eval_index: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    strategy_map = strategies_by_id()
    rows: List[Dict[str, Any]] = []

    for strategy_id, target, zone in cfg.RUNTIME_CALIBRATION_TASKS:
        if strategy_id not in strategy_map:
            raise ValueError(f"Unknown runtime calibration strategy: {strategy_id}")
        strategy = strategy_map[strategy_id]
        try:
            _, log_payload = fit_predict_one_task(
                panel=panel,
                eval_index=eval_index,
                strategy=strategy,
                target=target,
                zone=zone,
                split="validation",
                stage="runtime_calibration",
                cfg=cfg,
                paths=paths,
                device=device,
                logger=logger,
                max_eval_dates=None,
                max_eval_months=None,
                max_epochs_override=None,
                candidate_model=True,
            )
            rows.append(log_payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Runtime calibration task failed: %s %s %s", strategy_id, target, zone)
            rows.append({
                "status": "failed",
                "stage": "runtime_calibration",
                "strategy_id": strategy_id,
                "target": target,
                "zone": zone,
                "error": str(exc),
            })

    calibration = pd.DataFrame(rows)
    atomic_write_csv(calibration, paths["logs"] / "tcn_runtime_calibration.csv")

    ok = calibration.loc[calibration.get("status", "ok") == "ok"].copy()
    if ok.empty:
        estimate = pd.DataFrame([{
            "status": "failed",
            "message": "No successful runtime calibration tasks; cannot estimate runtime.",
        }])
    else:
        median_seconds = float(ok["total_seconds"].median())
        mean_seconds = float(ok["total_seconds"].mean())
        p75_seconds = float(ok["total_seconds"].quantile(0.75))
        n_full_validation_tasks = len(TCN_STRATEGIES) * len(cfg.TARGETS) * len(cfg.ZONES)
        n_final_tasks = len(cfg.TARGETS) * len(cfg.ZONES)
        full_val_sec = median_seconds * n_full_validation_tasks
        final_sec = median_seconds * n_final_tasks
        full_val_sec_p75 = p75_seconds * n_full_validation_tasks
        final_sec_p75 = p75_seconds * n_final_tasks
        estimate = pd.DataFrame([{
            "status": "ok",
            "median_seconds_per_fit": median_seconds,
            "mean_seconds_per_fit": mean_seconds,
            "p75_seconds_per_fit": p75_seconds,
            "number_of_full_validation_tasks": n_full_validation_tasks,
            "number_of_final_test_tasks": n_final_tasks,
            "estimated_full_validation_hours": full_val_sec / 3600.0,
            "estimated_final_test_hours": final_sec / 3600.0,
            "estimated_total_hours": (full_val_sec + final_sec) / 3600.0,
            "p75_estimated_full_validation_hours": full_val_sec_p75 / 3600.0,
            "p75_estimated_final_test_hours": final_sec_p75 / 3600.0,
            "p75_estimated_total_hours": (full_val_sec_p75 + final_sec_p75) / 3600.0,
            "warning_p75_exceeds_9_hours": (full_val_sec_p75 + final_sec_p75) / 3600.0 > cfg.RUNTIME_WARNING_HOURS,
        }])

    atomic_write_csv(estimate, paths["logs"] / "tcn_runtime_estimate.csv")
    logger.info("Runtime calibration report:\n%s", estimate.to_string(index=False))
    return calibration, estimate


def load_selected_strategy_table_or_config(cfg: Config, paths: Dict[str, Path], logger: logging.Logger) -> pd.DataFrame:
    selected_path = paths["selected_strategy"] / "tcn_selected_strategy_by_target.csv"
    if selected_path.exists():
        logger.info("Loading selected TCN strategies from %s", selected_path)
        selected = pd.read_csv(selected_path)
        if "strategy_id" in selected.columns and "target" in selected.columns:
            return selected[["target", "strategy_id"]].drop_duplicates()
    logger.warning("Selected strategy table not found; using Config.SELECTED_STRATEGY_BY_TARGET.")
    return pd.DataFrame([
        {"target": target, "strategy_id": strategy_id}
        for target, strategy_id in cfg.SELECTED_STRATEGY_BY_TARGET.items()
    ])


def load_selected_validation_predictions(selected: pd.DataFrame, paths: Dict[str, Path], cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    files = list(paths["validation_candidate_predictions"].glob("pred_validation_*.parquet"))
    if not files:
        logger.warning("No validation candidate prediction files found; final pred_tcn may contain test only.")
        return pd.DataFrame()

    selected_pairs = {(str(row.target), str(row.strategy_id)) for row in selected.itertuples(index=False)}
    chunks: List[pd.DataFrame] = []
    for file in files:
        try:
            df = pd.read_parquet(file)
        except Exception:
            continue
        if df.empty or "strategy_id" not in df.columns:
            continue
        df = df.loc[df.apply(lambda r: (str(r["target"]), str(r["strategy_id"])) in selected_pairs, axis=1)].copy()
        if not df.empty:
            df["model"] = cfg.MODEL_NAME
            chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(["model", "target", "zone", "split", "delivery_datetime_model"], keep="first")
    return out


def run_final_test(
    panel: pd.DataFrame,
    eval_index: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
    logger: logging.Logger,
) -> pd.DataFrame:
    strategy_map = strategies_by_id()
    all_predictions: List[pd.DataFrame] = []

    for row in selected.itertuples(index=False):
        target = str(row.target)
        strategy_id = str(row.strategy_id)
        if strategy_id not in strategy_map:
            raise ValueError(f"Selected strategy {strategy_id} for {target} is not in TCN_STRATEGIES.")
        strategy = strategy_map[strategy_id]
        for zone in cfg.ZONES:
            try:
                pred_task, _ = fit_predict_one_task(
                    panel=panel,
                    eval_index=eval_index,
                    strategy=strategy,
                    target=target,
                    zone=zone,
                    split="test",
                    stage="final_test",
                    cfg=cfg,
                    paths=paths,
                    device=device,
                    logger=logger,
                    max_eval_dates=None,
                    max_eval_months=None,
                    max_epochs_override=None,
                    candidate_model=False,
                )
                all_predictions.append(pred_task)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Final TCN task failed: %s %s %s", strategy_id, target, zone)
                raise exc

    return pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()


# =============================================================================
# 14. FINAL OUTPUT VALIDATION AND LOG SAVING
# =============================================================================


def validate_prediction_file(pred: pd.DataFrame, eval_index: pd.DataFrame, cfg: Config, final_only: bool = False) -> pd.DataFrame:
    if pred.empty:
        raise ValueError("Prediction dataframe is empty.")
    pred = pred.copy()
    require_columns(pred, cfg.OFFICIAL_PREDICTION_COLUMNS, "TCN predictions")

    pred["forecast_date"] = pd.to_datetime(pred["forecast_date"]).dt.date
    pred["delivery_date"] = pd.to_datetime(pred["delivery_date"]).dt.date
    pred["delivery_datetime_model"] = to_datetime_naive_utc(pred["delivery_datetime_model"])
    pred["hour"] = pred["hour"].astype(int)
    pred["horizon"] = pred["horizon"].astype(int)
    pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")

    if pred["y_pred"].isna().any():
        bad = pred.loc[pred["y_pred"].isna()].head()
        raise ValueError(f"Missing y_pred values found. Example:\n{bad}")

    dup_key = ["model", "target", "zone", "split", "delivery_datetime_model"]
    dup = pred.duplicated(dup_key, keep=False)
    if dup.any():
        raise ValueError(f"Duplicated prediction rows found by {dup_key}. Examples:\n{pred.loc[dup, dup_key].head(20)}")

    horizons = sorted(pred["horizon"].dropna().unique().tolist())
    if horizons != list(range(1, 25)):
        raise ValueError(f"Unexpected horizons in predictions. Expected 1..24, got {horizons}")

    daily_counts = (
        pred.groupby(["model", "target", "zone", "split", "delivery_date"], as_index=False)
        .agg(n_hours=("hour", "nunique"))
    )
    bad_counts = daily_counts.loc[daily_counts["n_hours"] != 24]
    if not bad_counts.empty:
        raise ValueError(f"Some target-zone-delivery days do not have 24 predictions. Examples:\n{bad_counts.head(20)}")

    eval_key_cols = ["target", "zone", "split", "delivery_datetime_model"]
    eval_keys = eval_index[eval_key_cols].copy()
    eval_keys["delivery_datetime_model"] = to_datetime_naive_utc(eval_keys["delivery_datetime_model"])
    merged = pred.merge(eval_keys.drop_duplicates(), on=eval_key_cols, how="left", indicator=True)
    not_aligned = merged.loc[merged["_merge"] != "both"]
    if not not_aligned.empty:
        raise ValueError(f"Predictions do not align with eval_index. Examples:\n{not_aligned.head(20)}")

    if final_only:
        non_tcn = sorted(set(pred["model"]) - {cfg.MODEL_NAME})
        if non_tcn:
            raise ValueError(f"Final prediction file must contain only model='{cfg.MODEL_NAME}', found {non_tcn}")

    pred = pred[list(cfg.OFFICIAL_PREDICTION_COLUMNS)].sort_values(
        ["model", "split", "target", "zone", "delivery_datetime_model"]
    ).reset_index(drop=True)
    return pred


def append_or_write_logs(training_logs: List[pd.DataFrame], quick_metrics: List[pd.DataFrame], cfg: Config, paths: Dict[str, Path]) -> None:
    log_path = paths["logs"] / "tcn_training_logs.csv"
    metrics_path = paths["logs"] / "tcn_quick_metrics.csv"
    if training_logs:
        new_log = pd.concat([df for df in training_logs if df is not None and not df.empty], ignore_index=True) if any(not df.empty for df in training_logs) else pd.DataFrame()
        if log_path.exists() and not new_log.empty:
            old = pd.read_csv(log_path)
            new_log = pd.concat([old, new_log], ignore_index=True)
        atomic_write_csv(new_log, log_path)
    elif not log_path.exists():
        atomic_write_csv(pd.DataFrame(), log_path)

    if quick_metrics:
        new_metrics = pd.concat([df for df in quick_metrics if df is not None and not df.empty], ignore_index=True) if any(not df.empty for df in quick_metrics) else pd.DataFrame()
        if metrics_path.exists() and not new_metrics.empty:
            old = pd.read_csv(metrics_path)
            new_metrics = pd.concat([old, new_metrics], ignore_index=True)
        atomic_write_csv(new_metrics, metrics_path)
    elif not metrics_path.exists():
        atomic_write_csv(pd.DataFrame(), metrics_path)


def save_runtime_summary(start_time: float, training_logs: Sequence[pd.DataFrame], selected: pd.DataFrame, paths: Dict[str, Path]) -> None:
    elapsed = time.perf_counter() - start_time
    logs = pd.concat([df for df in training_logs if df is not None and not df.empty], ignore_index=True) if training_logs else pd.DataFrame()
    if logs.empty:
        summary = pd.DataFrame([{
            "total_runtime_seconds": elapsed,
            "n_successful_fits": 0,
            "n_failed_fits": 0,
        }])
    else:
        status = logs.get("status", pd.Series(["ok"] * len(logs)))
        summary = pd.DataFrame([{
            "total_runtime_seconds": elapsed,
            "total_runtime_hours": elapsed / 3600.0,
            "n_successful_fits": int((status == "ok").sum()),
            "n_failed_fits": int((status == "failed").sum()),
            "mean_fit_seconds": float(pd.to_numeric(logs.get("fit_seconds"), errors="coerce").mean()),
            "mean_predict_seconds": float(pd.to_numeric(logs.get("predict_seconds"), errors="coerce").mean()),
            "mean_epochs_trained": float(pd.to_numeric(logs.get("epochs_trained"), errors="coerce").mean()),
            "selected_strategy_by_target": selected.to_json(orient="records") if selected is not None and not selected.empty else None,
        }])
    atomic_write_csv(summary, paths["logs"] / "tcn_runtime_summary.csv")


# =============================================================================
# 15. MAIN EXECUTION
# =============================================================================


def main() -> None:
    cfg = CFG
    start_all = time.perf_counter()
    check_runtime_dependencies()
    set_reproducibility(cfg.RANDOM_SEED)
    paths = setup_experiment_dirs(cfg)
    logger = setup_logging(paths)
    device = choose_device(cfg, logger)

    logger.info("Starting TCN script. Project root: %s", cfg.PROJECT_ROOT)
    logger.info("Smoke test and fast validation are structural checks, not runtime estimators.")
    logger.info("Runtime calibration is the recommended step before full validation.")
    atomic_write_json(serialize_config(cfg), paths["logs"] / "tcn_config.json")

    panel_raw, eval_raw, naive_raw = load_inputs(cfg, logger)
    panel = prepare_panel(panel_raw, cfg, logger)
    eval_index = prepare_eval_index(eval_raw, cfg)
    naive_week = prepare_naive_week(naive_raw)

    training_logs: List[pd.DataFrame] = []
    quick_metrics_logs: List[pd.DataFrame] = []
    selected = pd.DataFrame()
    final_output_parts: List[pd.DataFrame] = []

    strategy_map = strategies_by_id()

    if cfg.RUN_SMOKE_TEST:
        logger.info("RUN_SMOKE_TEST=True: running smoke test alone.")
        strategy = TCN_STRATEGIES[0]
        pred, train_log, metrics_by_target, metrics_by_tz, selected = run_validation_mode(
            panel=panel,
            eval_index=eval_index,
            naive_week=naive_week,
            strategies=[strategy],
            targets=cfg.SMOKE_TARGETS,
            zones=cfg.SMOKE_ZONES,
            cfg=cfg,
            paths=paths,
            device=device,
            logger=logger,
            stage="smoke",
            max_eval_dates=cfg.SMOKE_MAX_FORECAST_DATES,
            max_eval_months=None,
            max_epochs_override=cfg.SMOKE_MAX_EPOCHS,
        )
        # Smoke output uses official structure so the R evaluation script can be tested.
        smoke_pred = validate_prediction_file(pred, eval_index, cfg, final_only=False)
        atomic_write_parquet(smoke_pred, root_path(cfg, cfg.FINAL_PRED_PARQUET_REL), engine=cfg.PARQUET_ENGINE)
        if cfg.TRY_WRITE_RDS_OUTPUT:
            maybe_write_rds(smoke_pred, root_path(cfg, cfg.FINAL_PRED_RDS_REL), logger)
        training_logs.append(train_log)
        quick_metrics_logs.append(metrics_by_target)
        atomic_write_csv(metrics_by_target, paths["selected_strategy"] / "tcn_validation_strategy_results.csv")
        atomic_write_csv(selected, paths["selected_strategy"] / "tcn_selected_strategy_by_target.csv")
        atomic_write_csv(metrics_by_tz, paths["selected_strategy"] / "tcn_validation_strategy_results_by_target_zone.csv")
        append_or_write_logs(training_logs, quick_metrics_logs, cfg, paths)
        save_runtime_summary(start_all, training_logs, selected, paths)
        logger.info("Smoke test completed. Smoke predictions saved to pred_tcn.parquet.")
        return

    if cfg.RUN_RUNTIME_CALIBRATION and not cfg.ALLOW_COMBINED_RUNTIME_MODE:
        logger.info("RUN_RUNTIME_CALIBRATION=True: running runtime calibration alone.")
        calibration, estimate = run_runtime_calibration(panel, eval_index, cfg, paths, device, logger)
        training_logs.append(calibration)
        append_or_write_logs(training_logs, [], cfg, paths)
        save_runtime_summary(start_all, training_logs, selected, paths)
        logger.info("Runtime calibration completed.")
        return

    validation_predictions = pd.DataFrame()
    metrics_by_target = pd.DataFrame()
    metrics_by_tz = pd.DataFrame()

    if cfg.RUN_RUNTIME_CALIBRATION:
        calibration, estimate = run_runtime_calibration(panel, eval_index, cfg, paths, device, logger)
        training_logs.append(calibration)

    if cfg.RUN_FAST_VALIDATION:
        logger.info("RUN_FAST_VALIDATION=True: running reduced validation experiment.")
        fast_strategies = [strategy_map[sid] for sid in cfg.FAST_STRATEGY_IDS if sid in strategy_map]
        pred_fast, train_log_fast, metrics_fast, metrics_tz_fast, selected_fast = run_validation_mode(
            panel=panel,
            eval_index=eval_index,
            naive_week=naive_week,
            strategies=fast_strategies,
            targets=cfg.FAST_TARGETS,
            zones=cfg.FAST_ZONES,
            cfg=cfg,
            paths=paths,
            device=device,
            logger=logger,
            stage="fast_validation",
            max_eval_dates=cfg.FAST_MAX_FORECAST_DATES_PER_SERIES,
            max_eval_months=cfg.FAST_MAX_VALIDATION_MONTHS,
            max_epochs_override=cfg.FAST_MAX_EPOCHS,
        )
        training_logs.append(train_log_fast)
        quick_metrics_logs.append(metrics_fast)
        validation_predictions = pd.concat([validation_predictions, pred_fast], ignore_index=True)
        metrics_by_target = pd.concat([metrics_by_target, metrics_fast], ignore_index=True)
        metrics_by_tz = pd.concat([metrics_by_tz, metrics_tz_fast], ignore_index=True)
        selected = selected_fast

    if cfg.RUN_FULL_VALIDATION:
        logger.info("RUN_FULL_VALIDATION=True: running full 16-strategy validation grid.")
        pred_full, train_log_full, metrics_full, metrics_tz_full, selected_full = run_validation_mode(
            panel=panel,
            eval_index=eval_index,
            naive_week=naive_week,
            strategies=TCN_STRATEGIES,
            targets=cfg.TARGETS,
            zones=cfg.ZONES,
            cfg=cfg,
            paths=paths,
            device=device,
            logger=logger,
            stage="validation",
            max_eval_dates=None,
            max_eval_months=None,
            max_epochs_override=None,
        )
        training_logs.append(train_log_full)
        quick_metrics_logs.append(metrics_full)
        validation_predictions = pd.concat([validation_predictions, pred_full], ignore_index=True)
        metrics_by_target = pd.concat([metrics_by_target, metrics_full], ignore_index=True)
        metrics_by_tz = pd.concat([metrics_by_tz, metrics_tz_full], ignore_index=True)
        selected = selected_full

    # Save validation logs if validation has run.
    if not metrics_by_target.empty:
        # Keep the last/full result when duplicate fast/full rows exist.
        metrics_by_target = metrics_by_target.sort_values(["target", "rMAE", "MAE"]).reset_index(drop=True)
        metrics_by_tz = metrics_by_tz.sort_values(["target", "zone", "rMAE", "MAE"]).reset_index(drop=True)
        selected = select_strategy_by_target(metrics_by_target, cfg)
        atomic_write_csv(metrics_by_target, paths["selected_strategy"] / "tcn_validation_strategy_results.csv")
        atomic_write_csv(selected, paths["selected_strategy"] / "tcn_selected_strategy_by_target.csv")
        atomic_write_csv(metrics_by_tz, paths["selected_strategy"] / "tcn_validation_strategy_results_by_target_zone.csv")
        logger.info("Selected TCN strategies by target:\n%s", selected[["target", "strategy_id", "rMAE", "MAE"]].to_string(index=False) if not selected.empty else "<none>")

    if cfg.RUN_FINAL_TEST:
        logger.info("RUN_FINAL_TEST=True: training selected final TCN models and evaluating 2025 test period.")
        if selected.empty:
            selected = load_selected_strategy_table_or_config(cfg, paths, logger)
        selected_validation = pd.DataFrame()
        if not validation_predictions.empty and "strategy_id" in validation_predictions.columns:
            selected_pairs = {(str(row.target), str(row.strategy_id)) for row in selected.itertuples(index=False)}
            selected_validation = validation_predictions.loc[
                validation_predictions.apply(lambda r: (str(r["target"]), str(r["strategy_id"])) in selected_pairs, axis=1)
            ].copy()
            if not selected_validation.empty:
                selected_validation["model"] = cfg.MODEL_NAME
        if selected_validation.empty:
            selected_validation = load_selected_validation_predictions(selected, paths, cfg, logger)
        if not selected_validation.empty:
            final_output_parts.append(selected_validation)

        test_pred = run_final_test(panel, eval_index, selected, cfg, paths, device, logger)
        final_output_parts.append(test_pred)

        final_pred = pd.concat(final_output_parts, ignore_index=True) if final_output_parts else pd.DataFrame()
        final_pred = validate_prediction_file(final_pred, eval_index, cfg, final_only=True)
        atomic_write_parquet(final_pred, root_path(cfg, cfg.FINAL_PRED_PARQUET_REL), engine=cfg.PARQUET_ENGINE)
        logger.info("Saved final selected TCN predictions: %s", root_path(cfg, cfg.FINAL_PRED_PARQUET_REL))
        if cfg.TRY_WRITE_RDS_OUTPUT:
            maybe_write_rds(final_pred, root_path(cfg, cfg.FINAL_PRED_RDS_REL), logger)

    append_or_write_logs(training_logs, quick_metrics_logs, cfg, paths)
    save_runtime_summary(start_all, training_logs, selected, paths)

    # Ensure required files exist even in modes that do not populate all of them.
    for required in [
        paths["logs"] / "tcn_training_logs.csv",
        paths["logs"] / "tcn_quick_metrics.csv",
        paths["logs"] / "tcn_runtime_summary.csv",
        paths["logs"] / "tcn_runtime_calibration.csv",
        paths["logs"] / "tcn_runtime_estimate.csv",
        paths["selected_strategy"] / "tcn_validation_strategy_results.csv",
        paths["selected_strategy"] / "tcn_selected_strategy_by_target.csv",
        paths["selected_strategy"] / "tcn_validation_strategy_results_by_target_zone.csv",
    ]:
        if not required.exists():
            atomic_write_csv(pd.DataFrame(), required)

    logger.info("TCN script completed.")


if __name__ == "__main__":
    main()
