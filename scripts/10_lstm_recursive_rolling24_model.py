#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_lstm_recursive_rolling24_model.py

"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

_DEFAULT_PROJECT_ROOT = Path.cwd()


@dataclass
class LSTMStrategy:
    strategy_id: str
    lookback_hours: int
    window_months: int
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int


@dataclass
class Config:
    PROJECT_ROOT: Path = _DEFAULT_PROJECT_ROOT

    PANEL_PARQUET_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/processed/gme_model_panel_weather_hourly.parquet"
    PANEL_RDS_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/processed/gme_model_panel_weather_hourly.rds"
    EVAL_INDEX_PARQUET_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/evaluation/eval_index_hourly.parquet"
    EVAL_INDEX_RDS_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/evaluation/eval_index_hourly.rds"
    WEEKLY_NAIVE_PARQUET_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_naive_week_before.parquet"
    WEEKLY_NAIVE_RDS_PATH: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_naive_week_before.rds"

    PREDICTIONS_DIR: Path = _DEFAULT_PROJECT_ROOT / "data/predictions"
    FINAL_PREDICTION_PARQUET: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_lstm_recursive_rolling24.parquet"
    FINAL_PREDICTION_RDS: Path = _DEFAULT_PROJECT_ROOT / "data/predictions/pred_lstm_recursive_rolling24.rds"
    EXPERIMENT_DIR: Path = _DEFAULT_PROJECT_ROOT / "experiments/lstm_recursive_rolling24"

    # Start safely. Change these flags after the smoke test.
    RUN_SMOKE_TEST: bool = False
    RUN_FAST_VALIDATION: bool = False
    RUN_FULL_VALIDATION: bool = False
    RUN_FINAL_TEST: bool = True

    SAVE_CHECKPOINTS: bool = True
    RESUME_FROM_CHECKPOINTS: bool = True
    OVERWRITE_EXISTING_CHECKPOINTS: bool = False
    SAVE_FITTED_MODELS: bool = False

    MODEL_NAME_FINAL: str = "lstm_recursive_rolling24_selected"
    MODEL_NAME_SMOKE: str = "lstm_recursive_smoke"

    TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    ZONES: Tuple[str, ...] = ("NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD")

    INITIAL_TRAIN_START_DATE: str = "2021-01-01"
    RECALIBRATION_FREQUENCY: str = "none"  # keep fixed for comparability
    WINDOW_MONTHS: int = 24

    INNER_VALIDATION_DAYS: int = 30
    MIN_TRAIN_SAMPLES: int = 24 * 60
    MIN_INNER_TRAIN_SAMPLES: int = 24 * 30
    MIN_INNER_VALIDATION_SAMPLES: int = 24 * 7

    # Basic grid only. This is not intended to reproduce the large search.
    LOOKBACK_OPTIONS: Tuple[int, ...] = (168, 336)
    HIDDEN_SIZE_OPTIONS: Tuple[int, ...] = (64,)
    NUM_LAYERS_OPTIONS: Tuple[int, ...] = (1,)
    LEARNING_RATE_OPTIONS: Tuple[float, ...] = (1e-3, 5e-4)
    BATCH_SIZE_OPTIONS: Tuple[int, ...] = (128,)

    MAX_EPOCHS: int = 20
    FAST_MAX_EPOCHS: int = 12
    PATIENCE: int = 4
    MIN_DELTA: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    GRADIENT_CLIP_NORM: float = 1.0

    ALLOW_CUDA: bool = False
    ALLOW_MPS: bool = False
    RANDOM_SEED: int = 123
    NUM_WORKERS: int = 0

    SMOKE_TARGETS: Tuple[str, ...] = ("price",)
    SMOKE_ZONES: Tuple[str, ...] = ("NORD", "CSUD")
    SMOKE_MAX_FORECAST_DATES: int = 7

    # Optional mini-validation. It uses the last validation dates, not January.
    FAST_TARGETS: Tuple[str, ...] = ("price", "purchases", "sales")
    FAST_ZONES: Tuple[str, ...] = ("NORD", "CSUD", "SARD")
    FAST_MAX_FORECAST_DATES_PER_SERIES: Optional[int] = 31

    SELECTED_STRATEGY_BY_TARGET: Dict[str, str] = field(
        default_factory=lambda: {
            "price": "lstm_rec24_336h_h64_l1_lr5em04_b128",
            "purchases": "lstm_rec24_168h_h64_l1_lr5em04_b128",
            "sales": "lstm_rec24_168h_h64_l1_lr5em04_b128",
        }
    )

    WEATHER_VARIABLES: Tuple[str, ...] = (
        "temperature_2m",
        "wind_speed_100m",
        "shortwave_radiation",
    )

    REQUIRED_PREDICTION_COLUMNS: Tuple[str, ...] = (
        "model", "target", "zone", "split", "forecast_date",
        "delivery_datetime_model", "delivery_date", "hour", "horizon", "y_true", "y_pred"
    )


# =============================================================================
# 2. BASIC UTILITIES
# =============================================================================


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config) -> logging.Logger:
    ensure_dir(cfg.EXPERIMENT_DIR / "logs")
    logger = logging.getLogger("lstm_recursive")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(cfg.EXPERIMENT_DIR / "logs/lstm_recursive_run.log", mode="a", encoding="utf-8")
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
# 3. I/O
# =============================================================================


def read_rds(path: Path) -> pd.DataFrame:
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:
        raise ImportError(f"Cannot read {path}; pyreadr is not installed.") from exc
    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"pyreadr could not read any object from {path}")
    return next(iter(result.values()))


def load_table(parquet_path: Path, rds_path: Path, name: str) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if rds_path.exists():
        return read_rds(rds_path)
    raise FileNotFoundError(f"Missing {name}. Expected {parquet_path} or {rds_path}.")


def load_weekly_naive(cfg: Config, logger: logging.Logger) -> Optional[pd.DataFrame]:
    try:
        return load_table(cfg.WEEKLY_NAIVE_PARQUET_PATH, cfg.WEEKLY_NAIVE_RDS_PATH, "weekly naive predictions")
    except Exception as exc:
        logger.warning("Weekly naive not loaded; quick rMAE skipped. Error: %s", exc)
        return None


def save_outputs(df: pd.DataFrame, parquet_path: Path, rds_path: Path, logger: logging.Logger) -> None:
    ensure_dir(parquet_path.parent)
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet: %s", parquet_path)
    try:
        import pyreadr  # type: ignore
        pyreadr.write_rds(str(rds_path), df)
        logger.info("Saved RDS: %s", rds_path)
    except Exception as exc:
        logger.warning("Could not save RDS %s: %s", rds_path, exc)


# =============================================================================
# 4. DATA PREPARATION
# =============================================================================


def require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def normalize_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def prepare_panel(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    require_columns(panel, ["datetime_model", "date", "hour", "zone", "price", "purchases", "sales"], "panel")
    panel = panel.copy()
    panel["datetime_model"] = normalize_datetime(panel["datetime_model"])
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["hour"] = panel["hour"].astype(int)
    panel["zone"] = panel["zone"].astype(str)

    panel["weekday"] = panel["date"].dt.weekday + 1
    panel["month"] = panel["date"].dt.month
    panel["is_weekend"] = panel["weekday"].isin([6, 7]).astype(float)
    panel["sin_hour"] = np.sin(2 * np.pi * (panel["hour"].astype(float) - 1) / 24)
    panel["cos_hour"] = np.cos(2 * np.pi * (panel["hour"].astype(float) - 1) / 24)
    panel["sin_weekday"] = np.sin(2 * np.pi * (panel["weekday"].astype(float) - 1) / 7)
    panel["cos_weekday"] = np.cos(2 * np.pi * (panel["weekday"].astype(float) - 1) / 7)
    panel["sin_month"] = np.sin(2 * np.pi * (panel["month"].astype(float) - 1) / 12)
    panel["cos_month"] = np.cos(2 * np.pi * (panel["month"].astype(float) - 1) / 12)
    if "holiday" in panel.columns:
        panel["holiday"] = pd.to_numeric(panel["holiday"], errors="coerce").fillna(0.0)

    panel = panel[panel["zone"].isin(cfg.ZONES)].copy()
    return panel.sort_values(["zone", "datetime_model"]).reset_index(drop=True)


def prepare_eval_index(eval_index: pd.DataFrame) -> pd.DataFrame:
    required = ["target", "zone", "split", "forecast_date", "delivery_datetime_model", "delivery_date", "hour", "horizon", "y_true"]
    require_columns(eval_index, required, "eval_index")
    out = eval_index.copy()
    out["target"] = out["target"].astype(str)
    out["zone"] = out["zone"].astype(str)
    out["split"] = out["split"].astype(str)
    out["forecast_date"] = pd.to_datetime(out["forecast_date"], errors="coerce")
    out["delivery_datetime_model"] = normalize_datetime(out["delivery_datetime_model"])
    out["delivery_date"] = pd.to_datetime(out["delivery_date"], errors="coerce")
    out["hour"] = out["hour"].astype(int)
    out["horizon"] = out["horizon"].astype(int)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    return out


def available_columns(cols: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [col for col in cols if col in df.columns]


def known_covariate_columns(panel: pd.DataFrame, cfg: Config) -> List[str]:
    cols = ["sin_hour", "cos_hour", "sin_weekday", "cos_weekday", "sin_month", "cos_month", "is_weekend"]
    if "holiday" in panel.columns:
        cols.append("holiday")
    cols += available_columns(cfg.WEATHER_VARIABLES, panel)
    return list(dict.fromkeys(cols))


def build_recursive_feature_frame(panel: pd.DataFrame, target: str, zone: str, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """Create leakage-safe recursive features for one target-zone task."""
    z = panel[panel["zone"] == zone].copy()
    if z.empty:
        raise ValueError(f"No panel rows for zone={zone}")
    cov_base = known_covariate_columns(z, cfg)
    cols = ["datetime_model", target] + cov_base
    feat = z[cols].copy()
    rename = {target: "recursive_target"}
    for col in cov_base:
        rename[col] = f"weather_{col}" if col in cfg.WEATHER_VARIABLES else f"cal_{col}"
    feat = feat.rename(columns=rename).set_index("datetime_model").sort_index()
    feat = feat[~feat.index.duplicated(keep="first")]
    for col in feat.columns:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
    cov_cols = [col for col in feat.columns if col != "recursive_target"]
    return feat, cov_cols


# =============================================================================
# 5. PREPROCESSING AND ARRAYS
# =============================================================================


class SequencePreprocessor:
    def __init__(self, cols: Sequence[str]) -> None:
        self.cols = list(cols)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "SequencePreprocessor":
        x = df.reindex(columns=self.cols)
        x_imp = self.imputer.fit_transform(x)
        self.scaler.fit(x_imp)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df.reindex(columns=self.cols)
        x_imp = self.imputer.transform(x)
        return self.scaler.transform(x_imp).astype(np.float32)


class CovariatePreprocessor:
    def __init__(self, cols: Sequence[str]) -> None:
        self.cols = list(cols)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "CovariatePreprocessor":
        x = df.reindex(columns=self.cols)
        x_imp = self.imputer.fit_transform(x)
        self.scaler.fit(x_imp)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df.reindex(columns=self.cols)
        x_imp = self.imputer.transform(x)
        return self.scaler.transform(x_imp).astype(np.float32)


class TargetScaler:
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.scaler.fit(y.reshape(-1, 1))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return self.scaler.transform(y.reshape(-1, 1)).astype(np.float32)

    def inverse_one(self, y_scaled: float) -> float:
        return float(self.scaler.inverse_transform(np.array([[y_scaled]], dtype=np.float32))[0, 0])


def make_hourly_sample_index(
    feat: pd.DataFrame,
    lookback_hours: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cov_cols: Sequence[str],
) -> Tuple[List[pd.Timestamp], List[np.ndarray], pd.DataFrame, np.ndarray]:
    """Return timestamps, history positions, current covariates and y for one-step training."""
    idx = pd.Index(feat.index)
    pos = {ts: i for i, ts in enumerate(idx)}
    timestamps = pd.date_range(pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize() + pd.Timedelta(hours=23), freq="h")

    out_ts: List[pd.Timestamp] = []
    out_pos: List[np.ndarray] = []
    cov_rows: List[Dict[str, float]] = []
    ys: List[float] = []

    for ts in timestamps:
        if ts not in pos:
            continue
        y = feat.at[ts, "recursive_target"]
        if pd.isna(y):
            continue
        hist_ts = pd.date_range(ts - pd.Timedelta(hours=lookback_hours), ts - pd.Timedelta(hours=1), freq="h")
        if len(hist_ts) != lookback_hours or any(h not in pos for h in hist_ts):
            continue
        out_ts.append(ts)
        out_pos.append(np.array([pos[h] for h in hist_ts], dtype=np.int64))
        cov_rows.append({col: feat.at[ts, col] if col in feat.columns else np.nan for col in cov_cols})
        ys.append(float(y))

    return out_ts, out_pos, pd.DataFrame(cov_rows).reindex(columns=list(cov_cols)), np.array(ys, dtype=float)


def split_train_val_by_last_days(
    timestamps: Sequence[pd.Timestamp],
    inner_validation_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dates = sorted({pd.Timestamp(ts).normalize() for ts in timestamps})
    val_dates = set(dates[-inner_validation_days:])
    is_val = np.array([pd.Timestamp(ts).normalize() in val_dates for ts in timestamps], dtype=bool)
    return np.where(~is_val)[0], np.where(is_val)[0]


def build_arrays(
    hist_matrix: np.ndarray,
    positions: Sequence[np.ndarray],
    cov_df: pd.DataFrame,
    y: np.ndarray,
    row_indices: np.ndarray,
    cov_prep: CovariatePreprocessor,
    target_scaler: TargetScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_seq = np.stack([hist_matrix[positions[i], :] for i in row_indices]).astype(np.float32)
    x_cov = cov_prep.transform(cov_df.iloc[row_indices]).astype(np.float32)
    y_scaled = target_scaler.transform(y[row_indices])
    return x_seq, x_cov, y_scaled


# =============================================================================
# 6. PYTORCH MODEL
# =============================================================================


class HourlyDataset(Dataset):
    def __init__(self, x_seq: np.ndarray, x_cov: np.ndarray, y: Optional[np.ndarray]) -> None:
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.x_cov = torch.as_tensor(x_cov, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_seq.shape[0]

    def __getitem__(self, i: int):
        if self.y is None:
            return self.x_seq[i], self.x_cov[i], torch.empty(0)
        return self.x_seq[i], self.x_cov[i], self.y[i]


class RecursiveLSTM(nn.Module):
    def __init__(self, input_size: int, cov_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=lstm_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + cov_size, max(hidden_size, 32)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size, 32), 1),
        )

    def forward(self, x_seq: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_seq)
        h = self.dropout(h_n[-1])
        return self.head(torch.cat([h, x_cov], dim=1))


@dataclass
class TrainingResult:
    model: RecursiveLSTM
    history: pd.DataFrame
    epochs_trained: int
    best_validation_loss: float
    fit_seconds: float


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.MSELoss()
    losses, maes = [], []
    with torch.no_grad():
        for x_seq, x_cov, y in loader:
            x_seq, x_cov, y = x_seq.to(device), x_cov.to(device), y.to(device)
            pred = model(x_seq, x_cov)
            losses.append(float(loss_fn(pred, y).item()))
            maes.append(float(torch.mean(torch.abs(pred - y)).item()))
    return float(np.mean(losses)), float(np.mean(maes))


def train_model(x_train_seq, x_train_cov, y_train, x_val_seq, x_val_cov, y_val, strategy: LSTMStrategy, cfg: Config, device: torch.device) -> TrainingResult:
    start = time.time()
    train_loader = DataLoader(HourlyDataset(x_train_seq, x_train_cov, y_train), batch_size=strategy.batch_size, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(HourlyDataset(x_val_seq, x_val_cov, y_val), batch_size=strategy.batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = RecursiveLSTM(
        input_size=x_train_seq.shape[2],
        cov_size=x_train_cov.shape[1],
        hidden_size=strategy.hidden_size,
        num_layers=strategy.num_layers,
        dropout=strategy.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=strategy.learning_rate, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_state = None
    best_loss = math.inf
    best_epoch = 0
    no_improve = 0
    rows = []

    for epoch in range(1, strategy.max_epochs + 1):
        model.train()
        train_losses = []
        for x_seq, x_cov, y in train_loader:
            x_seq, x_cov, y = x_seq.to(device), x_cov.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_seq, x_cov)
            loss = loss_fn(pred, y)
            loss.backward()
            if cfg.GRADIENT_CLIP_NORM and cfg.GRADIENT_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_mae = evaluate_loss(model, val_loader, device)
        rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss, "validation_mae_scaled": val_mae})

        if val_loss < best_loss - cfg.MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(model, pd.DataFrame(rows), int(best_epoch if best_epoch > 0 else len(rows)), float(best_loss), float(time.time() - start))


def predict_scaled_one(model: RecursiveLSTM, x_seq: np.ndarray, x_cov: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        xs = torch.as_tensor(x_seq[None, :, :], dtype=torch.float32).to(device)
        xc = torch.as_tensor(x_cov[None, :], dtype=torch.float32).to(device)
        pred = model(xs, xc)
    return float(pred.detach().cpu().numpy().reshape(-1)[0])


# =============================================================================
# 7. STRATEGIES, MODES AND PATHS
# =============================================================================


def lr_id(lr: float) -> str:
    return f"{lr:.0e}".replace("-", "m")


def make_default_strategies(cfg: Config) -> Dict[str, LSTMStrategy]:
    """Minimal full-validation grid for recursive rolling-24 LSTM.

    This is not an exhaustive hyperparameter search. It only compares one-week
    and two-week look-back windows under the same compact architecture.
    """
    strategies = [
        LSTMStrategy(
            strategy_id="lstm_rec24_168h_h64_l1_lr5em04_b128",
            lookback_hours=168,
            window_months=cfg.WINDOW_MONTHS,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            learning_rate=5e-4,
            batch_size=128,
            max_epochs=cfg.MAX_EPOCHS,
        ),
        LSTMStrategy(
            strategy_id="lstm_rec24_336h_h64_l1_lr5em04_b128",
            lookback_hours=336,
            window_months=cfg.WINDOW_MONTHS,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            learning_rate=5e-4,
            batch_size=128,
            max_epochs=cfg.MAX_EPOCHS,
        ),
    ]
    return {s.strategy_id: s for s in strategies}
def make_smoke_strategy(cfg: Config) -> LSTMStrategy:
    return LSTMStrategy("lstm_rec24_smoke_168h", 168, cfg.WINDOW_MONTHS, 16, 1, 0.0, 1e-3, 128, 2)


def fast_strategy(s: LSTMStrategy, cfg: Config) -> LSTMStrategy:
    return LSTMStrategy(s.strategy_id, s.lookback_hours, s.window_months, s.hidden_size, s.num_layers, s.dropout, s.learning_rate, s.batch_size, cfg.FAST_MAX_EPOCHS)


def resolve_modes(cfg: Config) -> List[str]:
    if cfg.RUN_SMOKE_TEST:
        return ["smoke"]
    modes = []
    if cfg.RUN_FAST_VALIDATION:
        modes.append("fast_validation")
    if cfg.RUN_FULL_VALIDATION:
        modes.append("full_validation")
    if cfg.RUN_FINAL_TEST:
        modes.append("final_test")
    if not modes:
        raise ValueError("No execution mode enabled.")
    return modes


def mode_split(mode: str) -> str:
    return "test" if mode == "final_test" else "validation"


def mode_targets_zones(cfg: Config, mode: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if mode == "smoke":
        return cfg.SMOKE_TARGETS, cfg.SMOKE_ZONES
    if mode == "fast_validation":
        return cfg.FAST_TARGETS, cfg.FAST_ZONES
    return cfg.TARGETS, cfg.ZONES


def mode_strategies(mode: str, cfg: Config, strategy_dict: Dict[str, LSTMStrategy], selected: Dict[str, str]) -> List[LSTMStrategy]:
    if mode == "smoke":
        return [make_smoke_strategy(cfg)]
    if mode == "fast_validation":
        return [fast_strategy(s, cfg) for s in strategy_dict.values()]
    if mode == "full_validation":
        return list(strategy_dict.values())
    if mode == "final_test":
        ids = sorted(set(selected.values()))
        missing = [sid for sid in ids if sid not in strategy_dict]
        if missing:
            raise ValueError(f"Selected strategies are not in grid: {missing}")
        return [strategy_dict[sid] for sid in ids]
    raise ValueError(f"Unsupported mode: {mode}")


def limit_eval_dates(eval_tz: pd.DataFrame, mode: str, cfg: Config) -> pd.DataFrame:
    out = eval_tz.sort_values(["forecast_date", "delivery_date", "hour"]).copy()
    if mode == "smoke":
        dates = out["forecast_date"].drop_duplicates().sort_values().head(cfg.SMOKE_MAX_FORECAST_DATES)
        out = out[out["forecast_date"].isin(dates)].copy()
    elif mode == "fast_validation" and cfg.FAST_MAX_FORECAST_DATES_PER_SERIES is not None:
        dates = out["forecast_date"].drop_duplicates().sort_values().tail(cfg.FAST_MAX_FORECAST_DATES_PER_SERIES)
        out = out[out["forecast_date"].isin(dates)].copy()
    return out


def recalibration_blocks(eval_tz: pd.DataFrame, cfg: Config) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
    if eval_tz.empty:
        return []
    temp = eval_tz.sort_values(["forecast_date", "delivery_date", "hour"]).copy()
    freq = cfg.RECALIBRATION_FREQUENCY.lower()
    if freq in {"none", "no", "fixed", "no_recalibration"}:
        return [(pd.Timestamp(temp["forecast_date"].min()).normalize(), temp)]
    if freq == "monthly":
        temp["forecast_month"] = temp["forecast_date"].dt.to_period("M")
        return [(pd.Timestamp(b["forecast_date"].min()).normalize(), b.drop(columns=["forecast_month"])) for _, b in temp.groupby("forecast_month", sort=True)]
    raise ValueError(f"Unsupported RECALIBRATION_FREQUENCY: {cfg.RECALIBRATION_FREQUENCY}")


def train_window(recalibration_date: pd.Timestamp, strategy: LSTMStrategy) -> Tuple[pd.Timestamp, pd.Timestamp]:
    train_end = pd.Timestamp(recalibration_date).normalize()
    train_start = train_end - pd.DateOffset(months=strategy.window_months) + pd.Timedelta(days=1)
    return pd.Timestamp(train_start).normalize(), train_end


def checkpoint_path(cfg: Config, split: str, sid: str, target: str, zone: str, recal: pd.Timestamp) -> Path:
    name = "__".join([safe_filename(split), safe_filename(sid), safe_filename(target), safe_filename(zone), pd.Timestamp(recal).strftime("%Y-%m-%d")]) + ".parquet"
    return cfg.EXPERIMENT_DIR / "checkpoints" / name


def curve_path(cfg: Config, split: str, sid: str, target: str, zone: str, recal: pd.Timestamp) -> Path:
    name = "__".join([safe_filename(split), safe_filename(sid), safe_filename(target), safe_filename(zone), pd.Timestamp(recal).strftime("%Y-%m-%d")]) + ".csv"
    return cfg.EXPERIMENT_DIR / "training_curves" / name


# =============================================================================
# 8. FORECASTING ROUTINES
# =============================================================================


def recursive_predict_block(
    model: RecursiveLSTM,
    feat: pd.DataFrame,
    eval_block: pd.DataFrame,
    seq_prep: SequencePreprocessor,
    cov_prep: CovariatePreprocessor,
    target_scaler: TargetScaler,
    strategy: LSTMStrategy,
    target: str,
    zone: str,
    split: str,
    model_name: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    recal: pd.Timestamp,
    train_result: TrainingResult,
    device: torch.device,
) -> Tuple[pd.DataFrame, float]:
    start = time.time()
    runtime = feat.copy()
    rows = []
    days = eval_block[["forecast_date", "delivery_date"]].drop_duplicates().sort_values("delivery_date")

    for _, drow in days.iterrows():
        forecast_date = pd.Timestamp(drow["forecast_date"]).normalize()
        delivery_date = pd.Timestamp(drow["delivery_date"]).normalize()
        day_hours = pd.date_range(delivery_date, delivery_date + pd.Timedelta(hours=23), freq="h")

        original_targets = runtime.loc[runtime.index.intersection(day_hours), "recursive_target"].copy()
        runtime.loc[runtime.index.intersection(day_hours), "recursive_target"] = np.nan

        for hour in range(1, 25):
            ts = delivery_date + pd.Timedelta(hours=hour - 1)
            hist_ts = pd.date_range(ts - pd.Timedelta(hours=strategy.lookback_hours), ts - pd.Timedelta(hours=1), freq="h")
            if any(h not in runtime.index for h in hist_ts) or ts not in runtime.index:
                raise ValueError(f"Missing recursive timestamps for {target}-{zone}, {delivery_date.date()} h={hour}")

            x_seq = seq_prep.transform(runtime.loc[hist_ts])
            x_cov = cov_prep.transform(pd.DataFrame([runtime.loc[ts, cov_prep.cols].to_dict()]))[0]
            pred_scaled = predict_scaled_one(model, x_seq, x_cov, device)
            pred = target_scaler.inverse_one(pred_scaled)
            runtime.loc[ts, "recursive_target"] = pred

            rows.append({
                "model": model_name,
                "target": target,
                "zone": zone,
                "split": split,
                "forecast_date": forecast_date,
                "delivery_date": delivery_date,
                "hour": hour,
                "horizon": hour,
                "y_pred": pred,
                "strategy_id": strategy.strategy_id,
                "lookback_hours": strategy.lookback_hours,
                "window_type": "rolling",
                "window_months": strategy.window_months,
                "train_start_date": train_start,
                "train_end_date": train_end,
                "recalibration_date": recal,
                "hidden_size": strategy.hidden_size,
                "num_layers": strategy.num_layers,
                "dropout": strategy.dropout,
                "learning_rate": strategy.learning_rate,
                "batch_size": strategy.batch_size,
                "epochs_trained": train_result.epochs_trained,
                "best_validation_loss": train_result.best_validation_loss,
                "fit_seconds": train_result.fit_seconds,
            })

        runtime.loc[original_targets.index, "recursive_target"] = original_targets

    pred = pd.DataFrame(rows)
    if pred.empty:
        return pred, float(time.time() - start)

    join_cols = ["target", "zone", "split", "forecast_date", "delivery_date", "hour", "horizon"]
    eval_join = eval_block[join_cols + ["delivery_datetime_model", "y_true"]].copy()
    pred["forecast_date"] = pd.to_datetime(pred["forecast_date"])
    pred["delivery_date"] = pd.to_datetime(pred["delivery_date"])
    eval_join["forecast_date"] = pd.to_datetime(eval_join["forecast_date"])
    eval_join["delivery_date"] = pd.to_datetime(eval_join["delivery_date"])
    pred = pred.merge(eval_join, on=join_cols, how="left", validate="one_to_one")
    if pred["delivery_datetime_model"].isna().any():
        raise ValueError("Prediction/eval_index alignment failed.")
    pred["predict_seconds"] = float(time.time() - start)

    ordered = [
        "model", "target", "zone", "split", "forecast_date", "delivery_datetime_model", "delivery_date", "hour", "horizon", "y_true", "y_pred",
        "strategy_id", "lookback_hours", "window_type", "window_months", "train_start_date", "train_end_date", "recalibration_date",
        "hidden_size", "num_layers", "dropout", "learning_rate", "batch_size", "epochs_trained", "best_validation_loss", "fit_seconds", "predict_seconds"
    ]
    return pred[ordered].sort_values(["target", "zone", "delivery_datetime_model"]), float(time.time() - start)


def run_one_block(
    feat: pd.DataFrame,
    cov_cols: List[str],
    eval_block: pd.DataFrame,
    target: str,
    zone: str,
    split: str,
    strategy: LSTMStrategy,
    recal: pd.Timestamp,
    model_name: str,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    start = time.time()
    log = {
        "split": split, "target": target, "zone": zone, "strategy_id": strategy.strategy_id,
        "recalibration_date": recal, "lookback_hours": strategy.lookback_hours,
        "window_months": strategy.window_months, "hidden_size": strategy.hidden_size,
        "num_layers": strategy.num_layers, "learning_rate": strategy.learning_rate,
        "batch_size": strategy.batch_size, "status": "started", "error_message": "",
    }

    ckpt = checkpoint_path(cfg, split, strategy.strategy_id, target, zone, recal)
    if cfg.SAVE_CHECKPOINTS and cfg.RESUME_FROM_CHECKPOINTS and ckpt.exists() and not cfg.OVERWRITE_EXISTING_CHECKPOINTS:
        pred = pd.read_parquet(ckpt)
        log.update({"status": "loaded_checkpoint", "n_predictions": len(pred), "total_seconds": 0.0})
        return pred, log

    try:
        tr_start, tr_end = train_window(recal, strategy)
        log["train_start_date"] = tr_start
        log["train_end_date"] = tr_end

        ts, positions, cov_df, y = make_hourly_sample_index(feat, strategy.lookback_hours, tr_start, tr_end, cov_cols)
        if len(ts) < cfg.MIN_TRAIN_SAMPLES:
            raise ValueError(f"Not enough training samples: {len(ts)}")
        train_idx, val_idx = split_train_val_by_last_days(ts, cfg.INNER_VALIDATION_DAYS)
        if len(train_idx) < cfg.MIN_INNER_TRAIN_SAMPLES or len(val_idx) < cfg.MIN_INNER_VALIDATION_SAMPLES:
            raise ValueError(f"Invalid inner split: train={len(train_idx)}, val={len(val_idx)}")
        log["n_train_samples"] = int(len(train_idx))
        log["n_inner_validation_samples"] = int(len(val_idx))

        hist_fit_start = tr_start - pd.Timedelta(hours=strategy.lookback_hours)
        hist_fit_end = tr_end + pd.Timedelta(hours=23)
        hist_fit = feat.loc[(feat.index >= hist_fit_start) & (feat.index <= hist_fit_end)]
        seq_prep = SequencePreprocessor(list(feat.columns)).fit(hist_fit)
        hist_matrix = seq_prep.transform(feat)
        cov_prep = CovariatePreprocessor(cov_cols).fit(cov_df.iloc[train_idx])
        target_scaler = TargetScaler().fit(y[train_idx])

        x_tr_seq, x_tr_cov, y_tr = build_arrays(hist_matrix, positions, cov_df, y, train_idx, cov_prep, target_scaler)
        x_va_seq, x_va_cov, y_va = build_arrays(hist_matrix, positions, cov_df, y, val_idx, cov_prep, target_scaler)

        result = train_model(x_tr_seq, x_tr_cov, y_tr, x_va_seq, x_va_cov, y_va, strategy, cfg, device)
        cpath = curve_path(cfg, split, strategy.strategy_id, target, zone, recal)
        ensure_dir(cpath.parent)
        result.history.assign(split=split, target=target, zone=zone, strategy_id=strategy.strategy_id, recalibration_date=recal).to_csv(cpath, index=False)

        pred, predict_seconds = recursive_predict_block(result.model, feat, eval_block, seq_prep, cov_prep, target_scaler, strategy, target, zone, split, model_name, tr_start, tr_end, recal, result, device)
        if cfg.SAVE_CHECKPOINTS:
            ensure_dir(ckpt.parent)
            pred.to_parquet(ckpt, index=False)

        log.update({
            "status": "ok", "epochs_trained": result.epochs_trained,
            "best_validation_loss": result.best_validation_loss, "fit_seconds": result.fit_seconds,
            "predict_seconds": predict_seconds, "total_seconds": time.time() - start,
            "n_predictions": len(pred), "n_features": hist_matrix.shape[1],
        })
        del hist_matrix, x_tr_seq, x_tr_cov, y_tr, x_va_seq, x_va_cov, y_va
        gc.collect()
        return pred, log

    except Exception as exc:
        log.update({"status": "error", "error_message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "total_seconds": time.time() - start})
        logger.error("Failed block | split=%s strategy=%s target=%s zone=%s recal=%s | %s", split, strategy.strategy_id, target, zone, recal, exc)
        return pd.DataFrame(), log


def run_strategy_target_zone(panel, eval_index, target, zone, split, mode, strategy, model_name, cfg, device, logger):
    logger.info("Running | mode=%s split=%s strategy=%s target=%s zone=%s", mode, split, strategy.strategy_id, target, zone)
    eval_tz = eval_index[(eval_index["target"] == target) & (eval_index["zone"] == zone) & (eval_index["split"] == split)].copy()
    eval_tz = limit_eval_dates(eval_tz, mode, cfg)
    if eval_tz.empty:
        logger.warning("No eval rows for split=%s target=%s zone=%s", split, target, zone)
        return pd.DataFrame(), pd.DataFrame()

    feat, cov_cols = build_recursive_feature_frame(panel, target, zone, cfg)
    pred_parts, logs = [], []
    for recal, block in recalibration_blocks(eval_tz, cfg):
        pred, log = run_one_block(feat, cov_cols, block, target, zone, split, strategy, recal, model_name, cfg, device, logger)
        if not pred.empty:
            pred_parts.append(pred)
        logs.append(log)

    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    logs_df = pd.DataFrame(logs)
    if split == "validation" and not pred_df.empty:
        out_dir = cfg.EXPERIMENT_DIR / "validation_candidate_predictions"
        ensure_dir(out_dir)
        pred_df.to_parquet(out_dir / f"pred_{safe_filename(strategy.strategy_id)}__{safe_filename(target)}__{safe_filename(zone)}.parquet", index=False)
    return pred_df, logs_df


# =============================================================================
# 9. METRICS, SELECTION AND QUALITY CHECKS
# =============================================================================


def compute_quick_metrics(pred: pd.DataFrame, naive: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    df = pred.copy()
    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
    df["squared_error"] = (df["y_true"] - df["y_pred"]) ** 2
    group_cols = ["model", "strategy_id", "target", "zone", "split"]
    metrics = df.groupby(group_cols, dropna=False).agg(
        n=("abs_error", "size"),
        MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
    ).reset_index()

    if naive is None or naive.empty:
        metrics["rMAE"] = np.nan
        return metrics
    naive_df = naive.copy()
    required = ["model", "target", "zone", "split", "delivery_datetime_model", "y_true", "y_pred"]
    if any(c not in naive_df.columns for c in required):
        metrics["rMAE"] = np.nan
        return metrics
    naive_df["delivery_datetime_model"] = normalize_datetime(naive_df["delivery_datetime_model"])
    naive_df = naive_df[naive_df["model"].astype(str) == "naive_week_before"].copy()
    naive_df["naive_abs_error"] = (naive_df["y_true"] - naive_df["y_pred"]).abs()
    joined = df.merge(
        naive_df[["target", "zone", "split", "delivery_datetime_model", "naive_abs_error"]],
        on=["target", "zone", "split", "delivery_datetime_model"],
        how="left",
    )
    rmae = joined.groupby(group_cols, dropna=False).agg(naive_MAE=("naive_abs_error", "mean")).reset_index()
    metrics = metrics.merge(rmae, on=group_cols, how="left")
    metrics["rMAE"] = metrics["MAE"] / metrics["naive_MAE"]
    return metrics


def select_best_by_target(metrics: pd.DataFrame, cfg: Config, logger: logging.Logger) -> Dict[str, str]:
    if metrics.empty or metrics[metrics["split"] == "validation"].empty:
        return cfg.SELECTED_STRATEGY_BY_TARGET.copy()
    val = metrics[metrics["split"] == "validation"].copy()
    score_col = "rMAE" if val["rMAE"].notna().any() else "MAE"
    scores = val.groupby(["target", "strategy_id"], dropna=False).agg(score=(score_col, "mean"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), n=("n", "sum")).reset_index().sort_values(["target", "score", "MAE"])
    selected = {str(target): str(g.iloc[0]["strategy_id"]) for target, g in scores.groupby("target", sort=True)}
    out_dir = cfg.EXPERIMENT_DIR / "selected_strategy"
    ensure_dir(out_dir)
    scores.to_csv(out_dir / "lstm_recursive_validation_strategy_results.csv", index=False)
    pd.DataFrame([{"target": k, "selected_strategy_id": v, "selection_metric": score_col} for k, v in selected.items()]).to_csv(out_dir / "lstm_recursive_selected_strategy_by_target.csv", index=False)
    logger.info("Selected recursive strategies by target using %s: %s", score_col, selected)
    return selected


def save_logs_metrics(logs: pd.DataFrame, pred: pd.DataFrame, naive: Optional[pd.DataFrame], cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    out_dir = cfg.EXPERIMENT_DIR / "logs"
    ensure_dir(out_dir)
    if not logs.empty:
        logs.to_csv(out_dir / "lstm_recursive_training_logs.csv", index=False)
        logger.info("Saved training logs: %s", out_dir / "lstm_recursive_training_logs.csv")
    metrics = compute_quick_metrics(pred, naive)
    if not metrics.empty:
        metrics.to_csv(out_dir / "lstm_recursive_quick_metrics.csv", index=False)
        logger.info("Saved quick metrics: %s", out_dir / "lstm_recursive_quick_metrics.csv")
    return metrics


def validate_prediction_output(pred: pd.DataFrame, cfg: Config) -> None:
    if pred.empty:
        raise ValueError("Prediction output is empty.")
    missing = [c for c in cfg.REQUIRED_PREDICTION_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"Prediction output missing columns: {missing}")
    if pred["y_pred"].isna().any():
        raise ValueError(f"Prediction output contains {int(pred['y_pred'].isna().sum())} missing y_pred values.")
    dup = pred.duplicated(["model", "target", "zone", "delivery_datetime_model"]).sum()
    if dup > 0:
        raise ValueError(f"Prediction output contains {dup} duplicated model-target-zone-datetime rows.")
    bad_h = sorted(set(pred["horizon"].dropna().astype(int)) - set(range(1, 25)))
    if bad_h:
        raise ValueError(f"Unexpected horizons: {bad_h}")
    counts = pred.groupby(["model", "target", "zone", "delivery_date"])["hour"].nunique().reset_index(name="n_hours")
    incomplete = counts[counts["n_hours"] != 24]
    if not incomplete.empty:
        raise ValueError(f"Incomplete 24h daily profiles. First rows: {incomplete.head().to_dict(orient='records')}")


def print_quality_summary(pred: pd.DataFrame, logger: logging.Logger) -> None:
    logger.info("================ RECURSIVE LSTM PREDICTION QUALITY SUMMARY ================")
    logger.info("Models: %s", sorted(pred["model"].astype(str).unique().tolist()))
    counts = pred.groupby(["model", "split", "target", "zone"]).size().reset_index(name="n_predictions").sort_values(["model", "split", "target", "zone"])
    logger.info("Prediction counts by model/split/target/zone:\n%s", counts.to_string(index=False))
    missing = pred.groupby(["model", "split", "target"])["y_pred"].apply(lambda x: int(x.isna().sum()))
    logger.info("Missing y_pred counts:\n%s", missing.to_string())
    logger.info("Horizon distribution:\n%s", pred["horizon"].value_counts().sort_index().to_string())


# =============================================================================
# 10. MAIN
# =============================================================================


def run_mode(mode: str, panel: pd.DataFrame, eval_index: pd.DataFrame, cfg: Config, strategy_dict: Dict[str, LSTMStrategy], selected: Dict[str, str], device: torch.device, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = mode_split(mode)
    targets, zones = mode_targets_zones(cfg, mode)
    strategies = mode_strategies(mode, cfg, strategy_dict, selected)
    pred_parts, log_parts = [], []
    for strategy in strategies:
        for target in targets:
            for zone in zones:
                if mode == "final_test" and strategy.strategy_id != selected.get(target, cfg.SELECTED_STRATEGY_BY_TARGET.get(target)):
                    continue
                model_name = cfg.MODEL_NAME_SMOKE if mode == "smoke" else (cfg.MODEL_NAME_FINAL if mode == "final_test" else f"lstm_recursive__{strategy.strategy_id}")
                pred, logs = run_strategy_target_zone(panel, eval_index, target, zone, split, mode, strategy, model_name, cfg, device, logger)
                if not pred.empty:
                    pred_parts.append(pred)
                if not logs.empty:
                    log_parts.append(logs)
                gc.collect()
    pred_out = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    logs_out = pd.concat(log_parts, ignore_index=True) if log_parts else pd.DataFrame()
    return pred_out, logs_out


def main() -> None:
    cfg = Config()
    for folder in [
        cfg.PREDICTIONS_DIR,
        cfg.EXPERIMENT_DIR,
        cfg.EXPERIMENT_DIR / "logs",
        cfg.EXPERIMENT_DIR / "checkpoints",
        cfg.EXPERIMENT_DIR / "training_curves",
        cfg.EXPERIMENT_DIR / "validation_candidate_predictions",
        cfg.EXPERIMENT_DIR / "selected_strategy",
    ]:
        ensure_dir(folder)

    logger = setup_logging(cfg)
    set_global_seed(cfg.RANDOM_SEED)
    device = get_device(cfg)

    logger.info("Starting 11_lstm_recursive_rolling24_model.py")
    logger.info("Project root: %s", cfg.PROJECT_ROOT)
    logger.info("Device: %s", device)
    logger.info("Enabled modes: %s", resolve_modes(cfg))

    with open(cfg.EXPERIMENT_DIR / "logs/lstm_recursive_config.json", "w", encoding="utf-8") as f:
        json.dump(serialize_config(cfg), f, indent=2, default=str)

    panel = prepare_panel(load_table(cfg.PANEL_PARQUET_PATH, cfg.PANEL_RDS_PATH, "processed panel"), cfg)
    eval_index = prepare_eval_index(load_table(cfg.EVAL_INDEX_PARQUET_PATH, cfg.EVAL_INDEX_RDS_PATH, "evaluation index"))
    naive = load_weekly_naive(cfg, logger)

    strategy_dict = make_default_strategies(cfg)
    selected = cfg.SELECTED_STRATEGY_BY_TARGET.copy()
    all_preds, all_logs = [], []

    for mode in resolve_modes(cfg):
        pred_mode, logs_mode = run_mode(mode, panel, eval_index, cfg, strategy_dict, selected, device, logger)
        if not pred_mode.empty:
            all_preds.append(pred_mode)
        if not logs_mode.empty:
            all_logs.append(logs_mode)
        if mode in {"fast_validation", "full_validation"} and not pred_mode.empty:
            current_preds = pd.concat(all_preds, ignore_index=True)
            current_logs = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
            current_metrics = save_logs_metrics(current_logs, current_preds, naive, cfg, logger)
            selected = select_best_by_target(current_metrics, cfg, logger)

    final_pred = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    final_logs = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    if final_pred.empty:
        logger.warning("No predictions generated. Check logs.")
        if not final_logs.empty:
            save_logs_metrics(final_logs, final_pred, naive, cfg, logger)
        return

    for col in ["forecast_date", "delivery_date", "train_start_date", "train_end_date", "recalibration_date"]:
        if col in final_pred.columns:
            final_pred[col] = pd.to_datetime(final_pred[col], errors="coerce")
    if "delivery_datetime_model" in final_pred.columns:
        final_pred["delivery_datetime_model"] = normalize_datetime(final_pred["delivery_datetime_model"])

    validate_prediction_output(final_pred, cfg)
    print_quality_summary(final_pred, logger)
    save_logs_metrics(final_logs, final_pred, naive, cfg, logger)
    save_outputs(final_pred, cfg.FINAL_PREDICTION_PARQUET, cfg.FINAL_PREDICTION_RDS, logger)

    logger.info("Finished successfully.")
    logger.info("Main prediction output: %s", cfg.FINAL_PREDICTION_PARQUET)


if __name__ == "__main__":
    main()
