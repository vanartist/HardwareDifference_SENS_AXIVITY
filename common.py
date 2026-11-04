"""Data preparation, training and evaluation shared by the model scripts."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import secrets
import shutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from experiment_config import (
    AUGMENTATION_BASE_PROBABILITY,
    AUGMENTATION_JITTER_STD,
    AUGMENTATION_MASK_RATIO_CAP,
    AUGMENTATION_MAXIMUM_CLASS_BOOST,
    AUGMENTATION_MAXIMUM_MASK_PROBABILITY,
    AUGMENTATION_MAXIMUM_PROBABILITY,
    AUGMENTATION_MAX_MASK_RATIO,
    AUGMENTATION_SCALE_STD,
    AUGMENTATION_STD_BOOST_CAP,
    AUGMENTATION_TIME_MASK_PROBABILITY,
    CACHE_DIR,
    CLASS_ORDER,
    CLASS_MAP_PATH,
    CLASS_WEIGHT_BETA,
    CLASS_WEIGHT_MAXIMUM,
    CLASS_WEIGHT_MINIMUM,
    DETERMINISTIC,
    DEVICE,
    EPOCHS,
    FILE_PATTERN,
    MAX_TRIALS,
    MODEL_DISPLAY,
    MODEL_FOLDER_NAME,
    NUM_WORKERS,
    OTHER_DEVICE,
    PATIENCE,
    RESULT_FILES,
    SAMPLE_RATE_HZ,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SEARCH_VALIDATION_FRACTION,
    SOURCE_DEVICES,
    SPLIT_MODE,
    TEST_DATA_DIR,
    TEMPORAL_OCCLUSION_SEGMENTS,
    TRAIN_DATA_DIR,
    USE_CACHE,
    VALIDATION_FRACTION,
)


# Keep this legacy order: it is baked into the saved reports and model outputs.
ENCODED_CLASS_ORDER = ("Lie", "Run", "Sit", "Stairs", "Stand", "Walk")

DEFAULT_TRAIN_DIR = TRAIN_DATA_DIR
DEFAULT_TEST_DIR = TEST_DATA_DIR
DEFAULT_CLASS_MAP = CLASS_MAP_PATH
DEFAULT_CACHE_DIR = CACHE_DIR
DEFAULT_FILE_PATTERN = FILE_PATTERN


ModelFactory = Callable[[int, Mapping[str, Any]], nn.Module]


# Configuration and result containers


def normalise_purity(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"pure", "true", "drop_impure_true"}:
        return "pure"
    if text in {"mixed", "impure", "false", "drop_impure_false"}:
        return "mixed"
    raise ValueError(
        f"Unsupported purity value: {value!r}. Use 'pure' or 'mixed'."
    )


def serialisable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [serialisable(item) for item in value]
    if isinstance(value, list):
        return [serialisable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialisable(item) for key, item in value.items()}
    return value


@dataclass
class ExperimentConfig:
    """Configuration for one model, one window, and one purity condition."""

    model_name: str
    window_seconds: int
    purity: str
    results_root: Path
    run_name: str
    train_dir: Path = field(default_factory=lambda: DEFAULT_TRAIN_DIR)
    test_dir: Path = field(default_factory=lambda: DEFAULT_TEST_DIR)
    class_map_path: Path = field(default_factory=lambda: DEFAULT_CLASS_MAP)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    file_pattern: str = DEFAULT_FILE_PATTERN
    sample_rate_hz: int = SAMPLE_RATE_HZ
    seed: int | None = None
    epochs: int = EPOCHS
    patience: int = PATIENCE
    max_trials: int = MAX_TRIALS
    validation_fraction: float = VALIDATION_FRACTION
    search_validation_fraction: float = SEARCH_VALIDATION_FRACTION
    split_mode: str = SPLIT_MODE
    device_request: str = DEVICE
    deterministic: bool = DETERMINISTIC
    num_workers: int = NUM_WORKERS
    resume: bool = False
    overwrite: bool = False
    use_cache: bool = USE_CACHE

    def __post_init__(self) -> None:
        self.model_name = str(self.model_name).strip().lower()
        self.purity = normalise_purity(self.purity)
        self.results_root = Path(self.results_root)
        self.train_dir = Path(self.train_dir)
        self.test_dir = Path(self.test_dir)
        self.class_map_path = Path(self.class_map_path)
        self.cache_dir = Path(self.cache_dir)
        self.seed = (
            secrets.randbits(32) if self.seed is None else int(self.seed)
        )
        if self.model_name not in MODEL_DISPLAY:
            raise ValueError(f"Unknown model: {self.model_name}")
        if self.window_seconds < 1:
            raise ValueError("window_seconds must be positive")
        if self.sample_rate_hz < 1:
            raise ValueError("sample_rate_hz must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if not 0.0 < self.search_validation_fraction < 0.5:
            raise ValueError(
                "search_validation_fraction must be between 0 and 0.5"
            )
        if self.split_mode not in {"participant", "window"}:
            raise ValueError("split_mode must be 'participant' or 'window'")
        if self.resume and self.overwrite:
            raise ValueError("--resume and --overwrite cannot be used together")

    @property
    def drop_impure(self) -> bool:
        return self.purity == "pure"

    @property
    def purity_label(self) -> str:
        return "Pure" if self.drop_impure else "Mixed"

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.run_name

    def result_folder_name(self, source_device: str) -> str:
        return (
            f"{MODEL_FOLDER_NAME[self.model_name]}_"
            f"{self.window_seconds}_drop_impure_"
            f"{self.drop_impure}_{source_device}"
        )

    def output_dir(self, experiment_type: str, source_device: str) -> Path:
        return (
            self.run_dir
            / experiment_type
            / self.result_folder_name(source_device)
        )


@dataclass
class DatasetBundle:
    """Paired SENS/Axivity windows from the training and held-out test sets."""

    train_x: dict[str, np.ndarray]
    test_x: dict[str, np.ndarray]
    train_y: np.ndarray
    test_y: np.ndarray
    train_pid: np.ndarray
    test_pid: np.ndarray
    train_time: np.ndarray
    test_time: np.ndarray
    cache_path: Path

    def validate(self) -> None:
        for split_name, arrays, labels, pids in (
            ("train", self.train_x, self.train_y, self.train_pid),
            ("test", self.test_x, self.test_y, self.test_pid),
        ):
            missing_devices = set(SOURCE_DEVICES) - set(arrays)
            if missing_devices:
                raise ValueError(
                    f"{split_name} data is missing devices: {missing_devices}"
                )
            lengths = {device: len(arrays[device]) for device in SOURCE_DEVICES}
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"{split_name} paired-device lengths differ: {lengths}"
                )
            expected = next(iter(lengths.values()))
            if len(labels) != expected or len(pids) != expected:
                raise ValueError(
                    f"{split_name} labels/PIDs do not match window count"
                )
            for device, array in arrays.items():
                if array.ndim != 3 or array.shape[1] != 3:
                    raise ValueError(
                        f"{split_name}/{device} must have shape (N, 3, T); "
                        f"received {array.shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"{split_name}/{device} contains non-finite values"
                    )


@dataclass
class EvaluationResult:
    loss: float
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    pids: np.ndarray


@dataclass
class FitResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_validation_macro_f1: float
    stopped_early: bool


class AccelerometerDataset(Dataset):
    """Tensor dataset with the original accelerometer augmentations."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pid: np.ndarray,
        augment: bool = False,
        jitter_std: float = AUGMENTATION_JITTER_STD,
        scale_std: float = AUGMENTATION_SCALE_STD,
        time_mask_probability: float = AUGMENTATION_TIME_MASK_PROBABILITY,
        max_mask_ratio: float = AUGMENTATION_MAX_MASK_RATIO,
        class_boost_map: Mapping[int, float] | None = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.pid = np.asarray(pid)
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_std = scale_std
        self.time_mask_probability = time_mask_probability
        self.max_mask_ratio = max_mask_ratio
        self.class_boost_map = dict(class_boost_map or {})

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.x[index].clone()
        label = self.y[index]

        if self.augment:
            class_index = int(label.item())
            boost = float(self.class_boost_map.get(class_index, 1.0))
            jitter_probability = min(
                AUGMENTATION_BASE_PROBABILITY * boost,
                AUGMENTATION_MAXIMUM_PROBABILITY,
            )
            scale_probability = min(
                AUGMENTATION_BASE_PROBABILITY * boost,
                AUGMENTATION_MAXIMUM_PROBABILITY,
            )
            mask_probability = min(
                self.time_mask_probability * boost,
                AUGMENTATION_MAXIMUM_MASK_PROBABILITY,
            )
            jitter_std = self.jitter_std * min(
                math.sqrt(boost), AUGMENTATION_STD_BOOST_CAP
            )
            scale_std = self.scale_std * min(
                math.sqrt(boost), AUGMENTATION_STD_BOOST_CAP
            )
            max_mask_ratio = min(
                self.max_mask_ratio * max(1.0, boost),
                AUGMENTATION_MASK_RATIO_CAP,
            )

            if np.random.random() < jitter_probability:
                sample = sample + torch.randn_like(sample) * jitter_std

            if np.random.random() < scale_probability:
                scale = torch.normal(
                    mean=torch.ones((sample.shape[0], 1)),
                    std=scale_std,
                ).float()
                sample = sample * scale

            if np.random.random() < mask_probability:
                time_steps = sample.shape[1]
                max_mask_length = max(1, int(time_steps * max_mask_ratio))
                mask_length = np.random.randint(1, max_mask_length + 1)
                start = np.random.randint(0, time_steps - mask_length + 1)
                sample[:, start : start + mask_length] = 0.0

        return sample, label, str(self.pid[index])


class FocalCrossEntropyLoss(nn.Module):
    """Cross-entropy with focal scaling and optional label smoothing."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probability = torch.exp(-cross_entropy)
        focal_term = (1.0 - probability).clamp(min=1e-6).pow(self.gamma)
        return (focal_term * cross_entropy).mean()


# Reproducibility and device selection


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            deterministic, warn_only=True
        )


def resolve_torch_device(request: str) -> torch.device:
    text = str(request).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA was requested ({request}) but is not available"
            )
        return torch.device(request)
    raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")


def cache_path_for(config: ExperimentConfig) -> Path:
    tie_policy = "" if config.drop_impure else "_modefirst"
    return config.cache_dir / (
        "windowed_accelerometer_dataset_"
        f"win{config.window_seconds}_dropimpure{config.drop_impure}"
        f"{tie_policy}.npz"
    )


# Data loading and windowing


def load_device_dataframe(
    data_path: Path,
    device_name: str,
    class_map: Mapping[str, str],
) -> pd.DataFrame:
    x_column = f"{device_name}_x"
    y_column = f"{device_name}_y"
    z_column = f"{device_name}_z"
    dataframe = pd.read_csv(
        data_path,
        index_col="global_time",
        parse_dates=["global_time"],
        usecols=[
            "global_time",
            x_column,
            y_column,
            z_column,
            "posture_movement",
        ],
        dtype={
            x_column: "f4",
            y_column: "f4",
            z_column: "f4",
            "posture_movement": "string",
        },
        keep_default_na=False,
        na_values={
            x_column: ["NA"],
            y_column: ["NA"],
            z_column: ["NA"],
        },
        encoding="utf-8",
    )
    dataframe.columns = ["x", "y", "z", "annotation"]
    dataframe["annotation"] = dataframe["annotation"].map(class_map)
    dataframe["pid"] = data_path.parent.name
    dataframe = dataframe[
        dataframe["annotation"].isin([*CLASS_ORDER, "Other"])
    ]
    return dataframe


def paired_windows(
    sens: pd.DataFrame,
    axivity: pd.DataFrame,
    window_seconds: int,
    sample_rate_hz: int,
    drop_impure: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synchronized non-overlapping windows for both devices."""

    left = sens[["x", "y", "z", "annotation", "pid"]].rename(
        columns={
            "x": "sens_x",
            "y": "sens_y",
            "z": "sens_z",
            "annotation": "sens_annotation",
            "pid": "sens_pid",
        }
    )
    right = axivity[["x", "y", "z", "annotation", "pid"]].rename(
        columns={
            "x": "axivity_x",
            "y": "axivity_y",
            "z": "axivity_z",
            "annotation": "axivity_annotation",
            "pid": "axivity_pid",
        }
    )
    merged = left.join(right, how="inner")
    if merged.empty:
        raise ValueError("No overlapping SENS/Axivity timestamps were found")

    same_label = (
        merged["sens_annotation"] == merged["axivity_annotation"]
    )
    merged = merged[same_label]
    if merged.empty:
        raise ValueError("No data remained after label alignment")

    expected_length = sample_rate_hz * window_seconds
    sens_windows: list[np.ndarray] = []
    axivity_windows: list[np.ndarray] = []
    labels: list[str] = []
    timestamps: list[np.datetime64] = []
    participant_ids: list[str] = []

    groups = merged.groupby(
        pd.Grouper(
            freq=f"{window_seconds}s",
            origin="start",
            closed="left",
            dropna=False,
        )
    )
    for timestamp, window in groups:
        if len(window) != expected_length:
            continue
        sens_values = window[
            ["sens_x", "sens_y", "sens_z"]
        ].to_numpy(dtype=np.float32)
        axivity_values = window[
            ["axivity_x", "axivity_y", "axivity_z"]
        ].to_numpy(dtype=np.float32)
        if (
            not np.isfinite(sens_values).all()
            or not np.isfinite(axivity_values).all()
        ):
            continue

        label_series = window["sens_annotation"]
        if drop_impure:
            if label_series.nunique(dropna=False) != 1:
                continue
            label = str(label_series.iloc[0])
        else:
            modes = label_series.mode(dropna=False)
            if modes.empty:
                continue
            # Match the original notebooks: keep tied modes and take the
            # first value returned by pandas.
            label = str(modes.iloc[0])
        if label not in CLASS_ORDER:
            continue

        sens_pids = window["sens_pid"].dropna().unique()
        axivity_pids = window["axivity_pid"].dropna().unique()
        if (
            len(sens_pids) != 1
            or len(axivity_pids) != 1
            or sens_pids[0] != axivity_pids[0]
        ):
            raise ValueError(
                f"Inconsistent participant ID at window {timestamp}"
            )

        sens_windows.append(sens_values)
        axivity_windows.append(axivity_values)
        labels.append(label)
        timestamps.append(timestamp.to_datetime64())
        participant_ids.append(str(sens_pids[0]))

    if not labels:
        raise ValueError("No valid paired windows were created")
    return (
        np.stack(sens_windows),
        np.stack(axivity_windows),
        np.asarray(labels),
        np.asarray(timestamps),
        np.asarray(participant_ids),
    )


def load_split_from_raw(
    root: Path,
    file_pattern: str,
    class_map: Mapping[str, str],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(root.glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched {file_pattern!r} under {root}"
        )

    sens_arrays: list[np.ndarray] = []
    axivity_arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    times: list[np.ndarray] = []
    participant_ids: list[np.ndarray] = []
    for data_path in tqdm(files, desc=f"Windowing {root.name}"):
        sens_frame = load_device_dataframe(data_path, "sens", class_map)
        axivity_frame = load_device_dataframe(
            data_path, "axivity", class_map
        )
        sens, axivity, y, timestamps, pids = paired_windows(
            sens=sens_frame,
            axivity=axivity_frame,
            window_seconds=config.window_seconds,
            sample_rate_hz=config.sample_rate_hz,
            drop_impure=config.drop_impure,
        )
        sens_arrays.append(sens)
        axivity_arrays.append(axivity)
        labels.append(y)
        times.append(timestamps)
        participant_ids.append(pids)

    return (
        np.concatenate(sens_arrays),
        np.concatenate(axivity_arrays),
        np.concatenate(labels),
        np.concatenate(times),
        np.concatenate(participant_ids),
    )


def build_dataset_cache(
    config: ExperimentConfig, cache_path: Path
) -> DatasetBundle:
    """Build the paired dataset from raw CSV files and save the NPZ cache."""

    if not config.train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {config.train_dir}")
    if not config.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {config.test_dir}")
    if not config.class_map_path.exists():
        raise FileNotFoundError(
            f"Class-map JSON not found: {config.class_map_path}"
        )
    class_map = json.loads(
        config.class_map_path.read_text(encoding="utf-8")
    )

    train_sens, train_axivity, train_y, train_time, train_pid = (
        load_split_from_raw(
            config.train_dir, config.file_pattern, class_map, config
        )
    )
    test_sens, test_axivity, test_y, test_time, test_pid = (
        load_split_from_raw(
            config.test_dir, config.file_pattern, class_map, config
        )
    )

    # PyTorch Conv1d expects (samples, channels, time).
    train_sens = train_sens.transpose(0, 2, 1)
    train_axivity = train_axivity.transpose(0, 2, 1)
    test_sens = test_sens.transpose(0, 2, 1)
    test_axivity = test_axivity.transpose(0, 2, 1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train_1=train_sens,
        X_train_2=train_axivity,
        y_train=train_y,
        T_train=train_time,
        pid_train=train_pid,
        X_test_1=test_sens,
        X_test_2=test_axivity,
        y_test=test_y,
        T_test=test_time,
        pid_test=test_pid,
    )
    return DatasetBundle(
        train_x={"sens": train_sens, "axivity": train_axivity},
        test_x={"sens": test_sens, "axivity": test_axivity},
        train_y=train_y,
        test_y=test_y,
        train_pid=train_pid,
        test_pid=test_pid,
        train_time=train_time,
        test_time=test_time,
        cache_path=cache_path,
    )


def load_dataset_bundle(config: ExperimentConfig) -> DatasetBundle:
    """Load a compatible cache or construct it from the paired raw data."""

    cache_path = cache_path_for(config)
    if config.use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        required = {
            "X_train_1",
            "X_train_2",
            "y_train",
            "pid_train",
            "X_test_1",
            "X_test_2",
            "y_test",
            "pid_test",
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(
                f"Cache is missing arrays {sorted(missing)}: {cache_path}"
            )
        train_time = (
            cached["T_train"]
            if "T_train" in cached.files
            else np.arange(len(cached["y_train"]))
        )
        test_time = (
            cached["T_test"]
            if "T_test" in cached.files
            else np.arange(len(cached["y_test"]))
        )
        bundle = DatasetBundle(
            train_x={
                "sens": np.asarray(cached["X_train_1"]),
                "axivity": np.asarray(cached["X_train_2"]),
            },
            test_x={
                "sens": np.asarray(cached["X_test_1"]),
                "axivity": np.asarray(cached["X_test_2"]),
            },
            train_y=np.asarray(cached["y_train"]).astype(str),
            test_y=np.asarray(cached["y_test"]).astype(str),
            train_pid=np.asarray(cached["pid_train"]).astype(str),
            test_pid=np.asarray(cached["pid_test"]).astype(str),
            train_time=np.asarray(train_time),
            test_time=np.asarray(test_time),
            cache_path=cache_path,
        )
    else:
        bundle = build_dataset_cache(config, cache_path)
    bundle.validate()
    return bundle


def encode_labels(labels: Iterable[str]) -> np.ndarray:
    mapping = {
        label: index for index, label in enumerate(ENCODED_CLASS_ORDER)
    }
    encoded = []
    for label in labels:
        text = str(label)
        if text not in mapping:
            raise ValueError(f"Unexpected label: {text!r}")
        encoded.append(mapping[text])
    return np.asarray(encoded, dtype=np.int64)


def compute_channel_stats(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=True)
    standard_deviation = x.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, standard_deviation


def apply_zscore(
    x: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray
) -> np.ndarray:
    return ((x - mean) / standard_deviation).astype(np.float32)


def compute_effective_number_weights(
    y: np.ndarray,
    num_classes: int,
    beta: float = CLASS_WEIGHT_BETA,
    minimum: float = CLASS_WEIGHT_MINIMUM,
    maximum: float = CLASS_WEIGHT_MAXIMUM,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    effective_number = 1.0 - np.power(beta, counts)
    effective_number = np.where(
        effective_number == 0.0, 1.0, effective_number
    )
    weights = (1.0 - beta) / effective_number
    weights = weights / weights.mean()
    weights = np.clip(weights, minimum, maximum)
    weights = weights / weights.mean()
    return weights, counts


def build_class_boost_map(
    y: np.ndarray,
    num_classes: int,
    maximum_boost: float = AUGMENTATION_MAXIMUM_CLASS_BOOST,
) -> dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    positive_counts = counts[counts > 0]
    median_count = np.median(positive_counts) if len(positive_counts) else 1.0
    ratios = np.sqrt(median_count / np.maximum(counts, 1.0))
    ratios = np.clip(ratios, 1.0, maximum_boost)
    return {index: float(ratios[index]) for index in range(num_classes)}


# Splitting, training, and evaluation


def split_indices(
    y: np.ndarray,
    participant_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split training data and prefer participant-disjoint validation."""

    indices = np.arange(len(y))
    metadata: dict[str, Any] = {
        "requested_split_mode": split_mode,
        "fallback_used": False,
    }
    if split_mode == "participant":
        groups = np.asarray(participant_ids).astype(str)
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 3:
            requested_splits = max(2, int(round(1.0 / validation_fraction)))
            n_splits = min(requested_splits, len(unique_groups))
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=seed,
                )
                candidates = []
                all_classes = set(np.unique(y).tolist())
                for train_index, validation_index in splitter.split(
                    indices, y, groups
                ):
                    validation_classes = set(
                        np.unique(y[validation_index]).tolist()
                    )
                    missing_class_penalty = len(
                        all_classes - validation_classes
                    )
                    size_error = abs(
                        len(validation_index) / len(indices)
                        - validation_fraction
                    )
                    candidates.append(
                        (
                            missing_class_penalty,
                            size_error,
                            train_index,
                            validation_index,
                        )
                    )
                _, _, train_index, validation_index = min(
                    candidates, key=lambda item: (item[0], item[1])
                )
                overlap = set(groups[train_index]) & set(
                    groups[validation_index]
                )
                metadata.update(
                    {
                        "actual_split_mode": "participant",
                        "train_participants": int(
                            len(np.unique(groups[train_index]))
                        ),
                        "validation_participants": int(
                            len(np.unique(groups[validation_index]))
                        ),
                        "participant_overlap_count": int(len(overlap)),
                    }
                )
                return train_index, validation_index, metadata
            except ValueError as error:
                metadata["participant_split_error"] = str(error)

        metadata["fallback_used"] = True
        metadata["fallback_reason"] = (
            "Participant-disjoint stratification was not feasible."
        )
        warnings.warn(
            "Participant-disjoint validation was not feasible; falling back "
            "to the legacy stratified window split.",
            RuntimeWarning,
        )

    train_index, validation_index = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y,
    )
    train_groups = set(np.asarray(participant_ids)[train_index].astype(str))
    validation_groups = set(
        np.asarray(participant_ids)[validation_index].astype(str)
    )
    metadata.update(
        {
            "actual_split_mode": "window",
            "train_participants": int(len(train_groups)),
            "validation_participants": int(len(validation_groups)),
            "participant_overlap_count": int(
                len(train_groups & validation_groups)
            ),
        }
    )
    return train_index, validation_index, metadata


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    batch_size: int,
    augment: bool,
    class_boost_map: Mapping[int, float] | None,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = AccelerometerDataset(
        x=x,
        y=y,
        pid=participant_ids,
        augment=augment,
        class_boost_map=class_boost_map,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.train()
    running_loss = 0.0
    true_values: list[int] = []
    predictions: list[int] = []

    for features, labels, _ in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += float(loss.item()) * len(features)
        predicted = torch.argmax(logits, dim=1)
        true_values.extend(labels.detach().cpu().tolist())
        predictions.extend(predicted.detach().cpu().tolist())

    return (
        running_loss / len(loader.dataset),
        compute_metrics(true_values, predictions),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    running_loss = 0.0
    true_values: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    participant_ids: list[str] = []

    for features, labels, pids in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)
        probability = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probability, dim=1)

        running_loss += float(loss.item()) * len(features)
        true_values.extend(labels.detach().cpu().tolist())
        predictions.extend(predicted.detach().cpu().tolist())
        probabilities.extend(probability.detach().cpu().numpy())
        participant_ids.extend(str(pid) for pid in pids)

    y_true = np.asarray(true_values, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    return EvaluationResult(
        loss=running_loss / len(loader.dataset),
        metrics=compute_metrics(y_true, y_pred),
        y_true=y_true,
        y_pred=y_pred,
        probabilities=np.asarray(probabilities),
        pids=np.asarray(participant_ids),
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    class_weights: np.ndarray,
    focal_gamma: float,
    label_smoothing: float,
) -> FitResult:
    """Fit with early stopping on validation macro-F1."""

    weight_tensor = torch.as_tensor(
        class_weights, dtype=torch.float32, device=device
    )
    criterion = FocalCrossEntropyLoss(
        weight=weight_tensor,
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    history_rows: list[dict[str, float | int]] = []
    best_weights = copy.deepcopy(model.state_dict())
    best_validation_macro_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation = evaluate_model(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation.metrics["macro_f1"])
        learning_rate_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "learning_rate": float(learning_rate_now),
            "train_loss": float(train_loss),
            "validation_loss": float(validation.loss),
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": validation.metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_macro_f1": validation.metrics["macro_f1"],
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_loss:.4f}, "
            f"train macro-F1={train_metrics['macro_f1']:.4f} | "
            f"validation loss={validation.loss:.4f}, "
            f"validation macro-F1={validation.metrics['macro_f1']:.4f}"
        )

        if validation.metrics["macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = validation.metrics["macro_f1"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_early = True
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    history = pd.DataFrame(history_rows)
    return FitResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_macro_f1=float(best_validation_macro_f1),
        stopped_early=stopped_early,
    )


def count_parameters(model: nn.Module) -> int:
    return int(
        sum(parameter.numel() for parameter in model.parameters())
    )


def select_trial_space(
    search_space: Sequence[Mapping[str, Any]],
    max_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [dict(parameters) for parameters in search_space]
    if max_trials >= len(candidates):
        return candidates
    generator = random.Random(seed)
    selected_indices = sorted(
        generator.sample(range(len(candidates)), max_trials)
    )
    return [candidates[index] for index in selected_indices]


def run_hyperparameter_search(
    model_factory: ModelFactory,
    search_space: Sequence[Mapping[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    pid_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    pid_validation: np.ndarray,
    num_classes: int,
    class_weights: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    seed_offset: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the two search configurations carried over from the notebooks."""

    candidates = select_trial_space(
        search_space, config.max_trials, config.seed + seed_offset
    )
    class_boost_map = build_class_boost_map(
        y_train, num_classes=num_classes
    )
    trial_records: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    pin_memory = device.type == "cuda"

    for trial_index, parameters in enumerate(candidates, start=1):
        trial_seed = config.seed + seed_offset + trial_index
        seed_everything(trial_seed, config.deterministic)
        batch_size = int(parameters["batch_size"])
        train_loader = make_loader(
            x_train,
            y_train,
            pid_train,
            batch_size,
            augment=True,
            class_boost_map=class_boost_map,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        validation_loader = make_loader(
            x_validation,
            y_validation,
            pid_validation,
            batch_size,
            augment=False,
            class_boost_map=None,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
        model = model_factory(num_classes, parameters).to(device)
        print(f"Trial {trial_index}/{len(candidates)}: {parameters}")
        fit = fit_model(
            model,
            train_loader,
            validation_loader,
            device,
            learning_rate=float(parameters["lr"]),
            weight_decay=float(parameters["weight_decay"]),
            epochs=config.epochs,
            patience=config.patience,
            class_weights=class_weights,
            focal_gamma=float(parameters["focal_gamma"]),
            label_smoothing=float(parameters["label_smoothing"]),
        )
        record = {
            "trial": trial_index,
            "seed": trial_seed,
            "best_validation_macro_f1": fit.best_validation_macro_f1,
            "best_epoch": fit.best_epoch,
            "stopped_early": fit.stopped_early,
            "parameter_count": count_parameters(fit.model),
        }
        for key, value in parameters.items():
            record[key] = (
                json.dumps(serialisable(value))
                if isinstance(value, (tuple, list, dict))
                else value
            )
        trial_records.append(record)
        if fit.best_validation_macro_f1 > best_score:
            best_score = fit.best_validation_macro_f1
            best_params = dict(parameters)

        del model, fit, train_loader, validation_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if best_params is None:
        raise RuntimeError("Hyperparameter search did not complete a trial")
    results = pd.DataFrame(trial_records).sort_values(
        ["best_validation_macro_f1", "trial"],
        ascending=[False, True],
    )
    return best_params, results.reset_index(drop=True)


def prepare_output_directory(
    path: Path, config: ExperimentConfig
) -> bool:
    if config.resume and path.exists():
        existing = {
            item.name for item in path.iterdir() if item.is_file()
        }
        if existing == set(RESULT_FILES):
            print(f"Reusing complete result folder: {path}")
            return True
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        if config.overwrite:
            shutil.rmtree(path)
        elif not config.resume:
            raise FileExistsError(
                f"Result folder already contains files: {path}. "
                "Use --resume or --overwrite explicitly."
            )
    path.mkdir(parents=True, exist_ok=True)
    return False


def save_classification_outputs(
    output_dir: Path,
    evaluation: EvaluationResult,
) -> dict[str, Any]:
    labels = np.arange(len(ENCODED_CLASS_ORDER))
    report = classification_report(
        evaluation.y_true,
        evaluation.y_pred,
        labels=labels,
        target_names=ENCODED_CLASS_ORDER,
        output_dict=True,
        zero_division=0,
        digits=6,
    )
    report_frame = pd.DataFrame(report).T
    report_frame.to_csv(
        output_dir / "classification_report.csv", index=True
    )

    prediction_data: dict[str, Any] = {
        "pid": evaluation.pids,
        "y_true_idx": evaluation.y_true,
        "y_pred_idx": evaluation.y_pred,
        "y_true": [
            ENCODED_CLASS_ORDER[index] for index in evaluation.y_true
        ],
        "y_pred": [
            ENCODED_CLASS_ORDER[index] for index in evaluation.y_pred
        ],
        "correct": (
            evaluation.y_true == evaluation.y_pred
        ).astype(int),
        "confidence": evaluation.probabilities.max(axis=1),
    }
    predictions = pd.DataFrame(prediction_data)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics_row = {
        "loss": evaluation.loss,
        "n_samples": len(evaluation.y_true),
        "n_participants": int(len(np.unique(evaluation.pids.astype(str)))),
        **evaluation.metrics,
    }
    return metrics_row


def evaluate_array(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    loader = make_loader(
        features,
        labels,
        participant_ids,
        batch_size,
        augment=False,
        class_boost_map=None,
        num_workers=0,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    return evaluate_model(
        model, loader, nn.CrossEntropyLoss(), device
    )


def channel_occlusion_table(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    baseline: EvaluationResult,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    baseline_accuracy = baseline.metrics["accuracy"]
    baseline_balanced_accuracy = baseline.metrics["balanced_accuracy"]
    rows = [
        {
            "setting": "baseline",
            "accuracy": baseline_accuracy,
            "balanced_accuracy": baseline_balanced_accuracy,
            "acc_drop": 0.0,
            "bal_acc_drop": 0.0,
        }
    ]
    for channel_index, channel_name in enumerate(("x", "y", "z")):
        occluded = features.copy()
        occluded[:, channel_index, :] = 0.0
        result = evaluate_array(
            model,
            occluded,
            labels,
            participant_ids,
            batch_size,
            device,
        )
        accuracy = result.metrics["accuracy"]
        balanced_accuracy = result.metrics["balanced_accuracy"]
        rows.append(
            {
                "setting": f"occlude_{channel_name}",
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "acc_drop": baseline_accuracy - accuracy,
                "bal_acc_drop": (
                    baseline_balanced_accuracy - balanced_accuracy
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "setting",
            "accuracy",
            "balanced_accuracy",
            "acc_drop",
            "bal_acc_drop",
        ),
    )


def temporal_occlusion_table(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    baseline: EvaluationResult,
    batch_size: int,
    device: torch.device,
    number_of_segments: int = TEMPORAL_OCCLUSION_SEGMENTS,
) -> pd.DataFrame:
    sample_count = features.shape[2]
    segment_length = sample_count // number_of_segments
    baseline_accuracy = baseline.metrics["accuracy"]
    rows = [
        {
            "segment": "baseline",
            "start": None,
            "end": None,
            "accuracy": baseline_accuracy,
            "acc_drop": 0.0,
        }
    ]
    for segment_index in range(number_of_segments):
        start = segment_index * segment_length
        end = (
            sample_count
            if segment_index == number_of_segments - 1
            else (segment_index + 1) * segment_length
        )
        occluded = features.copy()
        occluded[:, :, start:end] = 0.0
        result = evaluate_array(
            model,
            occluded,
            labels,
            participant_ids,
            batch_size,
            device,
        )
        accuracy = result.metrics["accuracy"]
        rows.append(
            {
                "segment": f"segment_{segment_index + 1}",
                "start": start,
                "end": end,
                "accuracy": accuracy,
                "acc_drop": baseline_accuracy - accuracy,
            }
        )
    return pd.DataFrame(
        rows,
        columns=("segment", "start", "end", "accuracy", "acc_drop"),
    )


def legacy_hyperparameter_table(
    search_results: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    columns = ["batch_size"]
    if model_name != "cnn":
        columns.append("bidirectional")
    columns.extend(
        [
            "conv_channels",
            "dropout",
            "focal_gamma",
            "kernel_size",
            "label_smoothing",
            "lr",
        ]
    )
    if model_name != "cnn":
        columns.extend(["lstm_hidden", "lstm_layers"])
    columns.extend(["weight_decay", "best_val_f1"])

    table = search_results.rename(
        columns={"best_validation_macro_f1": "best_val_f1"}
    ).copy()

    def tuple_text(value: Any) -> str:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return value
            if isinstance(decoded, list):
                return str(tuple(decoded))
        if isinstance(value, list):
            return str(tuple(value))
        return str(value)

    table["conv_channels"] = table["conv_channels"].map(tuple_text)
    return table.loc[:, columns]


def save_result_folder(
    output_dir: Path,
    *,
    model: nn.Module,
    model_name: str,
    search_results: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    evaluation: EvaluationResult,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    save_classification_outputs(output_dir, evaluation)
    legacy_hyperparameter_table(
        search_results, model_name
    ).to_csv(
        output_dir / "hyperparameter_results.csv",
        index=False,
    )
    channel_occlusion_table(
        model,
        features,
        labels,
        participant_ids,
        evaluation,
        batch_size,
        device,
    ).to_csv(
        output_dir / "channel_occlusion_importance.csv",
        index=False,
    )
    temporal_occlusion_table(
        model,
        features,
        labels,
        participant_ids,
        evaluation,
        batch_size,
        device,
    ).to_csv(
        output_dir / "temporal_occlusion_importance.csv",
        index=False,
    )
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    return {
        "macro_f1": evaluation.metrics["macro_f1"],
        "accuracy": evaluation.metrics["accuracy"],
    }


# Experiment orchestration


def train_source_device(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    model_factory: ModelFactory,
    search_space: Sequence[Mapping[str, Any]],
    source_device: str,
    device: torch.device,
) -> dict[str, Any]:
    """Train on one source device and evaluate within and across devices."""

    target_device = OTHER_DEVICE[source_device]
    within_dir = config.output_dir("within_device", source_device)
    cross_dir = config.output_dir("cross_device", source_device)
    within_complete = prepare_output_directory(within_dir, config)
    cross_complete = prepare_output_directory(cross_dir, config)
    if within_complete and cross_complete:
        return {
            "source_device": source_device,
            "status": "skipped_complete",
            "within_dir": str(within_dir),
            "cross_dir": str(cross_dir),
        }
    if within_complete != cross_complete:
        raise RuntimeError(
            "Resume found only one complete paired result. Remove the "
            "incomplete pair or use --overwrite for this combination."
        )

    start_time = time.perf_counter()
    seed_offset = 0 if source_device == "axivity" else 10_000
    seed_everything(config.seed + seed_offset, config.deterministic)
    x_training_raw = bundle.train_x[source_device]
    y_training = encode_labels(bundle.train_y)
    pid_training = bundle.train_pid
    x_within_raw = bundle.test_x[source_device]
    x_cross_raw = bundle.test_x[target_device]
    y_test = encode_labels(bundle.test_y)
    pid_test = bundle.test_pid
    num_classes = len(ENCODED_CLASS_ORDER)

    search_train_index, search_validation_index, _ = (
        split_indices(
            y_training,
            pid_training,
            config.search_validation_fraction,
            config.seed + seed_offset,
            config.split_mode,
        )
    )
    search_mean, search_standard_deviation = compute_channel_stats(
        x_training_raw[search_train_index]
    )
    x_search_train = apply_zscore(
        x_training_raw[search_train_index],
        search_mean,
        search_standard_deviation,
    )
    x_search_validation = apply_zscore(
        x_training_raw[search_validation_index],
        search_mean,
        search_standard_deviation,
    )
    search_class_weights, _ = compute_effective_number_weights(
        y_training[search_train_index], num_classes
    )
    best_params, search_results = run_hyperparameter_search(
        model_factory=model_factory,
        search_space=search_space,
        x_train=x_search_train,
        y_train=y_training[search_train_index],
        pid_train=pid_training[search_train_index],
        x_validation=x_search_validation,
        y_validation=y_training[search_validation_index],
        pid_validation=pid_training[search_validation_index],
        num_classes=num_classes,
        class_weights=search_class_weights,
        config=config,
        device=device,
        seed_offset=seed_offset,
    )

    train_index, validation_index, _ = split_indices(
        y_training,
        pid_training,
        config.validation_fraction,
        config.seed + seed_offset + 500,
        config.split_mode,
    )
    development_test_overlap = len(
        set(bundle.train_pid.astype(str))
        & set(bundle.test_pid.astype(str))
    )
    if development_test_overlap:
        warnings.warn(
            "Development and held-out test sets share "
            f"{development_test_overlap} participant IDs. Confirm that this "
            "matches the study design and report it in the manuscript.",
            RuntimeWarning,
        )
    normalisation_mean, normalisation_standard_deviation = (
        compute_channel_stats(x_training_raw[train_index])
    )
    x_train = apply_zscore(
        x_training_raw[train_index],
        normalisation_mean,
        normalisation_standard_deviation,
    )
    x_validation = apply_zscore(
        x_training_raw[validation_index],
        normalisation_mean,
        normalisation_standard_deviation,
    )
    x_within = apply_zscore(
        x_within_raw,
        normalisation_mean,
        normalisation_standard_deviation,
    )
    x_cross = apply_zscore(
        x_cross_raw,
        normalisation_mean,
        normalisation_standard_deviation,
    )

    class_weights, _ = compute_effective_number_weights(
        y_training[train_index], num_classes
    )
    class_boost_map = build_class_boost_map(
        y_training[train_index], num_classes
    )
    batch_size = int(best_params["batch_size"])
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        x_train,
        y_training[train_index],
        pid_training[train_index],
        batch_size,
        augment=True,
        class_boost_map=class_boost_map,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    validation_loader = make_loader(
        x_validation,
        y_training[validation_index],
        pid_training[validation_index],
        batch_size,
        augment=False,
        class_boost_map=None,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    within_loader = make_loader(
        x_within,
        y_test,
        pid_test,
        batch_size,
        augment=False,
        class_boost_map=None,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    cross_loader = make_loader(
        x_cross,
        y_test,
        pid_test,
        batch_size,
        augment=False,
        class_boost_map=None,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    final_seed = config.seed + seed_offset + 1_000
    seed_everything(final_seed, config.deterministic)
    model = model_factory(num_classes, best_params).to(device)
    fit_model(
        model,
        train_loader,
        validation_loader,
        device,
        learning_rate=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        epochs=config.epochs,
        patience=config.patience,
        class_weights=class_weights,
        focal_gamma=float(best_params["focal_gamma"]),
        label_smoothing=float(best_params["label_smoothing"]),
    )
    evaluation_criterion = nn.CrossEntropyLoss()
    within_evaluation = evaluate_model(
        model, within_loader, evaluation_criterion, device
    )
    cross_evaluation = evaluate_model(
        model, cross_loader, evaluation_criterion, device
    )
    within_metrics = save_result_folder(
        within_dir,
        model=model,
        model_name=config.model_name,
        search_results=search_results,
        features=x_within,
        labels=y_test,
        participant_ids=pid_test,
        evaluation=within_evaluation,
        batch_size=batch_size,
        device=device,
    )
    cross_metrics = save_result_folder(
        cross_dir,
        model=model,
        model_name=config.model_name,
        search_results=search_results,
        features=x_cross,
        labels=y_test,
        participant_ids=pid_test,
        evaluation=cross_evaluation,
        batch_size=batch_size,
        device=device,
    )

    elapsed_seconds = time.perf_counter() - start_time
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "source_device": source_device,
        "status": "complete",
        "elapsed_seconds": elapsed_seconds,
        "within_dir": str(within_dir),
        "cross_dir": str(cross_dir),
        "within_macro_f1": within_metrics["macro_f1"],
        "cross_macro_f1": cross_metrics["macro_f1"],
    }


def run_model_experiment(
    config: ExperimentConfig,
    model_factory: ModelFactory,
    search_space: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    device = resolve_torch_device(config.device_request)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(
        f"Model={config.model_name}, window={config.window_seconds}s, "
        f"purity={config.purity_label}, run={config.run_name}"
    )
    bundle = load_dataset_bundle(config)
    results = []
    for source_device in SOURCE_DEVICES:
        results.append(
            train_source_device(
                bundle=bundle,
                config=config,
                model_factory=model_factory,
                search_space=search_space,
                source_device=source_device,
                device=device,
            )
        )
    return results
