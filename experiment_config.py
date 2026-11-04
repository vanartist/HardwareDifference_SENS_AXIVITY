"""Load and validate the external experiment configuration.

Runtime values live in ``config.json`` instead of being scattered through the
Python source. Set ``SENS_DL_CONFIG`` to use a different JSON file. Relative
paths are resolved from the directory containing that JSON file.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(
    os.environ.get("SENS_DL_CONFIG", PROJECT_DIR / "config.json")
).expanduser().resolve()


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}. "
            "Copy or edit config.json before running the experiment."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in configuration file {path}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return data


CONFIG = _read_config(CONFIG_PATH)
CONFIG_DIR = CONFIG_PATH.parent


def _section(name: str) -> dict[str, Any]:
    value = CONFIG.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {name!r}")
    return value


def _required(section: str, key: str) -> Any:
    values = _section(section)
    if key not in values:
        raise ValueError(f"Missing config value: {section}.{key}")
    return values[key]


def _path(key: str) -> Path:
    raw = str(_required("paths", key)).strip()
    if not raw:
        raise ValueError(f"Config path paths.{key} cannot be empty")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else CONFIG_DIR / path


def get_model_search_space(model_name: str) -> list[dict[str, Any]]:
    """Return a private copy of one model's configured search trials."""

    spaces = _section("search_spaces")
    value = spaces.get(model_name)
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"search_spaces.{model_name} must be a non-empty JSON array"
        )
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(
            f"Every item in search_spaces.{model_name} must be an object"
        )
    return copy.deepcopy(value)


# Stable result schema and labels (not runtime knobs).
CLASS_ORDER = ("Lie", "Sit", "Stand", "Walk", "Stairs", "Run")
SOURCE_DEVICES = ("axivity", "sens")
DEVICE_DISPLAY = {"axivity": "Axivity", "sens": "SENS"}
OTHER_DEVICE = {"axivity": "sens", "sens": "axivity"}
MODEL_DISPLAY = {
    "cnn": "CNN",
    "cnn_bilstm": "CNN-BiLSTM",
    "cnn_bilstm_attention": "CNN-BiLSTM-Attention",
}
MODEL_FOLDER_NAME = {
    "cnn": "cnn",
    "cnn_bilstm": "cnn_BiLSTM",
    "cnn_bilstm_attention": "cnn_BiLSTM_Attention",
}
RESULT_FILES = (
    "best_model.pt",
    "channel_occlusion_importance.csv",
    "classification_report.csv",
    "hyperparameter_results.csv",
    "temporal_occlusion_importance.csv",
    "test_predictions.csv",
)

# Paths and data preparation.
TRAIN_DATA_DIR = _path("train_data_dir")
TEST_DATA_DIR = _path("test_data_dir")
CLASS_MAP_PATH = _path("class_map_path")
CACHE_DIR = _path("cache_dir")
RESULTS_DIR = _path("results_dir")
SUMMARY_DIR = _path("summary_dir")
FILE_PATTERN = str(_required("paths", "file_pattern"))

# Experiment plan.
NUMBER_OF_RUNS = int(_required("experiment", "number_of_runs"))
RUN_PREFIX = str(_required("experiment", "run_prefix"))
MODELS = list(_required("experiment", "models"))
WINDOWS = [int(value) for value in _required("experiment", "windows_seconds")]
PURITIES = list(_required("experiment", "purities"))
CUSTOM_COMBINATIONS = [
    tuple(item) for item in _required("experiment", "custom_combinations")
]

# Training and execution.
SAMPLE_RATE_HZ = int(_required("training", "sample_rate_hz"))
EPOCHS = int(_required("training", "epochs"))
PATIENCE = int(_required("training", "patience"))
MAX_TRIALS = int(_required("training", "max_trials"))
VALIDATION_FRACTION = float(_required("training", "validation_fraction"))
SEARCH_VALIDATION_FRACTION = float(
    _required("training", "search_validation_fraction")
)
SPLIT_MODE = str(_required("training", "split_mode"))
DEVICE = str(_required("training", "device"))
NUM_WORKERS = int(_required("training", "num_workers"))
DETERMINISTIC = bool(_required("training", "deterministic"))
USE_CACHE = bool(_required("training", "use_cache"))
CONTINUE_ON_ERROR = bool(_required("training", "continue_on_error"))

_AUGMENTATION = _required("training", "augmentation")
if not isinstance(_AUGMENTATION, dict):
    raise ValueError("training.augmentation must be an object")
AUGMENTATION_JITTER_STD = float(_AUGMENTATION["jitter_std"])
AUGMENTATION_SCALE_STD = float(_AUGMENTATION["scale_std"])
AUGMENTATION_TIME_MASK_PROBABILITY = float(
    _AUGMENTATION["time_mask_probability"]
)
AUGMENTATION_MAX_MASK_RATIO = float(_AUGMENTATION["max_mask_ratio"])
AUGMENTATION_BASE_PROBABILITY = float(_AUGMENTATION["base_probability"])
AUGMENTATION_MAXIMUM_PROBABILITY = float(
    _AUGMENTATION["maximum_probability"]
)
AUGMENTATION_MAXIMUM_MASK_PROBABILITY = float(
    _AUGMENTATION["maximum_mask_probability"]
)
AUGMENTATION_STD_BOOST_CAP = float(
    _AUGMENTATION["standard_deviation_boost_cap"]
)
AUGMENTATION_MASK_RATIO_CAP = float(_AUGMENTATION["mask_ratio_cap"])
AUGMENTATION_MAXIMUM_CLASS_BOOST = float(
    _AUGMENTATION["maximum_class_boost"]
)

_CLASS_WEIGHTS = _required("training", "class_weights")
if not isinstance(_CLASS_WEIGHTS, dict):
    raise ValueError("training.class_weights must be an object")
CLASS_WEIGHT_BETA = float(_CLASS_WEIGHTS["beta"])
CLASS_WEIGHT_MINIMUM = float(_CLASS_WEIGHTS["minimum"])
CLASS_WEIGHT_MAXIMUM = float(_CLASS_WEIGHTS["maximum"])

_SCHEDULER = _required("training", "scheduler")
if not isinstance(_SCHEDULER, dict):
    raise ValueError("training.scheduler must be an object")
SCHEDULER_FACTOR = float(_SCHEDULER["factor"])
SCHEDULER_PATIENCE = int(_SCHEDULER["patience"])
TEMPORAL_OCCLUSION_SEGMENTS = int(
    _required("training", "temporal_occlusion_segments")
)

# Summary generation.
INCLUDE_PARTIAL_RESULTS = bool(_required("summary", "include_partial_results"))
OVERWRITE_SUMMARY = bool(_required("summary", "overwrite_output"))
SUMMARY_DPI = int(_required("summary", "dpi"))


def _validate() -> None:
    known_models = set(MODEL_DISPLAY)
    if NUMBER_OF_RUNS < 1:
        raise ValueError("experiment.number_of_runs must be positive")
    if not MODELS or not set(MODELS) <= known_models:
        raise ValueError(
            f"experiment.models must use: {sorted(known_models)}"
        )
    if not WINDOWS or any(window < 1 for window in WINDOWS):
        raise ValueError(
            "experiment.windows_seconds must contain positive integers"
        )
    if not PURITIES or any(
        str(value).lower() not in {"pure", "mixed"} for value in PURITIES
    ):
        raise ValueError(
            "experiment.purities must contain 'pure' and/or 'mixed'"
        )
    if SAMPLE_RATE_HZ < 1:
        raise ValueError("training.sample_rate_hz must be positive")
    if EPOCHS < 1 or PATIENCE < 1 or MAX_TRIALS < 1:
        raise ValueError("epochs, patience, and max_trials must be positive")
    if NUM_WORKERS < 0:
        raise ValueError("training.num_workers cannot be negative")
    probabilities = (
        AUGMENTATION_TIME_MASK_PROBABILITY,
        AUGMENTATION_BASE_PROBABILITY,
        AUGMENTATION_MAXIMUM_PROBABILITY,
        AUGMENTATION_MAXIMUM_MASK_PROBABILITY,
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("augmentation probabilities must be between 0 and 1")
    if AUGMENTATION_JITTER_STD < 0 or AUGMENTATION_SCALE_STD < 0:
        raise ValueError("augmentation standard deviations cannot be negative")
    if not 0.0 <= AUGMENTATION_MAX_MASK_RATIO <= 1.0:
        raise ValueError("augmentation.max_mask_ratio must be between 0 and 1")
    if not 0.0 <= AUGMENTATION_MASK_RATIO_CAP <= 1.0:
        raise ValueError("augmentation.mask_ratio_cap must be between 0 and 1")
    if not 0.0 < CLASS_WEIGHT_BETA < 1.0:
        raise ValueError("class_weights.beta must be between 0 and 1")
    if CLASS_WEIGHT_MINIMUM <= 0 or CLASS_WEIGHT_MAXIMUM < CLASS_WEIGHT_MINIMUM:
        raise ValueError("invalid class weight clipping range")
    if SCHEDULER_FACTOR <= 0 or SCHEDULER_FACTOR >= 1:
        raise ValueError("scheduler.factor must be between 0 and 1")
    if SCHEDULER_PATIENCE < 0:
        raise ValueError("scheduler.patience cannot be negative")
    if TEMPORAL_OCCLUSION_SEGMENTS < 1:
        raise ValueError("training.temporal_occlusion_segments must be positive")
    if SUMMARY_DPI < 1:
        raise ValueError("summary.dpi must be positive")
    for model_name in MODELS:
        get_model_search_space(model_name)


_validate()
