"""Aggregate repeated runs into tables and confusion matrices."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter
from PIL import Image, ImageDraw, ImageFont

from generate_appendix_table_a1 import generate as generate_appendix_table_a1
from generate_appendix_table_a2 import generate as generate_appendix_table_a2
from experiment_config import (
    CLASS_ORDER,
    DEVICE_DISPLAY,
    INCLUDE_PARTIAL_RESULTS,
    MODELS,
    MODEL_DISPLAY,
    OTHER_DEVICE,
    OVERWRITE_SUMMARY,
    PURITIES,
    RESULTS_DIR,
    SOURCE_DEVICES,
    SUMMARY_DIR,
    SUMMARY_DPI,
    WINDOWS,
)


MODEL_ORDER = tuple(MODELS)
WINDOW_ORDER = tuple(WINDOWS)
PURITY_ORDER = tuple(str(value).lower() for value in PURITIES)
EXPERIMENT_TYPES = ("within_device", "cross_device")
EXPECTED_CASES_PER_COMPLETE_RUN = (
    len(MODEL_ORDER)
    * len(WINDOW_ORDER)
    * len(PURITY_ORDER)
    * len(SOURCE_DEVICES)
    * len(EXPERIMENT_TYPES)
)
FOLDER_PATTERN = re.compile(
    r"^(cnn_bilstm_attention|cnn_bilstm|cnn)_"
    r"([1-9]\d*)_drop_impure_(True|False)_(axivity|sens)$",
    flags=re.IGNORECASE,
)
SUMMARY_REQUIRED_TABLES = (
    "classification_report",
    "test_predictions",
)
SUPPORTED_TABLE_SUFFIXES = (".csv", ".xlsx")
FIGURE_CLASS_ORDER = ("Lie", "Sit", "Stand", "Walk", "Run", "Stairs")
FIGURE_CLASS_LABELS = {
    "Lie": "Lie",
    "Sit": "Sit",
    "Stand": "Stand",
    "Walk": "Walk",
    "Run": "Run",
    "Stairs": "Stair",
}
TOTAL_CELL_COLOR = (205, 205, 205)

GREEN_STOPS = (
    (255, 255, 255),
    (237, 248, 233),
    (186, 228, 179),
    (116, 196, 118),
    (35, 139, 69),
    (0, 90, 50),
)
CELL_SIZE = 108
PANEL_LEFT = 118
PANEL_TOP = 28
PANEL_RIGHT = 98
PANEL_BOTTOM = 152
PANEL_GAP = 18
CASE_HEADER_HEIGHT = 34
COLORBAR_GAP = 20
COLORBAR_RIGHT_AREA = 132


# Result records


@dataclass(frozen=True)
class ModelCase:
    result_folder: str
    experiment_type: str
    folder_name: str
    folder_path: Path
    report_path: Path
    predictions_path: Path
    model: str
    window: int
    purity: str
    source_device: str
    target_device: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    n_samples: int
    n_participants: float

    @property
    def direction(self) -> str:
        return (
            f"{DEVICE_DISPLAY[self.source_device]}->"
            f"{DEVICE_DISPLAY[self.target_device]}"
        )


@dataclass
class PerformanceRecord:
    source_device: str
    window: int
    purity: str
    model: str
    baseline_direction: str
    cross_direction: str
    baseline_mean: float
    baseline_sd: float
    baseline_n: int
    cross_mean: float
    cross_sd: float
    cross_n: int
    relative_change_pct: float
    selected: bool = False


@dataclass(frozen=True)
class SelectionRecord:
    performance: PerformanceRecord
    baseline_case: ModelCase
    cross_case: ModelCase


# Result discovery and parsing


def result_sort_key(name: str) -> tuple[int, str]:
    if name == "Result":
        return (0, name)
    match = re.fullmatch(r"Result_(\d+)", name)
    if match:
        return (int(match.group(1)) + 1, name)
    return (10_000, name)


def parse_folder_name(
    folder_name: str,
) -> tuple[str, int, str, str] | None:
    match = FOLDER_PATTERN.fullmatch(folder_name)
    if not match:
        return None
    model = match.group(1).lower()
    window = int(match.group(2))
    purity = "pure" if match.group(3).lower() == "true" else "mixed"
    source_device = match.group(4).lower()
    return model, window, purity, source_device


def result_table_path(
    folder: Path,
    stem: str,
    required: bool = True,
) -> Path | None:
    for suffix in SUPPORTED_TABLE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    if required:
        supported = ", ".join(
            f"{stem}{suffix}" for suffix in SUPPORTED_TABLE_SUFFIXES
        )
        raise FileNotFoundError(
            f"Missing result table in {folder}; expected {supported}"
        )
    return None


def read_result_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported result table format: {path}")


def has_summary_inputs(folder: Path) -> bool:
    return all(
        result_table_path(folder, stem, required=False) is not None
        for stem in SUMMARY_REQUIRED_TABLES
    )


def count_valid_cases(result_dir: Path) -> int:
    count = 0
    for experiment_type in EXPERIMENT_TYPES:
        experiment_dir = result_dir / experiment_type
        if not experiment_dir.exists():
            continue
        for folder in experiment_dir.iterdir():
            if (
                folder.is_dir()
                and parse_folder_name(folder.name) is not None
                and has_summary_inputs(folder)
            ):
                count += 1
    return count


def discover_result_folders(
    results_root: Path, include_partial: bool
) -> list[str]:
    candidates = sorted(
        (
            path
            for path in results_root.iterdir()
            if path.is_dir() and result_sort_key(path.name)[0] < 10_000
        ),
        key=lambda path: result_sort_key(path.name),
    )
    if include_partial:
        return [path.name for path in candidates if count_valid_cases(path)]
    complete = [
        path.name
        for path in candidates
        if count_valid_cases(path) == EXPECTED_CASES_PER_COMPLETE_RUN
    ]
    if complete:
        return complete
    return [path.name for path in candidates if count_valid_cases(path)]


def metric_from_report(
    report_path: Path,
    metrics_path: Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    report = read_result_table(report_path, index_col=0)
    report.index = report.index.map(lambda value: str(value).strip())
    report = report.rename(
        columns={
            "f1-score": "f1_score",
        },
        index={
            "macro avg": "macro_avg",
            "weighted avg": "weighted_avg",
        },
    )
    for column in report.columns:
        report[column] = pd.to_numeric(report[column], errors="coerce")

    def get(row: str, column: str) -> float:
        if row not in report.index or column not in report.columns:
            return float("nan")
        value = report.loc[row, column]
        return float(value) if pd.notna(value) else float("nan")

    accuracy = get("accuracy", "f1_score")
    if not np.isfinite(accuracy):
        accuracy = get("accuracy", "precision")
    metrics = {
        "accuracy": accuracy,
        "macro_precision": get("macro_avg", "precision"),
        "macro_recall": get("macro_avg", "recall"),
        "macro_f1": get("macro_avg", "f1_score"),
        "weighted_f1": get("weighted_avg", "f1_score"),
        "n_samples": get("weighted_avg", "support"),
        "n_participants": float("nan"),
    }
    if metrics_path is not None and metrics_path.exists():
        metrics_frame = read_result_table(metrics_path)
        if not metrics_frame.empty:
            metrics_row = metrics_frame.iloc[0]
            for key in metrics:
                value = pd.to_numeric(metrics_row.get(key), errors="coerce")
                if pd.notna(value):
                    metrics[key] = float(value)
    if not np.isfinite(metrics["macro_f1"]):
        raise ValueError(f"macro-F1 is missing from {report_path}")
    return metrics, report


def scan_cases(
    results_root: Path, result_folders: Sequence[str]
) -> tuple[list[ModelCase], pd.DataFrame]:
    cases: list[ModelCase] = []
    completeness_rows = []
    for result_folder in result_folders:
        result_dir = results_root / result_folder
        valid_count = 0
        for experiment_type in EXPERIMENT_TYPES:
            experiment_dir = result_dir / experiment_type
            if not experiment_dir.exists():
                continue
            for folder in sorted(experiment_dir.iterdir()):
                if not folder.is_dir():
                    continue
                parsed = parse_folder_name(folder.name)
                if parsed is None:
                    continue
                if not has_summary_inputs(folder):
                    continue
                model, window, purity, source_device = parsed
                report_path = result_table_path(
                    folder, "classification_report"
                )
                predictions_path = result_table_path(
                    folder, "test_predictions"
                )
                assert report_path is not None
                assert predictions_path is not None
                metrics, _ = metric_from_report(
                    report_path,
                    result_table_path(folder, "metrics", required=False),
                )
                predictions_frame: pd.DataFrame | None = None
                if not np.isfinite(metrics["n_samples"]):
                    predictions_frame = read_result_table(predictions_path)
                    metrics["n_samples"] = float(len(predictions_frame))
                if not np.isfinite(metrics["n_participants"]):
                    try:
                        if predictions_frame is None:
                            predictions_frame = read_result_table(
                                predictions_path, usecols=["pid"]
                            )
                        participant_frame = predictions_frame[["pid"]]
                    except (ValueError, OSError, pd.errors.ParserError):
                        pass
                    else:
                        metrics["n_participants"] = float(
                            participant_frame["pid"].nunique()
                        )
                target_device = (
                    source_device
                    if experiment_type == "within_device"
                    else OTHER_DEVICE[source_device]
                )
                cases.append(
                    ModelCase(
                        result_folder=result_folder,
                        experiment_type=experiment_type,
                        folder_name=folder.name,
                        folder_path=folder,
                        report_path=report_path,
                        predictions_path=predictions_path,
                        model=model,
                        window=window,
                        purity=purity,
                        source_device=source_device,
                        target_device=target_device,
                        accuracy=metrics["accuracy"],
                        macro_precision=metrics["macro_precision"],
                        macro_recall=metrics["macro_recall"],
                        macro_f1=metrics["macro_f1"],
                        weighted_f1=metrics["weighted_f1"],
                        n_samples=int(metrics["n_samples"]),
                        n_participants=metrics["n_participants"],
                    )
                )
                valid_count += 1
        completeness_rows.append(
            {
                "result_folder": result_folder,
                "valid_model_folders": valid_count,
                "expected_for_full_design": EXPECTED_CASES_PER_COMPLETE_RUN,
                "complete_full_design": (
                    valid_count == EXPECTED_CASES_PER_COMPLETE_RUN
                ),
            }
        )
    return cases, pd.DataFrame(completeness_rows)


def matching_cases(
    cases: Iterable[ModelCase],
    experiment_type: str,
    source_device: str,
    model: str,
    window: int,
    purity: str,
) -> list[ModelCase]:
    return [
        case
        for case in cases
        if case.experiment_type == experiment_type
        and case.source_device == source_device
        and case.model == model
        and case.window == window
        and case.purity == purity
    ]


# Aggregation and representative-run selection


def mean_sd(cases: Sequence[ModelCase]) -> tuple[float, float, int]:
    values = np.asarray([case.macro_f1 for case in cases], dtype=float)
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        int(len(values)),
    )


def build_performance_records(
    cases: Sequence[ModelCase],
) -> list[PerformanceRecord]:
    records: list[PerformanceRecord] = []
    for source_device in SOURCE_DEVICES:
        for window in WINDOW_ORDER:
            for model in MODEL_ORDER:
                for purity in PURITY_ORDER:
                    baseline = matching_cases(
                        cases,
                        "within_device",
                        source_device,
                        model,
                        window,
                        purity,
                    )
                    cross = matching_cases(
                        cases,
                        "cross_device",
                        source_device,
                        model,
                        window,
                        purity,
                    )
                    if not baseline or not cross:
                        continue
                    baseline_mean, baseline_sd, baseline_n = mean_sd(baseline)
                    cross_mean, cross_sd, cross_n = mean_sd(cross)
                    relative_change = (
                        (cross_mean - baseline_mean)
                        / baseline_mean
                        * 100.0
                        if baseline_mean
                        else float("nan")
                    )
                    records.append(
                        PerformanceRecord(
                            source_device=source_device,
                            window=window,
                            purity=purity,
                            model=model,
                            baseline_direction=(
                                f"{DEVICE_DISPLAY[source_device]}->"
                                f"{DEVICE_DISPLAY[source_device]}"
                            ),
                            cross_direction=(
                                f"{DEVICE_DISPLAY[source_device]}->"
                                f"{DEVICE_DISPLAY[OTHER_DEVICE[source_device]]}"
                            ),
                            baseline_mean=baseline_mean,
                            baseline_sd=baseline_sd,
                            baseline_n=baseline_n,
                            cross_mean=cross_mean,
                            cross_sd=cross_sd,
                            cross_n=cross_n,
                            relative_change_pct=relative_change,
                        )
                    )

    for source_device in SOURCE_DEVICES:
        for window in WINDOW_ORDER:
            for purity in PURITY_ORDER:
                group = [
                    record
                    for record in records
                    if record.source_device == source_device
                    and record.window == window
                    and record.purity == purity
                ]
                if not group:
                    continue
                # Select the largest cross-device change from baseline.
                selected = max(
                    group,
                    key=lambda record: (
                        record.relative_change_pct,
                        record.cross_mean,
                        -record.cross_sd,
                        record.model,
                    ),
                )
                selected.selected = True
    return records


def select_mean_closest_case(
    cases: Sequence[ModelCase],
) -> ModelCase:
    if not cases:
        raise ValueError("Cannot select a representative from an empty set")
    condition_mean = float(
        np.mean([case.macro_f1 for case in cases])
    )
    result_order = {
        name: index
        for index, name in enumerate(
            sorted(
                {case.result_folder for case in cases},
                key=result_sort_key,
            )
        )
    }
    return min(
        cases,
        key=lambda case: (
            abs(case.macro_f1 - condition_mean),
            result_order.get(case.result_folder, 10_000),
            case.folder_name,
        ),
    )


def build_selection_records(
    cases: Sequence[ModelCase],
    performance: Sequence[PerformanceRecord],
) -> list[SelectionRecord]:
    selections: list[SelectionRecord] = []
    for record in performance:
        if not record.selected:
            continue
        baseline_cases = matching_cases(
            cases,
            "within_device",
            record.source_device,
            record.model,
            record.window,
            record.purity,
        )
        cross_cases = matching_cases(
            cases,
            "cross_device",
            record.source_device,
            record.model,
            record.window,
            record.purity,
        )
        selections.append(
            SelectionRecord(
                performance=record,
                baseline_case=select_mean_closest_case(baseline_cases),
                cross_case=select_mean_closest_case(cross_cases),
            )
        )
    return sorted(
        selections,
        key=lambda selection: (
            SOURCE_DEVICES.index(selection.performance.source_device),
            WINDOW_ORDER.index(selection.performance.window),
            PURITY_ORDER.index(selection.performance.purity),
        ),
    )


def normalise_label(value: Any) -> str:
    text = str(value).strip()
    mapping = {
        "lie": "Lie",
        "lying": "Lie",
        "sit": "Sit",
        "sitting": "Sit",
        "stand": "Stand",
        "standing": "Stand",
        "walk": "Walk",
        "walking": "Walk",
        "stairs": "Stairs",
        "stair": "Stairs",
        "run": "Run",
        "running": "Run",
    }
    return mapping.get(text.lower(), text)


# Confusion-matrix rendering


def load_confusion(
    predictions_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = read_result_table(predictions_path)
    missing = {"y_true", "y_pred"} - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)} in {predictions_path}"
        )
    y_true = predictions["y_true"].map(normalise_label)
    y_pred = predictions["y_pred"].map(normalise_label)
    label_to_index = {
        label: index for index, label in enumerate(FIGURE_CLASS_ORDER)
    }
    matrix = np.zeros(
        (len(FIGURE_CLASS_ORDER), len(FIGURE_CLASS_ORDER)), dtype=int
    )
    for true_label, predicted_label in zip(y_true, y_pred):
        if (
            true_label not in label_to_index
            or predicted_label not in label_to_index
        ):
            continue
        matrix[
            label_to_index[true_label], label_to_index[predicted_label]
        ] += 1
    true_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix.astype(float),
        true_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=true_totals != 0,
    )
    return matrix, normalised


def load_figure_font(
    size: int,
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path(
            r"C:\Windows\Fonts\arialbd.ttf"
            if bold
            else r"C:\Windows\Fonts\arial.ttf"
        ),
        Path(
            r"C:\Windows\Fonts\calibrib.ttf"
            if bold
            else r"C:\Windows\Fonts\calibri.ttf"
        ),
        Path(
            r"C:\Windows\Fonts\segoeuib.ttf"
            if bold
            else r"C:\Windows\Fonts\segoeui.ttf"
        ),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def figure_fonts() -> dict[
    str,
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
]:
    return {
        "heading": load_figure_font(20, bold=True),
        "axis": load_figure_font(24, bold=True),
        "tick": load_figure_font(22),
        "cell": load_figure_font(24),
        "cell_bold": load_figure_font(24, bold=True),
        "colorbar": load_figure_font(22, bold=True),
        "colorbar_tick": load_figure_font(20),
    }


def interpolate_color(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    return tuple(
        int(round(left[index] + (right[index] - left[index]) * ratio))
        for index in range(3)
    )


def percentage_color(value: float) -> tuple[int, int, int]:
    clipped = min(max(float(value), 0.0), 1.0)
    if clipped >= 1.0:
        return GREEN_STOPS[-1]
    scaled = clipped * (len(GREEN_STOPS) - 1)
    index = int(np.floor(scaled))
    return interpolate_color(
        GREEN_STOPS[index],
        GREEN_STOPS[index + 1],
        scaled - index,
    )


def text_size(
    drawing: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    bounds = drawing.textbbox((0, 0), value, font=font)
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def draw_centered_text(
    drawing: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    font: ImageFont.ImageFont,
    spacing: int = 4,
) -> None:
    lines = value.splitlines()
    line_sizes = [text_size(drawing, line, font) for line in lines]
    total_height = (
        sum(height for _, height in line_sizes)
        + spacing * (len(lines) - 1)
    )
    y_position = box[1] + ((box[3] - box[1]) - total_height) / 2
    for line, (width, height) in zip(lines, line_sizes):
        x_position = box[0] + ((box[2] - box[0]) - width) / 2
        drawing.text(
            (x_position, y_position),
            line,
            font=font,
            fill=(0, 0, 0),
        )
        y_position += height + spacing


def draw_rotated_text(
    image: Image.Image,
    value: str,
    position: tuple[int, int],
    font: ImageFont.ImageFont,
    angle: float,
    anchor: str,
) -> None:
    sample = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    sample_drawing = ImageDraw.Draw(sample)
    width, height = text_size(sample_drawing, value, font)
    text_image = Image.new(
        "RGBA",
        (width + 8, height + 8),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(text_image).text(
        (4, 4),
        value,
        font=font,
        fill=(0, 0, 0, 255),
    )
    rotated = text_image.rotate(
        angle,
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )
    x_position, y_position = position
    if anchor == "rt":
        x_position -= rotated.width
    elif anchor == "mm":
        x_position -= rotated.width // 2
        y_position -= rotated.height // 2
    image.alpha_composite(
        rotated,
        (int(x_position), int(y_position)),
    )


def panel_size() -> tuple[int, int]:
    grid_size = CELL_SIZE * (len(FIGURE_CLASS_ORDER) + 1)
    return (
        PANEL_LEFT + grid_size + PANEL_RIGHT,
        PANEL_TOP + grid_size + PANEL_BOTTOM,
    )


def draw_confusion_panel(
    image: Image.Image,
    matrix: np.ndarray,
    normalised: np.ndarray,
    panel_x: int,
    panel_y: int,
    title: str,
    side_label: str,
    fonts: dict[
        str,
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ],
) -> None:
    drawing = ImageDraw.Draw(image)
    class_count = len(FIGURE_CLASS_ORDER)
    size = class_count + 1
    matrix_left = panel_x + PANEL_LEFT
    matrix_top = panel_y + PANEL_TOP
    true_totals = matrix.sum(axis=1)
    predicted_totals = matrix.sum(axis=0)
    grand_total = int(matrix.sum())

    for row in range(size):
        for column in range(size):
            x_start = matrix_left + column * CELL_SIZE
            y_start = matrix_top + row * CELL_SIZE
            box = (
                x_start,
                y_start,
                x_start + CELL_SIZE,
                y_start + CELL_SIZE,
            )
            if row == 0 and column < class_count:
                fill = TOTAL_CELL_COLOR
                label = f"{true_totals[column]:,}"
                font = fonts["cell"]
            elif row == 0 and column == class_count:
                fill = TOTAL_CELL_COLOR
                label = f"{grand_total:,}"
                font = fonts["cell"]
            elif row > 0 and column < class_count:
                predicted_index = class_count - row
                fill = percentage_color(
                    normalised[column, predicted_index]
                )
                label = (
                    f"{normalised[column, predicted_index] * 100:.1f}%\n"
                    f"{matrix[column, predicted_index]:,}"
                )
                font = fonts["cell"]
            else:
                predicted_index = class_count - row
                fill = TOTAL_CELL_COLOR
                label = f"{predicted_totals[predicted_index]:,}"
                font = fonts["cell"]
            drawing.rectangle(
                box,
                fill=fill,
                outline=(240, 240, 240),
                width=1,
            )
            draw_centered_text(drawing, box, label, font)

    for diagonal in range(class_count):
        x_start = matrix_left + diagonal * CELL_SIZE
        y_start = matrix_top + (class_count - diagonal) * CELL_SIZE
        drawing.rectangle(
            (
                x_start,
                y_start,
                x_start + CELL_SIZE,
                y_start + CELL_SIZE,
            ),
            outline=(0, 0, 0),
            width=4,
        )

    y_labels = [
        "Total",
        *(
            FIGURE_CLASS_LABELS[label]
            for label in reversed(FIGURE_CLASS_ORDER)
        ),
    ]
    for index, label in enumerate(y_labels):
        y_position = matrix_top + index * CELL_SIZE + CELL_SIZE // 2
        label_width, label_height = text_size(
            drawing,
            label,
            fonts["tick"],
        )
        drawing.text(
            (
                matrix_left - 14 - label_width,
                y_position - label_height / 2,
            ),
            label,
            font=fonts["tick"],
            fill=(0, 0, 0),
        )
    x_labels = [
        *(FIGURE_CLASS_LABELS[label] for label in FIGURE_CLASS_ORDER),
        "Total",
    ]
    for index, label in enumerate(x_labels):
        x_position = matrix_left + index * CELL_SIZE + CELL_SIZE // 2
        draw_rotated_text(
            image,
            label,
            (x_position + 5, matrix_top + size * CELL_SIZE + 14),
            fonts["tick"],
            angle=45,
            anchor="rt",
        )

    panel_width, panel_height = panel_size()
    axis_width, axis_height = text_size(
        drawing,
        "True label",
        fonts["axis"],
    )
    drawing.text(
        (
            matrix_left + (size * CELL_SIZE - axis_width) / 2,
            panel_y + panel_height - axis_height - 16,
        ),
        "True label",
        font=fonts["axis"],
        fill=(0, 0, 0),
    )
    draw_rotated_text(
        image,
        side_label,
        (panel_x + 28, matrix_top + size * CELL_SIZE // 2),
        fonts["axis"],
        angle=90,
        anchor="mm",
    )

    title_width, _ = text_size(drawing, title, fonts["heading"])
    drawing.text(
        (panel_x + (panel_width - title_width) / 2, 0),
        title,
        font=fonts["heading"],
        fill=(0, 0, 0),
    )


def draw_colorbar(
    image: Image.Image,
    x_position: int,
    y_position: int,
    height: int,
    fonts: dict[
        str,
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ],
) -> None:
    drawing = ImageDraw.Draw(image)
    bar_width = 32
    for offset in range(height):
        value = 1.0 - offset / max(height - 1, 1)
        drawing.line(
            (
                x_position,
                y_position + offset,
                x_position + bar_width,
                y_position + offset,
            ),
            fill=percentage_color(value),
            width=1,
        )
    drawing.rectangle(
        (
            x_position,
            y_position,
            x_position + bar_width,
            y_position + height,
        ),
        outline=(80, 80, 80),
        width=1,
    )
    for tick in np.linspace(0.0, 1.0, 6):
        tick_y = y_position + height - int(round(tick * height))
        drawing.line(
            (
                x_position + bar_width,
                tick_y,
                x_position + bar_width + 7,
                tick_y,
            ),
            fill=(0, 0, 0),
            width=1,
        )
        drawing.text(
            (x_position + bar_width + 11, tick_y - 8),
            f"{int(round(tick * 100))}",
            font=fonts["colorbar_tick"],
            fill=(0, 0, 0),
        )
    draw_centered_text(
        drawing,
        (
            x_position - 12,
            y_position - CELL_SIZE + 8,
            x_position + bar_width + 100,
            y_position - 8,
        ),
        "Column\npercentage",
        fonts["colorbar"],
    )


def save_paired_confusion_matrix(
    selection: SelectionRecord, output_path: Path, dpi: int
) -> None:
    save_case_pair_confusion_matrix(
        selection.baseline_case,
        selection.cross_case,
        output_path,
        dpi,
    )


def save_case_pair_confusion_matrix(
    baseline_case: ModelCase,
    cross_case: ModelCase,
    output_path: Path,
    dpi: int,
) -> None:
    """Save one within-device/cross-device confusion-matrix pair."""
    baseline_matrix, baseline_normalised = load_confusion(
        baseline_case.predictions_path
    )
    cross_matrix, cross_normalised = load_confusion(
        cross_case.predictions_path
    )
    fonts = figure_fonts()
    panel_width, panel_height = panel_size()
    panel_y = CASE_HEADER_HEIGHT
    grid_width = panel_width * 2 + PANEL_GAP
    matrix_right = grid_width - PANEL_RIGHT
    colorbar_x = matrix_right + COLORBAR_GAP
    image_width = colorbar_x + COLORBAR_RIGHT_AREA
    image_height = panel_y + panel_height
    image = Image.new(
        "RGBA",
        (image_width, image_height),
        (255, 255, 255, 255),
    )
    model_name = MODEL_DISPLAY[baseline_case.model]
    draw_confusion_panel(
        image,
        baseline_matrix,
        baseline_normalised,
        0,
        panel_y,
        f"(a) {model_name} | {baseline_case.direction}",
        f"{DEVICE_DISPLAY[baseline_case.source_device]} same device",
        fonts,
    )
    draw_confusion_panel(
        image,
        cross_matrix,
        cross_normalised,
        panel_width + PANEL_GAP,
        panel_y,
        f"(b) {model_name} | {cross_case.direction}",
        f"{DEVICE_DISPLAY[cross_case.source_device]} cross device",
        fonts,
    )
    draw_colorbar(
        image,
        colorbar_x,
        panel_y + PANEL_TOP + CELL_SIZE,
        CELL_SIZE * len(FIGURE_CLASS_ORDER),
        fonts,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, dpi=(dpi, dpi))


def confusion_case_sort_key(case: ModelCase) -> tuple[Any, ...]:
    """Sort run-level pairs in the same order as the experiment plan."""
    return (
        result_sort_key(case.result_folder),
        SOURCE_DEVICES.index(case.source_device),
        WINDOW_ORDER.index(case.window),
        MODEL_ORDER.index(case.model),
        PURITY_ORDER.index(case.purity),
        case.folder_name,
    )


def generate_confusion_matrices(
    cases: Sequence[ModelCase],
    selections: Sequence[SelectionRecord],
    output_dir: Path,
    dpi: int,
) -> None:
    matrix_root = output_dir / "confusion_matrices"

    cross_cases = {
        (
            case.result_folder,
            case.source_device,
            case.model,
            case.window,
            case.purity,
        ): case
        for case in cases
        if case.experiment_type == "cross_device"
    }
    run_pairs: list[tuple[ModelCase, ModelCase]] = []
    baseline_cases = sorted(
        (
            case
            for case in cases
            if case.experiment_type == "within_device"
        ),
        key=confusion_case_sort_key,
    )
    for baseline_case in baseline_cases:
        key = (
            baseline_case.result_folder,
            baseline_case.source_device,
            baseline_case.model,
            baseline_case.window,
            baseline_case.purity,
        )
        cross_case = cross_cases.get(key)
        if cross_case is None:
            continue
        run_pairs.append((baseline_case, cross_case))
        filename = (
            f"{baseline_case.result_folder}_"
            f"{baseline_case.source_device}_{baseline_case.window}s_"
            f"{baseline_case.purity}_{baseline_case.model}_"
            "within_vs_cross_confusion_matrix.png"
        )
        save_case_pair_confusion_matrix(
            baseline_case,
            cross_case,
            matrix_root / filename,
            dpi,
        )

    # Also save the representative pairs used in the reviewer tables.
    for selection in selections:
        record = selection.performance
        purity_label = "pure" if record.purity == "pure" else "mixed"
        filename = (
            f"{record.source_device}_{record.window}s_{purity_label}_"
            f"{record.model}_within_vs_cross_confusion_matrix.png"
        )
        image_path = matrix_root / filename
        save_paired_confusion_matrix(selection, image_path, dpi)

    print(
        f"Confusion matrices: {len(run_pairs)} run-level paired matrices and "
        f"{len(selections)} representative paired matrices"
    )


def metric_text(
    mean: float,
    standard_deviation: float,
    _sample_size: int,
) -> str:
    return f"{mean:.4f} ± {standard_deviation:.4f}"


# Workbook output


def sorted_performance_records(
    records: Sequence[PerformanceRecord],
    source_device: str,
) -> list[PerformanceRecord]:
    return sorted(
        (
            record
            for record in records
            if record.source_device == source_device
        ),
        key=lambda record: (
            WINDOW_ORDER.index(record.window),
            MODEL_ORDER.index(record.model),
            PURITY_ORDER.index(record.purity),
        ),
    )


def complete_table_frame(
    records: Sequence[PerformanceRecord],
    source_device: str,
) -> pd.DataFrame:
    rows = []
    subset = sorted_performance_records(records, source_device)
    for record in subset:
        rows.append(
            {
                "Window": f"{record.window} s",
                "Model": MODEL_DISPLAY[record.model],
                "Purity": (
                    "Pure" if record.purity == "pure" else "Mixed"
                ),
                "Baseline direction": record.baseline_direction,
                "Baseline macro-F1": metric_text(
                    record.baseline_mean,
                    record.baseline_sd,
                    record.baseline_n,
                ),
                "Cross-device direction": record.cross_direction,
                "Cross-device macro-F1": metric_text(
                    record.cross_mean,
                    record.cross_sd,
                    record.cross_n,
                ),
                "Relative change": record.relative_change_pct / 100.0,
            }
        )
    return pd.DataFrame(rows)


def selection_table_frame(
    selections: Sequence[SelectionRecord],
) -> pd.DataFrame:
    rows = []
    for selection in selections:
        record = selection.performance
        rows.append(
            {
                "Source device": DEVICE_DISPLAY[record.source_device],
                "Window": f"{record.window} s",
                "Purity": (
                    "Pure" if record.purity == "pure" else "Mixed"
                ),
                "Selected model": MODEL_DISPLAY[record.model],
                "Baseline macro-F1": metric_text(
                    record.baseline_mean,
                    record.baseline_sd,
                    record.baseline_n,
                ),
                "Cross-device macro-F1": metric_text(
                    record.cross_mean,
                    record.cross_sd,
                    record.cross_n,
                ),
                "Relative change": record.relative_change_pct / 100.0,
            }
        )
    return pd.DataFrame(rows)


def write_table_sheet(
    workbook: Workbook,
    title: str,
    frame: pd.DataFrame,
    bold_data_rows: set[int] | None = None,
) -> None:
    sheet = workbook.create_sheet(title)
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A2"
    sheet.append(frame.columns.tolist())
    for row in frame.itertuples(index=False, name=None):
        sheet.append(list(row))

    top_border = Side(style="medium", color="000000")
    header_border = Border(top=top_border, bottom=top_border)
    bottom_border = Border(bottom=top_border)
    header_font = Font(name="Times New Roman", size=10, bold=True)
    body_font = Font(name="Times New Roman", size=10)
    selected_font = Font(name="Times New Roman", size=10, bold=True)
    centered = Alignment(
        horizontal="center",
        vertical="center",
        wrap_text=True,
    )

    for cell in sheet[1]:
        cell.font = header_font
        cell.alignment = centered
        cell.border = header_border
    sheet.row_dimensions[1].height = 32

    bold_data_rows = bold_data_rows or set()
    for data_index, row in enumerate(
        sheet.iter_rows(min_row=2, max_row=sheet.max_row)
    ):
        font = selected_font if data_index in bold_data_rows else body_font
        for cell in row:
            cell.font = font
            cell.alignment = centered
        sheet.row_dimensions[data_index + 2].height = 22

    if sheet.max_row > 1:
        for cell in sheet[sheet.max_row]:
            cell.border = bottom_border
    sheet.auto_filter.ref = sheet.dimensions

    relative_change_column = None
    for column_index, header in enumerate(frame.columns, start=1):
        if header == "Relative change":
            relative_change_column = column_index
            break
    if relative_change_column is not None:
        for row_index in range(2, sheet.max_row + 1):
            sheet.cell(row_index, relative_change_column).number_format = (
                "+0.00%;-0.00%;0.00%"
            )

    width_by_header = {
        "Source device": 17,
        "Window": 11,
        "Model": 28,
        "Selected model": 28,
        "Purity": 12,
        "Baseline direction": 23,
        "Baseline macro-F1": 24,
        "Cross-device direction": 25,
        "Cross-device macro-F1": 26,
        "Relative change": 18,
    }
    for column_index, header in enumerate(frame.columns, start=1):
        sheet.column_dimensions[
            get_column_letter(column_index)
        ].width = width_by_header.get(str(header), 18)

    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_title_rows = "1:1"
    sheet.auto_filter.ref = f"A1:{get_column_letter(sheet.max_column)}{sheet.max_row}"


def save_summary_workbook(
    performance: Sequence[PerformanceRecord],
    selections: Sequence[SelectionRecord],
    output_path: Path,
) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = "Cross-device performance tables"
    workbook.properties.subject = (
        "Within-device and cross-device activity-recognition performance"
    )

    axivity_records = sorted_performance_records(performance, "axivity")
    sens_records = sorted_performance_records(performance, "sens")
    write_table_sheet(
        workbook,
        "Table S1 Axivity",
        complete_table_frame(performance, "axivity"),
        {
            index
            for index, record in enumerate(axivity_records)
            if record.selected
        },
    )
    write_table_sheet(
        workbook,
        "Table S2 SENS",
        complete_table_frame(performance, "sens"),
        {
            index
            for index, record in enumerate(sens_records)
            if record.selected
        },
    )
    write_table_sheet(
        workbook,
        "Table S3 Selected",
        selection_table_frame(selections),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def safe_output_directory(
    requested: Path,
    overwrite: bool,
) -> Path:
    output_dir = requested.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Set summary.overwrite_output to true in config.json."
            )
        generated_names = {
            "Additional_file_1_cross_device_performance.xlsx",
            "Appendix_Table_A1_window_distribution.xlsx",
            "Appendix_Table_A2_acceleration_variability.xlsx",
            "confusion_matrices",
        }
        unexpected = {
            path.name
            for path in output_dir.iterdir()
            if path.name not in generated_names
        }
        if unexpected:
            raise FileExistsError(
                f"Output directory contains unrelated items: "
                f"{sorted(unexpected)}"
            )
        for path in output_dir.iterdir():
            if path.name in {
                "Appendix_Table_A1_window_distribution.xlsx",
                "Appendix_Table_A2_acceleration_variability.xlsx",
            }:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return output_dir


def main() -> None:
    results_root = RESULTS_DIR.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    result_folders = discover_result_folders(
        results_root, INCLUDE_PARTIAL_RESULTS
    )
    if not result_folders:
        raise RuntimeError("No usable Result folders were found")
    output_dir = safe_output_directory(
        SUMMARY_DIR,
        OVERWRITE_SUMMARY,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, _ = scan_cases(results_root, result_folders)
    if not cases:
        raise RuntimeError("No valid model cases were found")
    performance = build_performance_records(cases)
    selections = build_selection_records(cases, performance)

    save_summary_workbook(
        performance,
        selections,
        output_dir
        / "Additional_file_1_cross_device_performance.xlsx",
    )
    generate_confusion_matrices(
        cases,
        selections,
        output_dir,
        SUMMARY_DPI,
    )
    generate_appendix_table_a1(
        output_dir / "Appendix_Table_A1_window_distribution.xlsx"
    )
    generate_appendix_table_a2(
        output_dir / "Appendix_Table_A2_acceleration_variability.xlsx"
    )
    print(f"Summary output: {output_dir}")


if __name__ == "__main__":
    main()
