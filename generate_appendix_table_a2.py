"""Generate Appendix Table A2 from the paired accelerometer windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from common import ExperimentConfig, load_dataset_bundle
from experiment_config import RESULTS_DIR, SUMMARY_DIR


ACTIVITY_ORDER = ("Lie", "Sit", "Stand", "Walk", "Stairs", "Run")
DEVICE_ORDER = ("axivity", "sens")
DEVICE_DISPLAY = {"axivity": "Axivity", "sens": "SENS"}
AXES = ("SDx", "SDy", "SDz")
WINDOW_SECONDS = 3
PURITY = "mixed"
OUTPUT_PATH = SUMMARY_DIR / "Appendix_Table_A2_acceleration_variability.xlsx"


def window_standard_deviations(
    values: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    device: str,
) -> pd.DataFrame:
    """Calculate sample SDs for each axis in every window."""
    standard_deviations = values.std(axis=2, ddof=1)
    return pd.DataFrame(
        {
            "participant": participant_ids.astype(str),
            "activity": labels.astype(str),
            "device": device,
            "SDx": standard_deviations[:, 0],
            "SDy": standard_deviations[:, 1],
            "SDz": standard_deviations[:, 2],
        }
    )


def calculate_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = ExperimentConfig(
        model_name="cnn",
        window_seconds=WINDOW_SECONDS,
        purity=PURITY,
        results_root=RESULTS_DIR,
        run_name="appendix_table_a2",
    )
    bundle = load_dataset_bundle(config)
    labels = np.concatenate((bundle.train_y, bundle.test_y))
    participant_ids = np.concatenate((bundle.train_pid, bundle.test_pid))

    window_frames = []
    for device in DEVICE_ORDER:
        values = np.concatenate(
            (bundle.train_x[device], bundle.test_x[device]), axis=0
        )
        window_frames.append(
            window_standard_deviations(
                values, labels, participant_ids, device
            )
        )
    windows = pd.concat(window_frames, ignore_index=True)

    # Each participant contributes one median per activity and device. The
    # table is the unweighted mean of those participant-level medians.
    participant_medians = (
        windows.groupby(
            ["participant", "activity", "device"], as_index=False
        )
        .agg(
            SDx=("SDx", "median"),
            SDy=("SDy", "median"),
            SDz=("SDz", "median"),
            n_windows=("SDx", "size"),
        )
    )
    summary = (
        participant_medians.groupby(["activity", "device"], as_index=False)[
            list(AXES)
        ]
        .mean()
    )
    return summary, participant_medians


def write_workbook(
    summary: pd.DataFrame,
    participant_medians: pd.DataFrame,
    output_path: Path,
) -> None:
    workbook = Workbook()
    table_sheet = workbook.active
    table_sheet.title = "Table A2"
    table_sheet.sheet_view.showGridLines = False

    table_sheet.append(["Activity", "SDx", "", "SDy", "", "SDz", ""])
    table_sheet.append(
        ["", "Axivity", "SENS", "Axivity", "SENS", "Axivity", "SENS"]
    )
    table_sheet.merge_cells("A1:A2")
    table_sheet.merge_cells("B1:C1")
    table_sheet.merge_cells("D1:E1")
    table_sheet.merge_cells("F1:G1")

    indexed = summary.set_index(["activity", "device"])
    for activity in ACTIVITY_ORDER:
        row: list[str | float] = [activity]
        for axis in AXES:
            for device in DEVICE_ORDER:
                row.append(float(indexed.loc[(activity, device), axis]))
        table_sheet.append(row)

    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    header_font = Font(name="Times New Roman", size=10, bold=True)
    body_font = Font(name="Times New Roman", size=10)
    centered = Alignment(horizontal="center", vertical="center")
    for row in table_sheet.iter_rows():
        for cell in row:
            cell.border = border
            cell.alignment = centered
            cell.font = header_font if cell.row <= 2 else body_font
            if cell.row <= 2:
                cell.fill = header_fill
    for row_index in range(3, table_sheet.max_row + 1):
        for column_index in range(2, table_sheet.max_column + 1):
            table_sheet.cell(row_index, column_index).number_format = "0.0000"
    table_sheet.column_dimensions["A"].width = 16
    for column_index in range(2, 8):
        table_sheet.column_dimensions[get_column_letter(column_index)].width = 13
    table_sheet.freeze_panes = "B3"

    audit_sheet = workbook.create_sheet("Participant medians")
    audit_columns = [
        "participant",
        "activity",
        "device",
        "SDx",
        "SDy",
        "SDz",
        "n_windows",
    ]
    audit_sheet.append(audit_columns)
    ordered = participant_medians.copy()
    ordered["activity"] = pd.Categorical(
        ordered["activity"], categories=ACTIVITY_ORDER, ordered=True
    )
    ordered["device"] = pd.Categorical(
        ordered["device"], categories=DEVICE_ORDER, ordered=True
    )
    ordered = ordered.sort_values(["participant", "activity", "device"])
    for record in ordered.loc[:, audit_columns].itertuples(
        index=False, name=None
    ):
        audit_sheet.append(list(record))
    audit_sheet.freeze_panes = "A2"
    audit_sheet.auto_filter.ref = audit_sheet.dimensions
    for cell in audit_sheet[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = centered
        cell.border = border
    for row in audit_sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = body_font
            cell.border = border
            if cell.column >= 4 and cell.column <= 6:
                cell.number_format = "0.000000"
    widths = (15, 14, 12, 13, 13, 13, 12)
    for column_index, width in enumerate(widths, start=1):
        audit_sheet.column_dimensions[
            get_column_letter(column_index)
        ].width = width

    workbook.properties.title = "Appendix Table A2 acceleration variability"
    workbook.properties.subject = (
        "Participant-level summaries of window-level acceleration variability"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    summary, participant_medians = calculate_table()
    resolved_output = Path(output_path).resolve()
    write_workbook(summary, participant_medians, resolved_output)
    print(
        f"Appendix Table A2: {resolved_output}\n"
        f"Participants: {participant_medians['participant'].nunique()}"
    )
    return resolved_output


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
