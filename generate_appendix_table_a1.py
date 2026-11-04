"""Generate Appendix Table A1 from the paired three-second windows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from common import ExperimentConfig, load_dataset_bundle
from experiment_config import RESULTS_DIR, SUMMARY_DIR


ACTIVITY_ORDER = ("Lie", "Sit", "Stand", "Walk", "Stairs", "Run")
ACTIVITY_DISPLAY = {
    "Lie": "Lying",
    "Sit": "Sitting",
    "Stand": "Standing",
    "Walk": "Walking",
    "Stairs": "Stair climbing",
    "Run": "Running",
}
DEVICE_ORDER = ("sens", "axivity")
DATASET_ORDER = ("training", "test")
DATASET_DISPLAY = {
    "training": "Training dataset\nn (%)",
    "test": "Test dataset\nn (%)",
}
WINDOW_SECONDS = 3
PURITY = "mixed"
OUTPUT_PATH = SUMMARY_DIR / "Appendix_Table_A1_window_distribution.xlsx"


def calculate_table() -> pd.DataFrame:
    """Count paired windows and calculate percentages within each dataset."""

    config = ExperimentConfig(
        model_name="cnn",
        window_seconds=WINDOW_SECONDS,
        purity=PURITY,
        results_root=RESULTS_DIR,
        run_name="appendix_table_a1",
    )
    bundle = load_dataset_bundle(config)
    labels_by_dataset = {
        "training": pd.Series(bundle.train_y.astype(str)),
        "test": pd.Series(bundle.test_y.astype(str)),
    }

    records: list[dict[str, str | int | float]] = []
    for dataset in DATASET_ORDER:
        labels = labels_by_dataset[dataset]
        counts = labels.value_counts()
        total = int(len(labels))
        unexpected = sorted(set(counts.index) - set(ACTIVITY_ORDER))
        if unexpected:
            raise ValueError(
                f"Unexpected labels in {dataset} windows: {unexpected}"
            )
        for device in DEVICE_ORDER:
            for activity in ACTIVITY_ORDER:
                count = int(counts.get(activity, 0))
                records.append(
                    {
                        "activity": activity,
                        "device": device,
                        "dataset": dataset,
                        "count": count,
                        "percentage": count / total if total else 0.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def formatted_count(count: int, percentage: float) -> str:
    return f"{count:,} ({percentage * 100:.1f})"


def write_workbook(summary: pd.DataFrame, output_path: Path) -> None:
    workbook = Workbook()
    table_sheet = workbook.active
    table_sheet.title = "Table A1"
    table_sheet.sheet_view.showGridLines = False

    table_sheet.append(
        ["Posture and movement types", "SENS", "", "Axivity", ""]
    )
    table_sheet.append(
        [
            "",
            DATASET_DISPLAY["training"],
            DATASET_DISPLAY["test"],
            DATASET_DISPLAY["training"],
            DATASET_DISPLAY["test"],
        ]
    )
    table_sheet.merge_cells("A1:A2")
    table_sheet.merge_cells("B1:C1")
    table_sheet.merge_cells("D1:E1")

    indexed = summary.set_index(["activity", "device", "dataset"])
    for activity in ACTIVITY_ORDER:
        row = [ACTIVITY_DISPLAY[activity]]
        for device in DEVICE_ORDER:
            for dataset in DATASET_ORDER:
                record = indexed.loc[(activity, device, dataset)]
                row.append(
                    formatted_count(
                        int(record["count"]), float(record["percentage"])
                    )
                )
        table_sheet.append(row)

    total_row = ["Total"]
    for device in DEVICE_ORDER:
        for dataset in DATASET_ORDER:
            subset = summary[
                (summary["device"] == device)
                & (summary["dataset"] == dataset)
            ]
            total_row.append(formatted_count(int(subset["count"].sum()), 1.0))
    table_sheet.append(total_row)

    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    header_font = Font(name="Times New Roman", size=10, bold=True)
    body_font = Font(name="Times New Roman", size=10)
    centered = Alignment(
        horizontal="center", vertical="center", wrap_text=True
    )
    for row in table_sheet.iter_rows():
        for cell in row:
            cell.border = border
            cell.alignment = centered
            cell.font = header_font if cell.row <= 2 else body_font
            if cell.row <= 2:
                cell.fill = header_fill
    for cell in table_sheet[table_sheet.max_row]:
        cell.font = header_font
    table_sheet.row_dimensions[2].height = 30
    table_sheet.column_dimensions["A"].width = 29
    for column_index in range(2, 6):
        table_sheet.column_dimensions[get_column_letter(column_index)].width = 21
    table_sheet.freeze_panes = "B3"

    audit_sheet = workbook.create_sheet("Window counts")
    audit_columns = ["activity", "device", "dataset", "count", "percentage"]
    audit_sheet.append(audit_columns)
    ordered = summary.copy()
    ordered["activity"] = pd.Categorical(
        ordered["activity"], categories=ACTIVITY_ORDER, ordered=True
    )
    ordered["device"] = pd.Categorical(
        ordered["device"], categories=DEVICE_ORDER, ordered=True
    )
    ordered["dataset"] = pd.Categorical(
        ordered["dataset"], categories=DATASET_ORDER, ordered=True
    )
    ordered = ordered.sort_values(["device", "dataset", "activity"])
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
            if cell.column == 5:
                cell.number_format = "0.0%"
    for column_index, width in enumerate((16, 12, 12, 12, 14), start=1):
        audit_sheet.column_dimensions[get_column_letter(column_index)].width = width

    workbook.properties.title = "Appendix Table A1 window distribution"
    workbook.properties.subject = (
        "Distribution of paired three-second windows by activity and dataset"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    summary = calculate_table()
    resolved_output = Path(output_path).resolve()
    write_workbook(summary, resolved_output)
    print(f"Appendix Table A1: {resolved_output}")
    return resolved_output


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
