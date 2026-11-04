"""Run every experiment selected in config.json."""

from __future__ import annotations

import os
import re
import secrets
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

# Deterministic CUDA runs need this setting before PyTorch is imported.
# setdefault leaves an existing user setting unchanged.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch.nn as nn

from common import ExperimentConfig, normalise_purity, run_model_experiment
from experiment_config import (
    CONTINUE_ON_ERROR,
    CUSTOM_COMBINATIONS,
    MODEL_DISPLAY,
    MODELS,
    NUMBER_OF_RUNS,
    PURITIES,
    RESULTS_DIR,
    RUN_PREFIX,
    WINDOWS,
    get_model_search_space,
)
from model_cnn import build_model as build_cnn
from model_cnn_bilstm import build_model as build_cnn_bilstm
from model_cnn_bilstm_attention import build_model as build_cnn_bilstm_attention


ModelFactory = Callable[[int, Mapping[str, Any]], nn.Module]

MODEL_FACTORIES: dict[str, ModelFactory] = {
    "cnn": build_cnn,
    "cnn_bilstm": build_cnn_bilstm,
    "cnn_bilstm_attention": build_cnn_bilstm_attention,
}


@dataclass(frozen=True, order=True)
class Combination:
    model: str
    window: int
    purity: str

    @property
    def task_key(self) -> str:
        return f"{self.model}_{self.window}_{self.purity}"


def make_combination(model: str, window: int, purity: str) -> Combination:
    model = str(model).strip().lower()
    window = int(window)
    purity = normalise_purity(purity)

    if model not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model in config.json: {model}")
    if window < 1:
        raise ValueError("Window length must be positive")
    return Combination(model, window, purity)


def build_plan() -> list[Combination]:
    if CUSTOM_COMBINATIONS:
        combinations = [
            make_combination(model, window, purity)
            for model, window, purity in CUSTOM_COMBINATIONS
        ]
    else:
        combinations = [
            make_combination(model, window, purity)
            for model in MODELS
            for window in WINDOWS
            for purity in PURITIES
        ]
    return list(dict.fromkeys(combinations))


def result_index(name: str, prefix: str) -> int | None:
    if name == prefix:
        return 0
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", name)
    return int(match.group(1)) if match else None


def result_name(prefix: str, index: int) -> str:
    return prefix if index == 0 else f"{prefix}_{index}"


def allocate_run_names(results_root: Path, count: int) -> list[str]:
    existing_indices: list[int] = []
    if results_root.exists():
        for path in results_root.iterdir():
            if not path.is_dir():
                continue
            index = result_index(path.name, RUN_PREFIX)
            if index is not None:
                existing_indices.append(index)

    start = max(existing_indices, default=-1) + 1
    return [result_name(RUN_PREFIX, start + offset) for offset in range(count)]


def print_plan(
    results_root: Path,
    run_names: list[str],
    plan: list[Combination],
) -> None:
    print(f"Results root: {results_root}")
    print(f"Run folders: {', '.join(run_names)}")
    print(f"Combinations per run: {len(plan)}")
    for index, combination in enumerate(plan, start=1):
        print(
            f"  {index:02d}. {MODEL_DISPLAY[combination.model]}, "
            f"{combination.window} s, {combination.purity}"
        )


def run_combination(
    combination: Combination,
    results_root: Path,
    run_name: str,
    seed: int,
) -> bool:
    config = ExperimentConfig(
        model_name=combination.model,
        window_seconds=combination.window,
        purity=combination.purity,
        results_root=results_root,
        run_name=run_name,
        seed=seed,
    )
    started = time.perf_counter()

    try:
        run_model_experiment(
            config=config,
            model_factory=MODEL_FACTORIES[combination.model],
            search_space=get_model_search_space(combination.model),
        )
    except Exception:
        print(f"Failed: {combination.task_key}")
        traceback.print_exc()
        return False

    elapsed_minutes = (time.perf_counter() - started) / 60
    print(f"Complete after {elapsed_minutes:.1f} minutes")
    return True


def main() -> int:
    results_root = RESULTS_DIR.resolve()
    plan = build_plan()
    if not plan:
        raise ValueError("The experiment plan is empty")

    run_names = allocate_run_names(results_root, NUMBER_OF_RUNS)
    print_plan(results_root, run_names, plan)
    results_root.mkdir(parents=True, exist_ok=True)

    failed = False
    for run_name in run_names:
        seed = secrets.randbits(32)
        print(f"\n{run_name} random seed: {seed}")

        for task_number, combination in enumerate(plan, start=1):
            print(
                f"\n[{run_name} {task_number}/{len(plan)}] "
                f"{combination.task_key}"
            )
            if run_combination(combination, results_root, run_name, seed):
                continue
            failed = True
            if not CONTINUE_ON_ERROR:
                return 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
