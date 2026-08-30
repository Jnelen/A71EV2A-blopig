#!/usr/bin/env python3
"""
Collect and summarise GNINA redocking runtimes.

The script scans ``gnina/redock`` output directories and parses the
``ELAPSED_SECONDS`` values written by ``dock_complexes.py``. The number of
ligand inputs is inferred from numbered log files such as ``<complex>_1.log``
and ``<complex>_5.log``.

Three runtime strategies are summarised automatically:

- ``single``: one ligand input and one random seed;
- ``seedsN``: N independent single-input seed runs, summed per complex;
- ``conformersN``: N ligand-input runs within one seed, summed per complex.

For multi-run strategies, the summed runtime represents total compute rather
than necessarily wall-clock time, since independent runs may be executed in
parallel.

Outputs
-------
``gnina_redock_runtime_summary.csv``
    Configuration-level summary used for reporting. Includes median, 25th and
    75th percentiles, IQR, mean, standard deviation, minimum, maximum and total
    runtime.

``gnina_redock_runtime_invocations.csv``
    One row per GNINA invocation. This preserves the raw timing provenance and
    is useful for downstream plotting or re-analysis.

The optional ``--write-strategy-table`` flag additionally writes the
per-complex strategy runtimes used to construct the summary.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd


PARAM_DIR_RE = re.compile(
    r"^exh_(?P<exhaustiveness>\d+)_boxsize_(?P<box_size>[^_]+)_num-mode_(?P<num_modes>\d+)$"
)
LOG_FILE_RE = re.compile(r"_(?P<input_index>\d+)\.log$")
ELAPSED_RE = re.compile(
    r"^ELAPSED_SECONDS=(?P<seconds>[0-9]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)


def parse_parameter_dir(path: Path) -> dict[str, int | str] | None:
    """Parse a docking parameter-directory name."""
    match = PARAM_DIR_RE.match(path.name)
    if match is None:
        return None

    return {
        "exhaustiveness": int(match.group("exhaustiveness")),
        "box_size": match.group("box_size"),
        "num_modes_per_run": int(match.group("num_modes")),
    }


def read_elapsed_seconds(log_path: Path) -> float | None:
    """Return the final ``ELAPSED_SECONDS`` value from a docking log."""
    text = log_path.read_text(errors="replace")
    matches = list(ELAPSED_RE.finditer(text))
    if not matches:
        return None

    # Use the last value if a log contains repeated timing lines.
    return float(matches[-1].group("seconds"))


def collect_invocations(root: Path) -> pd.DataFrame:
    """Collect one row per GNINA invocation."""
    rows: list[dict] = []

    for complex_dir in sorted(root.iterdir()):
        if not complex_dir.is_dir() or complex_dir.name.startswith("slurm"):
            continue

        redock_root = complex_dir / "gnina" / "redock"
        if not redock_root.is_dir():
            continue

        for parameter_dir in sorted(
            path for path in redock_root.iterdir() if path.is_dir()
        ):
            params = parse_parameter_dir(parameter_dir)
            if params is None:
                continue

            for seed_dir in sorted(parameter_dir.glob("seed_*")):
                seed = seed_dir.name.removeprefix("seed_")

                log_records: list[tuple[Path, int]] = []
                for log_path in sorted(seed_dir.glob(f"{complex_dir.name}_*.log")):
                    match = LOG_FILE_RE.search(log_path.name)
                    if match is None:
                        continue
                    log_records.append((log_path, int(match.group("input_index"))))

                if not log_records:
                    continue

                input_indices = sorted({index for _, index in log_records})
                num_inputs = max(input_indices)
                inputs_complete = input_indices == list(range(1, num_inputs + 1))

                for log_path, input_index in log_records:
                    elapsed = read_elapsed_seconds(log_path)
                    rows.append(
                        {
                            "complex_id": complex_dir.name,
                            "exhaustiveness": int(params["exhaustiveness"]),
                            "box_size": params["box_size"],
                            "num_modes_per_run": int(params["num_modes_per_run"]),
                            "seed": seed,
                            "input_index": input_index,
                            "num_inputs": num_inputs,
                            "inputs_complete": inputs_complete,
                            "elapsed_seconds": elapsed,
                            "timing_found": elapsed is not None,
                            "log_path": str(log_path.relative_to(root)),
                        }
                    )

    if not rows:
        return pd.DataFrame()

    invocations = pd.DataFrame(rows)
    invocations["seed"] = invocations["seed"].astype(str)
    return invocations.sort_values(
        [
            "complex_id",
            "exhaustiveness",
            "box_size",
            "num_modes_per_run",
            "seed",
            "input_index",
        ]
    ).reset_index(drop=True)


def build_strategy_runtimes(invocations: pd.DataFrame) -> pd.DataFrame:
    """Build per-complex runtimes for single, seed and conformer strategies."""
    if invocations.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    config_columns = ["exhaustiveness", "box_size", "num_modes_per_run"]

    # First collapse each seed directory into one run. For a multi-input seed,
    # this is the sum of its component conformer/input runs.
    seed_group_columns = ["complex_id", *config_columns, "seed"]

    for keys, group in invocations.groupby(seed_group_columns, dropna=False):
        complex_id, exhaustiveness, box_size, num_modes_per_run, seed = keys
        valid = group["elapsed_seconds"].dropna()
        num_inputs = int(group["num_inputs"].max())
        observed_inputs = int(group["input_index"].nunique())
        inputs_complete = (
            bool(group["inputs_complete"].all()) and observed_inputs == num_inputs
        )
        timing_complete = len(valid) == len(group)
        complete = inputs_complete and timing_complete

        if num_inputs == 1:
            strategy = "single"
        else:
            strategy = f"conformers{num_inputs}"

        rows.append(
            {
                "complex_id": complex_id,
                "strategy": strategy,
                "exhaustiveness": int(exhaustiveness),
                "box_size": box_size,
                "num_modes_per_run": int(num_modes_per_run),
                "seed": str(seed),
                "num_component_runs": len(group),
                "num_inputs": num_inputs,
                "timing_complete": timing_complete,
                "inputs_complete": inputs_complete,
                "complete": complete,
                "runtime_seconds": valid.sum() if complete else pd.NA,
            }
        )

    seed_level = pd.DataFrame(rows)

    # Each complete single-input seed is itself a 'single' strategy runtime.
    strategy_rows = [seed_level]

    # If several single-input seeds are present for the same complex/config,
    # also construct the corresponding seedsN strategy by summing them.
    single = seed_level[seed_level["strategy"] == "single"].copy()
    if not single.empty:
        ensemble_rows: list[dict] = []
        group_columns = ["complex_id", *config_columns]

        for keys, group in single.groupby(group_columns, dropna=False):
            complex_id, exhaustiveness, box_size, num_modes_per_run = keys
            n_seeds = int(group["seed"].nunique())

            # A one-seed group is already represented by the 'single' rows.
            if n_seeds <= 1:
                continue

            complete = bool(group["complete"].all())
            valid = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
            timing_complete = complete and len(valid) == n_seeds

            ensemble_rows.append(
                {
                    "complex_id": complex_id,
                    "strategy": f"seeds{n_seeds}",
                    "exhaustiveness": int(exhaustiveness),
                    "box_size": box_size,
                    "num_modes_per_run": int(num_modes_per_run),
                    "seed": pd.NA,
                    "num_component_runs": n_seeds,
                    "num_inputs": 1,
                    "timing_complete": timing_complete,
                    "inputs_complete": True,
                    "complete": timing_complete,
                    "runtime_seconds": valid.sum() if timing_complete else pd.NA,
                }
            )

        if ensemble_rows:
            strategy_rows.append(pd.DataFrame(ensemble_rows))

    strategy = pd.concat(strategy_rows, ignore_index=True)
    return strategy.sort_values(
        ["strategy", "exhaustiveness", "box_size", "complex_id", "seed"],
        na_position="last",
    ).reset_index(drop=True)


def summarise_runtimes(strategy: pd.DataFrame) -> pd.DataFrame:
    """Summarise complete strategy runtimes across complexes/configurations."""
    if strategy.empty:
        return pd.DataFrame()

    complete = strategy[strategy["complete"]].copy()
    if complete.empty:
        return pd.DataFrame()

    group_columns = [
        "strategy",
        "exhaustiveness",
        "box_size",
        "num_modes_per_run",
    ]

    summary = (
        complete.groupby(group_columns, dropna=False)["runtime_seconds"]
        .agg(
            n_runtime="count",
            runtime_median_seconds="median",
            runtime_q25_seconds=lambda values: values.quantile(0.25),
            runtime_q75_seconds=lambda values: values.quantile(0.75),
            runtime_mean_seconds="mean",
            runtime_std_seconds="std",
            min_runtime_seconds="min",
            max_runtime_seconds="max",
            total_runtime_seconds="sum",
        )
        .reset_index()
    )

    summary["runtime_iqr_seconds"] = (
        summary["runtime_q75_seconds"] - summary["runtime_q25_seconds"]
    )

    column_order = [
        "strategy",
        "exhaustiveness",
        "box_size",
        "num_modes_per_run",
        "n_runtime",
        "runtime_median_seconds",
        "runtime_q25_seconds",
        "runtime_q75_seconds",
        "runtime_iqr_seconds",
        "runtime_mean_seconds",
        "runtime_std_seconds",
        "min_runtime_seconds",
        "max_runtime_seconds",
        "total_runtime_seconds",
    ]
    return (
        summary[column_order]
        .sort_values(["strategy", "exhaustiveness", "box_size"])
        .reset_index(drop=True)
    )


def to_bool(series: pd.Series) -> pd.Series:
    """Convert common Boolean-like values to a Boolean Series."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def load_benchmark_ids(annotation_file: Path) -> set[str]:
    """Load the benchmark denominator from the complex annotation table."""
    if annotation_file.suffix.lower() in {".parquet", ".pq"}:
        annotations = pd.read_parquet(annotation_file)
    else:
        annotations = pd.read_csv(annotation_file)

    id_col = next(
        (
            column
            for column in ["complex_id", "complex_name", "name"]
            if column in annotations.columns
        ),
        None,
    )
    if id_col is None:
        raise ValueError(
            "Could not infer the annotation complex-ID column. Expected "
            "complex_id, complex_name, or name."
        )

    pb_col = next(
        (
            column
            for column in ["pb_valid", "pb_valid_groundtruth"]
            if column in annotations.columns
        ),
        None,
    )
    if pb_col is None:
        raise ValueError("Could not find a ground-truth PoseBusters validity column.")

    for column in ["fragment_screen", "artefact", pb_col]:
        if column not in annotations.columns:
            raise ValueError(f"Annotation table is missing required column {column!r}")
        annotations[column] = to_bool(annotations[column])

    keep = (
        (~annotations["fragment_screen"])
        & (~annotations["artefact"])
        & annotations[pb_col]
    )
    return set(annotations.loc[keep, id_col].dropna().astype(str))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect and summarise GNINA redocking runtimes."
    )
    parser.add_argument(
        "--root",
        "-r",
        type=Path,
        required=True,
        help="Dataset root containing one subdirectory per complex.",
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=repo_root / "analysis" / "runtime",
        help="Directory in which to write runtime CSV files (default: analysis/runtime).",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=repo_root / "analysis" / "metadata" / "annotated_complexes.csv",
        help=(
            "Complex annotation table used to restrict runtimes to the benchmark "
            "denominator (default: analysis/metadata/annotated_complexes.csv)."
        ),
    )
    parser.add_argument(
        "--all-complexes",
        action="store_true",
        help="Do not restrict runtimes using the annotation table.",
    )
    parser.add_argument(
        "--write-strategy-table",
        action="store_true",
        help="Also write the per-complex strategy runtime table.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    invocations = collect_invocations(root)
    if invocations.empty:
        raise RuntimeError(f"No GNINA redocking runtime logs found in {root}")

    if not args.all_complexes:
        annotation_file = args.annotation_file.expanduser().resolve()
        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation file does not exist: {annotation_file}. "
                "Supply --annotation-file or use --all-complexes."
            )
        benchmark_ids = load_benchmark_ids(annotation_file)
        invocations = invocations[
            invocations["complex_id"].astype(str).isin(benchmark_ids)
        ].copy()
        logging.info(
            "Restricted runtime analysis to %d benchmark complexes",
            invocations["complex_id"].nunique(),
        )
        if invocations.empty:
            raise RuntimeError(
                "No runtime rows remain after applying the benchmark annotation filter"
            )

    strategy = build_strategy_runtimes(invocations)
    summary = summarise_runtimes(strategy)
    if summary.empty:
        raise RuntimeError("No complete runtime strategies were available to summarise")

    args.outdir.mkdir(parents=True, exist_ok=True)

    invocation_path = args.outdir / "gnina_redock_runtime_invocations.csv"
    summary_path = args.outdir / "gnina_redock_runtime_summary.csv"

    invocations.to_csv(invocation_path, index=False)
    summary.to_csv(summary_path, index=False)

    logging.info("Wrote %s", invocation_path)
    logging.info("Wrote %s", summary_path)

    if args.write_strategy_table:
        strategy_path = args.outdir / "gnina_redock_runtime_strategies.csv"
        strategy.to_csv(strategy_path, index=False)
        logging.info("Wrote %s", strategy_path)

    missing_timings = int((~invocations["timing_found"]).sum())
    incomplete_strategies = int((~strategy["complete"]).sum())

    logging.info(
        "Collected %d GNINA invocations; %d missing timings; "
        "%d incomplete strategy rows.",
        len(invocations),
        missing_timings,
        incomplete_strategies,
    )


if __name__ == "__main__":
    main()
