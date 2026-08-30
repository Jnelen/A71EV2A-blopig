#!/usr/bin/env python3
"""
Generate the final figures and figure-level summary tables for the GNINA
sampling benchmark.

By default, this script expects the repository layout:

analysis/
├── metadata/
│   └── annotated_complexes.csv
├── parquets/
│   └── gnina_*_redock_exh_*_poses.parquet
└── runtime/
    └── gnina_redock_runtime_summary.csv

and writes:

figures/
├── gnina_sampling_main_strategy.png
├── gnina_sampling_main_strategy_full.png
├── gnina_ensemble_size_curves.png
├── gnina_runtime_pareto_front.png
└── csvs/
    ├── input_validation.csv
    ├── main_strategy_summary.csv
    ├── ensemble_subset_success.csv
    ├── ensemble_size_summary.csv
    ├── runtime_strategy_summary.csv
    └── pareto_points.csv

The strict pose-recovery criterion is:
- ligand RMSD <= 2 Å;
- PoseBusters valid;
- LDDT-PLI >= 0.8.

Alternative thresholds may be supplied through the command line.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_RMSD_THRESHOLD = 2.0
DEFAULT_LDDT_PLI_THRESHOLD = 0.8
MAIN_TOP_N = (1, 25, 40)
BLOG_MAIN_TOP_N = (1, 25)
FULL_MAIN_TOP_N = (1, 25, 40)
PARETO_TOP_N = (1, 25)
ENSEMBLE_TOP_N = (1, 5)

EXH_COLORS = {
    8: "#F28A8A",
    16: "#E63232",
    32: "#9E1F1F",
}

STRATEGY_MARKERS = {
    "single": "o",
    "seeds": "s",
    "conformers": "^",
}

POSE_FILE_RE = re.compile(
    r"^gnina_(?P<kind>single|multi)_redock_exh_(?P<exhaustiveness>\d+)_poses\.parquet$"
)


def to_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean representations to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def prepare_pose_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns required for the sampling analysis."""
    required = {
        "complex_id",
        "lig_rmsd",
        "lddt_pli",
        "pb_valid",
        "seed",
        "input_conformer",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Pose table is missing required columns: {missing}")

    out = df.copy()
    out["complex_id"] = out["complex_id"].astype(str)
    out["pb_valid"] = to_bool(out["pb_valid"])

    if "cnn_score" not in out.columns and "rank_score" in out.columns:
        out["cnn_score"] = out["rank_score"]

    if "cnn_score" not in out.columns:
        raise ValueError(
            "Pose table lacks cnn_score (and legacy rank_score). "
            "Regenerate the pose Parquet files with GNINA score information."
        )

    numeric_columns = [
        "cnn_score",
        "lig_rmsd",
        "lddt_pli",
        "seed",
        "input_conformer",
        "source_pose",
        "sample",
        "rank",
        "cnnscore_rank",
        "exhaustiveness",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def load_pose_tables(pose_dir: Path) -> dict[tuple[str, int], pd.DataFrame]:
    """Discover and load matching single- and multi-input pose tables."""
    tables: dict[tuple[str, int], pd.DataFrame] = {}

    for path in sorted(pose_dir.glob("gnina_*_redock_exh_*_poses.parquet")):
        match = POSE_FILE_RE.match(path.name)
        if match is None:
            continue

        kind = match.group("kind")
        method = "gnina_single" if kind == "single" else "gnina_multi"
        exhaustiveness = int(match.group("exhaustiveness"))

        key = (method, exhaustiveness)
        if key in tables:
            raise ValueError(
                f"Multiple pose files found for {method}, exhaustiveness {exhaustiveness}"
            )

        tables[key] = prepare_pose_data(read_table(path))

    if not tables:
        raise FileNotFoundError(f"No GNINA pose Parquet files found in {pose_dir}")

    single_exh = {
        exhaustiveness for method, exhaustiveness in tables if method == "gnina_single"
    }
    multi_exh = {
        exhaustiveness for method, exhaustiveness in tables if method == "gnina_multi"
    }

    if single_exh != multi_exh:
        raise ValueError(
            "Single- and multi-input pose tables do not cover the same "
            f"exhaustiveness values: single={sorted(single_exh)}, "
            f"multi={sorted(multi_exh)}"
        )

    return tables


def exhaustiveness_values(
    pose_tables: dict[tuple[str, int], pd.DataFrame],
) -> list[int]:
    """Return sorted exhaustiveness values represented in the pose tables."""
    return sorted(
        exhaustiveness
        for method, exhaustiveness in pose_tables
        if method == "gnina_single"
    )


def infer_annotation_id_col(
    df: pd.DataFrame,
    requested: str | None,
) -> str:
    """Choose the complex identifier column from the annotation table."""
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Annotation ID column {requested!r} not found")
        return requested

    for candidate in ["complex_id", "complex_name", "name"]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not infer annotation complex-ID column. Use --annotation-id-col."
    )


def choose_annotation_pb_col(
    df: pd.DataFrame,
    requested: str | None,
) -> str:
    """Choose the ground-truth PoseBusters-validity column."""
    if requested and requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Annotation PB column {requested!r} not found")
        return requested

    for candidate in ["pb_valid", "pb_valid_groundtruth"]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not find a ground-truth PoseBusters-validity column. "
        "Expected pb_valid or pb_valid_groundtruth, or use --annotation-pb-col."
    )


def load_denominator_ids(
    annotation_file: Path,
    annotation_id_col: str | None,
    annotation_pb_col: str | None,
) -> list[str]:
    """Load the benchmark denominator from the annotation table."""
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")

    annotations = read_table(annotation_file).copy()
    id_col = infer_annotation_id_col(annotations, annotation_id_col)
    pb_col = choose_annotation_pb_col(annotations, annotation_pb_col)

    annotations[id_col] = annotations[id_col].astype(str)

    for column in ["fragment_screen", "artefact", pb_col]:
        if column not in annotations.columns:
            raise ValueError(f"Annotation table is missing required column {column!r}")
        annotations[column] = to_bool(annotations[column])

    keep = (
        (~annotations["fragment_screen"])
        & (~annotations["artefact"])
        & annotations[pb_col]
    )
    denominator = sorted(annotations.loc[keep, id_col].dropna().astype(str).unique())

    if not denominator:
        raise ValueError("Benchmark denominator is empty after annotation filtering")

    print(
        f"Annotation denominator: n={len(denominator)} "
        f"(~fragment_screen & ~artefact & {pb_col})"
    )
    return denominator


def validate_inputs(
    pose_tables: dict[tuple[str, int], pd.DataFrame],
    denominator_ids: Sequence[str],
) -> pd.DataFrame:
    """Summarise pose-table completeness against the benchmark denominator."""
    denominator_set = set(denominator_ids)
    rows: list[dict] = []

    for (method, exhaustiveness), df in sorted(pose_tables.items()):
        component_col = "seed" if method == "gnina_single" else "input_conformer"

        per_complex = df.groupby("complex_id").size()
        per_component = df.groupby(["complex_id", component_col]).size()
        ids = set(df["complex_id"].unique())

        row = {
            "method": method,
            "exhaustiveness": exhaustiveness,
            "rows": len(df),
            "unique_complexes": df["complex_id"].nunique(),
            "poses_per_complex_min": int(per_complex.min()),
            "poses_per_complex_max": int(per_complex.max()),
            "components_per_complex_min": int(
                df.groupby("complex_id")[component_col].nunique().min()
            ),
            "components_per_complex_max": int(
                df.groupby("complex_id")[component_col].nunique().max()
            ),
            "poses_per_component_min": int(per_component.min()),
            "poses_per_component_max": int(per_component.max()),
            "denominator_present": len(ids & denominator_set),
            "denominator_missing": len(denominator_set - ids),
        }
        rows.append(row)

        print(
            f"{method} Exh. {exhaustiveness}: "
            f"complexes={row['unique_complexes']}, "
            f"poses/complex={row['poses_per_complex_min']}-"
            f"{row['poses_per_complex_max']}, "
            f"components/complex={row['components_per_complex_min']}-"
            f"{row['components_per_complex_max']}, "
            f"denominator missing={row['denominator_missing']}"
        )

    return pd.DataFrame(rows)


def rank_within(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    rank_col: str,
) -> pd.DataFrame:
    """Rank poses by descending CNNscore within the requested groups."""
    out = df.dropna(subset=["cnn_score"]).copy()

    tie_breakers = [
        column
        for column in ["input_conformer", "seed", "sample", "source_pose"]
        if column in out.columns
    ]

    sort_columns = [*group_cols, "cnn_score", *tie_breakers]
    ascending = [True] * len(group_cols) + [False] + [True] * len(tie_breakers)

    out = out.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    )
    out[rank_col] = out.groupby(list(group_cols), sort=False).cumcount() + 1
    return out


def component_values(df: pd.DataFrame, component_col: str) -> list[int | float]:
    """Return sorted component identifiers."""
    return sorted(df[component_col].dropna().unique())


def infer_poses_per_component(
    df: pd.DataFrame,
    component_col: str,
) -> int:
    """
    Infer the number of poses produced per component.

    The minimum observed count is used so incomplete components cannot
    contribute more poses than complete ones.
    """
    counts = df.groupby(["complex_id", component_col]).size()
    if counts.empty:
        raise ValueError(f"No components found using column {component_col!r}")

    poses_per_component = int(counts.min())
    if poses_per_component <= 0:
        raise ValueError("Inferred poses per component must be positive")

    return poses_per_component


def prepare_components(
    df: pd.DataFrame,
    component_col: str,
    poses_per_component: int,
) -> pd.DataFrame:
    """Retain the top-ranked poses from each component."""
    ranked = rank_within(
        df,
        ["complex_id", component_col],
        "component_rank",
    )
    return ranked[ranked["component_rank"] <= poses_per_component].copy()


def select_subset_pool(
    component_df: pd.DataFrame,
    component_col: str,
    subset: Sequence[int | float],
) -> pd.DataFrame:
    """Pool selected components and globally rerank them by CNNscore."""
    selected = component_df[component_df[component_col].isin(subset)].copy()
    return rank_within(selected, ["complex_id"], "ensemble_rank")


def success_per_complex(
    selected: pd.DataFrame,
    denominator_ids: Sequence[str],
    top_n: int,
    rmsd_threshold: float,
    lddt_threshold: float,
    rank_col: str,
) -> pd.DataFrame:
    """Evaluate pose-recovery success for each complex."""
    top = selected[selected[rank_col] <= top_n].copy()

    top["metric_rmsd"] = top["lig_rmsd"] <= rmsd_threshold
    top["metric_rmsd_pb"] = top["metric_rmsd"] & top["pb_valid"]
    top["metric_strict"] = top["metric_rmsd_pb"] & (top["lddt_pli"] >= lddt_threshold)

    grouped = top.groupby("complex_id", as_index=True).agg(
        n_poses=("complex_id", "size"),
        rmsd=("metric_rmsd", "any"),
        rmsd_pb=("metric_rmsd_pb", "any"),
        strict=("metric_strict", "any"),
    )

    grouped = grouped.reindex(pd.Index(denominator_ids, name="complex_id"))

    for column in ["rmsd", "rmsd_pb", "strict"]:
        grouped[column] = grouped[column].eq(True)

    grouped["n_poses"] = (
        pd.to_numeric(grouped["n_poses"], errors="coerce").fillna(0).astype(int)
    )

    return grouped.reset_index()


def rates_from_per_complex(
    per_complex: pd.DataFrame,
) -> dict[str, float | int]:
    """Convert per-complex Boolean outcomes to benchmark-wide success rates."""
    n = len(per_complex)
    row: dict[str, float | int] = {
        "n_total": n,
        "n_with_predictions": int((per_complex["n_poses"] > 0).sum()),
    }

    for metric in ["rmsd", "rmsd_pb", "strict"]:
        count = int(per_complex[metric].sum())
        row[f"n_{metric}"] = count
        row[f"{metric}_pct"] = 100.0 * count / n if n else np.nan

    return row


def evaluate_selected(
    selected: pd.DataFrame,
    denominator_ids: Sequence[str],
    top_n: int,
    rmsd_threshold: float,
    lddt_threshold: float,
    rank_col: str,
) -> dict[str, float | int]:
    """Evaluate a ranked pose pool."""
    per_complex = success_per_complex(
        selected,
        denominator_ids,
        top_n=top_n,
        rmsd_threshold=rmsd_threshold,
        lddt_threshold=lddt_threshold,
        rank_col=rank_col,
    )
    return rates_from_per_complex(per_complex)


def strategy_names(
    single: pd.DataFrame,
    multi: pd.DataFrame,
) -> tuple[str, str, int, int]:
    """Infer ensemble sizes and corresponding strategy keys."""
    seeds = component_values(single, "seed")
    conformers = component_values(multi, "input_conformer")

    n_seeds = len(seeds)
    n_conformers = len(conformers)

    if n_seeds == 0 or n_conformers == 0:
        raise ValueError("Could not infer seed or conformer counts")

    return (
        f"seeds{n_seeds}",
        f"conformers{n_conformers}",
        n_seeds,
        n_conformers,
    )


def strategy_label(strategy: str) -> str:
    """Return a human-readable strategy label."""
    if strategy == "single":
        return "Single run"
    if strategy.startswith("seeds"):
        return f"{strategy.removeprefix('seeds')} seeds"
    if strategy.startswith("conformers"):
        return f"{strategy.removeprefix('conformers')} conformers"
    return strategy


def strategy_marker(strategy: str) -> str:
    """Return the plotting marker for a strategy."""
    if strategy == "single":
        return STRATEGY_MARKERS["single"]
    if strategy.startswith("seeds"):
        return STRATEGY_MARKERS["seeds"]
    if strategy.startswith("conformers"):
        return STRATEGY_MARKERS["conformers"]
    return "o"


def build_main_strategy_results(
    pose_tables: dict[tuple[str, int], pd.DataFrame],
    denominator_ids: Sequence[str],
    rmsd_threshold: float,
    lddt_threshold: float,
) -> pd.DataFrame:
    """Build the single-run, multi-seed, and multi-conformer comparison."""
    rows: list[dict] = []

    for exhaustiveness in exhaustiveness_values(pose_tables):
        single = pose_tables[("gnina_single", exhaustiveness)]
        multi = pose_tables[("gnina_multi", exhaustiveness)]

        seeds = component_values(single, "seed")
        conformers = component_values(multi, "input_conformer")
        seed_strategy = f"seeds{len(seeds)}"
        conformer_strategy = f"conformers{len(conformers)}"

        seed_poses_per_component = infer_poses_per_component(single, "seed")
        conformer_poses_per_component = infer_poses_per_component(
            multi,
            "input_conformer",
        )

        single_ranked = rank_within(
            single,
            ["complex_id", "seed"],
            "single_rank",
        )

        seed_components = prepare_components(
            single,
            "seed",
            seed_poses_per_component,
        )
        seed_pool = select_subset_pool(
            seed_components,
            "seed",
            seeds,
        )

        conformer_components = prepare_components(
            multi,
            "input_conformer",
            conformer_poses_per_component,
        )
        conformer_pool = select_subset_pool(
            conformer_components,
            "input_conformer",
            conformers,
        )

        for top_n in MAIN_TOP_N:
            replicate_rates = []
            for seed in seeds:
                selected = single_ranked[single_ranked["seed"] == seed]
                replicate_rates.append(
                    evaluate_selected(
                        selected,
                        denominator_ids,
                        top_n,
                        rmsd_threshold,
                        lddt_threshold,
                        rank_col="single_rank",
                    )
                )

            single_row: dict[str, float | int | str] = {
                "strategy": "single",
                "exhaustiveness": exhaustiveness,
                "top_n": top_n,
                "rmsd_threshold": rmsd_threshold,
                "lddt_threshold": lddt_threshold,
                "n_total": len(denominator_ids),
                "n_replicates": len(replicate_rates),
            }

            for metric in ["rmsd", "rmsd_pb", "strict"]:
                values = np.asarray(
                    [float(rate[f"{metric}_pct"]) for rate in replicate_rates],
                    dtype=float,
                )
                single_row[f"{metric}_pct"] = float(values.mean())
                single_row[f"{metric}_sd"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
                single_row[f"{metric}_min"] = float(values.min())
                single_row[f"{metric}_max"] = float(values.max())

            rows.append(single_row)

            for strategy, pooled in [
                (seed_strategy, seed_pool),
                (conformer_strategy, conformer_pool),
            ]:
                rates = evaluate_selected(
                    pooled,
                    denominator_ids,
                    top_n,
                    rmsd_threshold,
                    lddt_threshold,
                    rank_col="ensemble_rank",
                )

                row: dict[str, float | int | str] = {
                    "strategy": strategy,
                    "exhaustiveness": exhaustiveness,
                    "top_n": top_n,
                    "rmsd_threshold": rmsd_threshold,
                    "lddt_threshold": lddt_threshold,
                    "n_replicates": 1,
                    **rates,
                }

                for metric in ["rmsd", "rmsd_pb", "strict"]:
                    row[f"{metric}_sd"] = 0.0
                    row[f"{metric}_min"] = row[f"{metric}_pct"]
                    row[f"{metric}_max"] = row[f"{metric}_pct"]

                rows.append(row)

    return pd.DataFrame(rows)


def build_ensemble_subset_results(
    pose_tables: dict[tuple[str, int], pd.DataFrame],
    denominator_ids: Sequence[str],
    rmsd_threshold: float,
    lddt_threshold: float,
) -> pd.DataFrame:
    """Evaluate all subsets of seeds and starting conformers."""
    rows: list[dict] = []

    for exhaustiveness in exhaustiveness_values(pose_tables):
        single = pose_tables[("gnina_single", exhaustiveness)]
        multi = pose_tables[("gnina_multi", exhaustiveness)]

        configs = [
            ("seeds", single, "seed"),
            ("conformers", multi, "input_conformer"),
        ]

        for source, data, component_col in configs:
            poses_per_component = infer_poses_per_component(
                data,
                component_col,
            )
            prepared = prepare_components(
                data,
                component_col,
                poses_per_component,
            )
            components = component_values(prepared, component_col)

            for size in range(1, len(components) + 1):
                for subset in itertools.combinations(components, size):
                    pooled = select_subset_pool(
                        prepared,
                        component_col,
                        subset,
                    )
                    subset_label = ";".join(str(value) for value in subset)

                    for top_n in ENSEMBLE_TOP_N:
                        rates = evaluate_selected(
                            pooled,
                            denominator_ids,
                            top_n,
                            rmsd_threshold,
                            lddt_threshold,
                            rank_col="ensemble_rank",
                        )
                        rows.append(
                            {
                                "source": source,
                                "exhaustiveness": exhaustiveness,
                                "ensemble_size": size,
                                "subset": subset_label,
                                "top_n": top_n,
                                "rmsd_threshold": rmsd_threshold,
                                "lddt_threshold": lddt_threshold,
                                **rates,
                            }
                        )

    return pd.DataFrame(rows)


def summarise_ensemble_sizes(
    subset_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise strict success across subsets of equal ensemble size."""
    return subset_df.groupby(
        [
            "source",
            "exhaustiveness",
            "ensemble_size",
            "top_n",
            "rmsd_threshold",
            "lddt_threshold",
        ],
        as_index=False,
    )["strict_pct"].agg(
        mean="mean",
        std="std",
        minimum="min",
        maximum="max",
        n_subsets="count",
    )


def load_runtime_summary(path: Path) -> pd.DataFrame:
    """Load the runtime summary produced by collect_runtimes.py."""
    if not path.is_file():
        raise FileNotFoundError(f"Runtime summary does not exist: {path}")

    runtime = read_table(path).copy()

    aliases = {
        "median_runtime_seconds": "runtime_median_seconds",
        "runtime_median_seconds": "runtime_median_seconds",
        "runtime_q25_seconds": "runtime_q25_seconds",
        "runtime_q75_seconds": "runtime_q75_seconds",
        "runtime_iqr_seconds": "runtime_iqr_seconds",
    }

    for source, target in aliases.items():
        if source in runtime.columns and target not in runtime.columns:
            runtime = runtime.rename(columns={source: target})

    required = {
        "strategy",
        "exhaustiveness",
        "runtime_median_seconds",
    }
    missing = sorted(required - set(runtime.columns))
    if missing:
        raise ValueError(f"Runtime summary is missing required columns: {missing}")

    runtime["strategy"] = runtime["strategy"].astype(str)
    runtime["exhaustiveness"] = pd.to_numeric(
        runtime["exhaustiveness"],
        errors="coerce",
    )

    for column in [
        "runtime_median_seconds",
        "runtime_q25_seconds",
        "runtime_q75_seconds",
        "runtime_iqr_seconds",
    ]:
        if column in runtime.columns:
            runtime[column] = pd.to_numeric(
                runtime[column],
                errors="coerce",
            )

    return runtime


def align_runtime_strategies(
    runtime: pd.DataFrame,
    main_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict runtime rows to strategy/exhaustiveness combinations used in plots.

    Raises a clear error if required runtime combinations are missing.
    """
    combinations = (
        main_summary[["strategy", "exhaustiveness"]]
        .drop_duplicates()
        .sort_values(["strategy", "exhaustiveness"])
    )

    runtime = runtime.merge(
        combinations,
        on=["strategy", "exhaustiveness"],
        how="inner",
    )

    expected = set(
        map(
            tuple,
            combinations[["strategy", "exhaustiveness"]].to_numpy(),
        )
    )
    observed = set(
        map(
            tuple,
            runtime[["strategy", "exhaustiveness"]].to_numpy(),
        )
    )

    missing = sorted(expected - observed)
    if missing:
        formatted = ", ".join(
            f"{strategy} Exh. {int(exhaustiveness)}"
            for strategy, exhaustiveness in missing
        )
        raise ValueError(
            "Runtime summary does not contain all required strategy/"
            f"exhaustiveness combinations: {formatted}"
        )

    return runtime


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the non-dominated points when minimising x and maximising y."""
    order = np.argsort(x, kind="mergesort")
    keep = np.zeros(len(x), dtype=bool)
    best_y = -np.inf

    for index in order:
        if not np.isfinite(x[index]) or not np.isfinite(y[index]):
            continue

        if y[index] > best_y + 1e-12:
            keep[index] = True
            best_y = y[index]

    return keep


def build_pareto_points(
    main_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine performance and runtime summaries and mark Pareto-front points."""
    runtime_columns = [
        column
        for column in [
            "strategy",
            "exhaustiveness",
            "runtime_median_seconds",
            "runtime_q25_seconds",
            "runtime_q75_seconds",
            "runtime_iqr_seconds",
        ]
        if column in runtime_summary.columns
    ]

    merged = main_summary.merge(
        runtime_summary[runtime_columns],
        on=["strategy", "exhaustiveness"],
        how="left",
        validate="many_to_one",
    )

    if merged["runtime_median_seconds"].isna().any():
        missing = (
            merged.loc[
                merged["runtime_median_seconds"].isna(),
                ["strategy", "exhaustiveness"],
            ]
            .drop_duplicates()
            .values.tolist()
        )
        raise ValueError(f"Missing runtime values after merge: {missing}")

    merged["pareto"] = False

    for _, indices in merged.groupby("top_n").groups.items():
        sub = merged.loc[indices]
        mask = pareto_mask(
            sub["runtime_median_seconds"].to_numpy(float),
            sub["strict_pct"].to_numpy(float),
        )
        merged.loc[sub.index, "pareto"] = mask

    return merged


def colour_for_exhaustiveness(
    exhaustiveness: int,
    all_values: Sequence[int],
) -> str:
    """
    Return the established benchmark colour for 8/16/32.

    Additional exhaustiveness values fall back to Matplotlib's default cycle.
    """
    if exhaustiveness in EXH_COLORS:
        return EXH_COLORS[exhaustiveness]

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        return "0.4"

    index = list(all_values).index(exhaustiveness) % len(cycle)
    return cycle[index]


def style_axis(
    ax: plt.Axes,
    ylabel: str = "Success rate (%)",
) -> None:
    """Apply the shared benchmark figure style."""
    ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def metric_handles(
    rmsd_threshold: float,
    lddt_threshold: float,
) -> list[Patch]:
    """Legend handles for the two stacked bar criteria."""
    return [
        Patch(
            facecolor="gray",
            edgecolor="black",
            label=(
                f"RMSD ≤ {rmsd_threshold:g} Å & PB-valid & "
                f"LDDT-PLI ≥ {lddt_threshold:g}"
            ),
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label=f"RMSD ≤ {rmsd_threshold:g} Å & PB-valid",
        ),
    ]


def plot_main_strategy_bars(
    summary: pd.DataFrame,
    save_path: Path,
    rmsd_threshold: float,
    lddt_threshold: float,
    top_n_values: Sequence[int],
) -> None:
    """Plot the main single-run, multi-seed, and multi-conformer comparison."""
    strategies = list(
        dict.fromkeys(
            summary.sort_values(["exhaustiveness", "top_n"])["strategy"].tolist()
        )
    )
    preferred = ["single"]
    preferred.extend(
        strategy for strategy in strategies if strategy.startswith("seeds")
    )
    preferred.extend(
        strategy for strategy in strategies if strategy.startswith("conformers")
    )
    strategies = preferred

    exhaustiveness = sorted(summary["exhaustiveness"].unique())
    x = np.arange(len(strategies), dtype=float)
    width = min(0.23, 0.72 / max(1, len(exhaustiveness)))

    n_panels = len(top_n_values)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.4 * n_panels, 6.8),
        dpi=600,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for panel, (ax, top_n) in enumerate(zip(axes, top_n_values)):
        sub = summary[summary["top_n"] == top_n]

        for j, exh in enumerate(exhaustiveness):
            rows = sub[sub["exhaustiveness"] == exh].set_index("strategy")
            offset = (j - (len(exhaustiveness) - 1) / 2) * width

            for i, strategy in enumerate(strategies):
                if strategy not in rows.index:
                    continue

                row = rows.loc[strategy]
                strict = float(row["strict_pct"])
                rmsd_pb = float(row["rmsd_pb_pct"])
                extension = max(0.0, rmsd_pb - strict)
                xpos = x[i] + offset
                colour = colour_for_exhaustiveness(int(exh), exhaustiveness)

                ax.bar(
                    xpos,
                    strict,
                    width=width * 0.92,
                    color=colour,
                    edgecolor=colour,
                    linewidth=1.2,
                    zorder=3,
                )
                ax.bar(
                    xpos,
                    extension,
                    width=width * 0.92,
                    bottom=strict,
                    color="white",
                    edgecolor=colour,
                    hatch="///",
                    linewidth=1.2,
                    zorder=3,
                )

                if strict > 4:
                    ax.text(
                        xpos,
                        strict / 2,
                        f"{strict:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="white",
                        fontweight="bold",
                    )

                ax.text(
                    xpos,
                    rmsd_pb + 1.0,
                    f"{rmsd_pb:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [strategy_label(strategy) for strategy in strategies],
            fontsize=14,
        )
        ax.set_title(
            "Top 1 pose" if top_n == 1 else f"Top {top_n} poses",
            loc="center",
            fontsize=18,
            fontweight="bold",
            pad=10,
        )
        ax.set_ylim(0, 100)
        style_axis(ax)
        if panel > 0:
            ax.set_ylabel("")

    exh_handles = [
        Patch(
            facecolor=colour_for_exhaustiveness(int(exh), exhaustiveness),
            edgecolor=colour_for_exhaustiveness(int(exh), exhaustiveness),
            label=f"Exh. {int(exh)}",
        )
        for exh in exhaustiveness
    ]

    combined_handles = [
        *exh_handles,
        *metric_handles(rmsd_threshold, lddt_threshold),
    ]
    axes[0].legend(
        handles=combined_handles,
        frameon=False,
        loc="upper left",
        fontsize=13,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.55,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.94,
        bottom=0.14,
        wspace=0.12,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_size_curves(
    summary: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plot strict success as the seed/conformer ensemble size increases."""
    exhaustiveness = sorted(summary["exhaustiveness"].unique())
    max_size = int(summary["ensemble_size"].max())

    style_map = {
        "seeds": ("--", "o", "Seeds"),
        "conformers": ("-", "s", "Conformers"),
    }

    fig, axes = plt.subplots(
        1,
        len(ENSEMBLE_TOP_N),
        figsize=(15.2, 6.6),
        dpi=600,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for panel, (ax, top_n) in enumerate(zip(axes, ENSEMBLE_TOP_N)):
        sub = summary[summary["top_n"] == top_n]

        for exh in exhaustiveness:
            colour = colour_for_exhaustiveness(int(exh), exhaustiveness)

            for source in ["seeds", "conformers"]:
                line_style, marker, _ = style_map[source]
                rows = sub[
                    (sub["exhaustiveness"] == exh) & (sub["source"] == source)
                ].sort_values("ensemble_size")

                if rows.empty:
                    continue

                x = rows["ensemble_size"].to_numpy(float)
                mean = rows["mean"].to_numpy(float)
                low = rows["minimum"].to_numpy(float)
                high = rows["maximum"].to_numpy(float)

                ax.fill_between(
                    x,
                    low,
                    high,
                    color=colour,
                    alpha=0.08,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    mean,
                    linestyle=line_style,
                    marker=marker,
                    color=colour,
                    linewidth=2.2,
                    markersize=6.5,
                    zorder=3,
                )

        ax.set_xticks(range(1, max_size + 1))
        ax.set_xlabel(
            "No. seeds / starting conformers",
            fontsize=20,
            fontweight="bold",
        )
        ax.set_title(
            "Top 1 pose" if top_n == 1 else "Top 5 poses",
            loc="center",
            fontsize=20,
            fontweight="bold",
            pad=10,
        )
        ax.set_ylim(0, 100)
        style_axis(ax)
        if panel > 0:
            ax.set_ylabel("")

    exh_handles = [
        Line2D(
            [0],
            [0],
            color=colour_for_exhaustiveness(int(exh), exhaustiveness),
            linewidth=4,
            label=f"Exh. {int(exh)}",
        )
        for exh in exhaustiveness
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=(0, (3, 2)) if source == "seeds" else "-",
            marker=style_map[source][1],
            linewidth=2,
            markersize=6,
            label=style_map[source][2],
        )
        for source in ["seeds", "conformers"]
    ]

    axes[0].legend(
        handles=[*exh_handles, *source_handles],
        frameon=False,
        loc="upper left",
        fontsize=13.0,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.55,
        handlelength=2.2,
        numpoints=1,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.94,
        bottom=0.14,
        wspace=0.12,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    pareto_df: pd.DataFrame,
    save_path: Path,
    top_n_values: Sequence[int],
) -> None:
    """Plot strict success against median runtime and show the Pareto frontier."""
    exhaustiveness = sorted(pareto_df["exhaustiveness"].unique())

    strategies = list(
        dict.fromkeys(
            pareto_df.sort_values(["exhaustiveness", "top_n"])["strategy"].tolist()
        )
    )
    preferred = ["single"]
    preferred.extend(
        strategy for strategy in strategies if strategy.startswith("seeds")
    )
    preferred.extend(
        strategy for strategy in strategies if strategy.startswith("conformers")
    )
    strategies = preferred

    n_panels = len(top_n_values)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.4 * n_panels, 6.6),
        dpi=600,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for panel, (ax, top_n) in enumerate(zip(axes, top_n_values)):
        sub = pareto_df[pareto_df["top_n"] == top_n].copy()

        for strategy in strategies:
            rows = sub[sub["strategy"] == strategy].sort_values("exhaustiveness")
            if rows.empty:
                continue

            ax.plot(
                rows["runtime_median_seconds"],
                rows["strict_pct"],
                color="0.75",
                linewidth=1.2,
                zorder=1,
            )

            for _, row in rows.iterrows():
                exh = int(row["exhaustiveness"])
                ax.scatter(
                    float(row["runtime_median_seconds"]),
                    float(row["strict_pct"]),
                    marker=strategy_marker(strategy),
                    s=85,
                    color=colour_for_exhaustiveness(exh, exhaustiveness),
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=4,
                )

        frontier = sub[sub["pareto"]].sort_values("runtime_median_seconds")
        if len(frontier) >= 2:
            ax.plot(
                frontier["runtime_median_seconds"],
                frontier["strict_pct"],
                linestyle="--",
                color="black",
                linewidth=2.0,
                zorder=2,
            )

        ax.set_xlabel(
            "Median runtime per complex (s)",
            fontsize=17,
            fontweight="bold",
        )
        ax.set_title(
            "Top 1 pose" if top_n == 1 else f"Top {top_n} poses",
            loc="center",
            fontsize=18,
            fontweight="bold",
            pad=10,
        )
        ax.margins(x=0.06)
        style_axis(ax)
        ax.set_ylim(0, 100)

        if panel > 0:
            ax.set_ylabel("")

    strategy_handles = [
        Line2D(
            [0],
            [0],
            marker=strategy_marker(strategy),
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label=strategy_label(strategy),
        )
        for strategy in strategies
    ]
    exh_handles = [
        Patch(
            facecolor=colour_for_exhaustiveness(int(exh), exhaustiveness),
            edgecolor=colour_for_exhaustiveness(int(exh), exhaustiveness),
            label=f"Exh. {int(exh)}",
        )
        for exh in exhaustiveness
    ]
    frontier_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Pareto frontier",
    )

    spacer_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="",
        label="",
    )

    axes[0].legend(
        handles=[
            *exh_handles,
            spacer_handle,
            *strategy_handles,
            frontier_handle,
        ],
        frameon=False,
        loc="upper left",
        fontsize=13,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.94,
        bottom=0.14,
        wspace=0.15,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Generate the final GNINA sampling benchmark figures and "
            "figure-level summary tables."
        )
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=repo_root / "analysis" / "parquets",
        help=(
            "Directory containing GNINA pose Parquet files "
            "(default: analysis/parquets)."
        ),
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=repo_root / "analysis" / "metadata" / "annotated_complexes.csv",
        help=(
            "Complex annotation table defining the benchmark denominator "
            "(default: analysis/metadata/annotated_complexes.csv)."
        ),
    )
    parser.add_argument(
        "--runtime-file",
        type=Path,
        default=repo_root / "analysis" / "runtime" / "gnina_redock_runtime_summary.csv",
        help=(
            "Runtime summary produced by collect_runtimes.py "
            "(default: analysis/runtime/gnina_redock_runtime_summary.csv)."
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=repo_root / "figures",
        help="Output directory for figures (default: figures).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=repo_root / "figures" / "csvs",
        help="Output directory for figure-level CSVs (default: figures/csvs).",
    )
    parser.add_argument(
        "--annotation-id-col",
        default=None,
        help="Complex identifier column in the annotation table.",
    )
    parser.add_argument(
        "--annotation-pb-col",
        default="auto",
        help=(
            "Ground-truth PoseBusters-validity column. "
            "Default: auto-detect pb_valid, then pb_valid_groundtruth."
        ),
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=DEFAULT_RMSD_THRESHOLD,
        help="Ligand RMSD threshold in Å (default: 2.0).",
    )
    parser.add_argument(
        "--lddt-threshold",
        type=float,
        default=DEFAULT_LDDT_PLI_THRESHOLD,
        help="LDDT-PLI threshold for strict success (default: 0.8).",
    )
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Generate the sampling figures without the runtime/Pareto figure.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    pose_dir = args.pose_dir.expanduser().resolve()
    annotation_file = args.annotation_file.expanduser().resolve()
    runtime_file = args.runtime_file.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    csv_dir = args.csv_dir.expanduser().resolve()

    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory does not exist: {pose_dir}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pose Parquet files...")
    pose_tables = load_pose_tables(pose_dir)

    print("Loading benchmark annotations...")
    denominator_ids = load_denominator_ids(
        annotation_file,
        args.annotation_id_col,
        args.annotation_pb_col,
    )

    print("Validating processed pose tables...")
    validation = validate_inputs(pose_tables, denominator_ids)
    validation.to_csv(
        csv_dir / "input_validation.csv",
        index=False,
    )

    print("Calculating main sampling-strategy comparison...")
    main_summary = build_main_strategy_results(
        pose_tables,
        denominator_ids,
        args.rmsd_threshold,
        args.lddt_threshold,
    )
    main_summary.to_csv(
        csv_dir / "main_strategy_summary.csv",
        index=False,
    )

    print("Calculating ensemble-size curves...")
    subset_results = build_ensemble_subset_results(
        pose_tables,
        denominator_ids,
        args.rmsd_threshold,
        args.lddt_threshold,
    )
    subset_results.to_csv(
        csv_dir / "ensemble_subset_success.csv",
        index=False,
    )

    ensemble_summary = summarise_ensemble_sizes(subset_results)
    ensemble_summary.to_csv(
        csv_dir / "ensemble_size_summary.csv",
        index=False,
    )

    print("Drawing sampling figures...")
    plot_main_strategy_bars(
        main_summary,
        figure_dir / "gnina_sampling_main_strategy.png",
        args.rmsd_threshold,
        args.lddt_threshold,
        BLOG_MAIN_TOP_N,
    )
    plot_main_strategy_bars(
        main_summary,
        figure_dir / "gnina_sampling_main_strategy_full.png",
        args.rmsd_threshold,
        args.lddt_threshold,
        FULL_MAIN_TOP_N,
    )
    plot_ensemble_size_curves(
        ensemble_summary,
        figure_dir / "gnina_ensemble_size_curves.png",
    )

    if args.skip_runtime:
        print("Skipping runtime/Pareto analysis (--skip-runtime).")
    else:
        print("Loading runtime summary and building Pareto front...")
        runtime_summary = load_runtime_summary(runtime_file)
        runtime_summary = align_runtime_strategies(
            runtime_summary,
            main_summary,
        )
        runtime_summary.to_csv(
            csv_dir / "runtime_strategy_summary.csv",
            index=False,
        )

        pareto = build_pareto_points(
            main_summary,
            runtime_summary,
        )
        pareto.to_csv(
            csv_dir / "pareto_points.csv",
            index=False,
        )

        plot_pareto(
            pareto,
            figure_dir / "gnina_runtime_pareto_front.png",
            PARETO_TOP_N,
        )

    print(f"Done. Denominator n={len(denominator_ids)}")
    print(f"Figures: {figure_dir}")
    print(f"CSVs:    {csv_dir}")


if __name__ == "__main__":
    main()
