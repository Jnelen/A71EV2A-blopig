#!/usr/bin/env python3
"""
Process one GNINA redocking configuration into a pose-level table.

A configuration is defined by:
- method: gnina_single or gnina_multi
- exhaustiveness: e.g. 8, 16, or 32

For each complex, this script:
- finds the matching GNINA redock output directory;
- uses the single-conformer SDF directly, or pools and reranks the five
  multi-conformer SDFs by CNNscore;
- runs PoseBusters and OST, reusing existing outputs unless --overwrite is set;
- writes one row per ranked pose.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem, RDLogger

PARAM_DIR_RE = re.compile(
    r"^exh_(?P<exhaustiveness>\d+)_boxsize_(?P<box_size>[^_]+)_num-mode_(?P<num_modes>\d+)$"
)
DOCKED_FILE_RE = re.compile(r"_docked_(?P<input_conformer>\d+)\.sdf$")


def file_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def parse_parameter_dir(path: Path) -> dict[str, int | str] | None:
    match = PARAM_DIR_RE.match(path.name)
    if match is None:
        return None
    return {
        "exhaustiveness": int(match.group("exhaustiveness")),
        "box_size": match.group("box_size"),
        "num_modes": int(match.group("num_modes")),
    }


def rank_molecules(
    molecules: list[Chem.Mol],
    score_property: str = "CNNscore",
    descending: bool = True,
) -> list[Chem.Mol]:
    def score(mol: Chem.Mol) -> float:
        try:
            return float(mol.GetProp(score_property))
        except Exception:
            return float("-inf") if descending else float("inf")

    ranked = sorted(molecules, key=score, reverse=descending)
    for rank, mol in enumerate(ranked, start=1):
        mol.SetIntProp("rank", rank)
    return ranked


def build_multi_conformer_sdf(
    seed_dir: Path,
    output_sdf: Path,
    score_property: str = "CNNscore",
    expected_conformers: int = 5,
    poses_per_conformer: int = 8,
) -> None:
    molecules: list[Chem.Mol] = []
    seen_conformers: set[int] = set()

    for sdf_path in sorted(seed_dir.glob("*_docked_*.sdf")):
        match = DOCKED_FILE_RE.search(sdf_path.name)
        if match is None:
            continue

        input_conformer = int(match.group("input_conformer"))
        seen_conformers.add(input_conformer)

        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for source_pose, mol in enumerate(supplier, start=1):
            if source_pose > poses_per_conformer:
                break
            if mol is None:
                logging.warning("Unreadable molecule %s pose %d", sdf_path, source_pose)
                continue
            if not mol.HasProp(score_property):
                logging.warning(
                    "%s pose %d lacks %s; skipping",
                    sdf_path,
                    source_pose,
                    score_property,
                )
                continue

            mol.SetIntProp("input_conformer", input_conformer)
            mol.SetIntProp("source_pose", source_pose)
            molecules.append(mol)

    expected = set(range(1, expected_conformers + 1))
    missing = sorted(expected - seen_conformers)
    if missing:
        logging.warning("%s is missing conformer outputs: %s", seed_dir, missing)

    if not molecules:
        raise ValueError(f"No valid docked molecules found in {seed_dir}")

    ranked = rank_molecules(molecules, score_property=score_property, descending=True)
    writer = Chem.SDWriter(str(output_sdf))
    try:
        for mol in ranked:
            writer.write(mol)
    finally:
        writer.close()


def sdf_to_pose_df(sdf_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)

    for sample, mol in enumerate(supplier):
        if mol is None:
            continue

        input_conformer = (
            mol.GetIntProp("input_conformer") if mol.HasProp("input_conformer") else 1
        )

        source_pose = (
            mol.GetIntProp("source_pose") if mol.HasProp("source_pose") else sample + 1
        )

        # This is the original CNNscore-derived ordering only.
        cnnscore_rank = mol.GetIntProp("rank") if mol.HasProp("rank") else sample + 1

        rows.append(
            {
                # Pose identity / provenance
                "sample": sample,
                "input_conformer": input_conformer,
                "source_pose": source_pose,
                # Original CNNscore ordering
                "cnnscore_rank": cnnscore_rank,
                # GNINA scores
                "cnn_score": get_float_prop(mol, "CNNscore"),
                "cnn_affinity": get_float_prop(mol, "CNNaffinity"),
                "cnn_vs": get_float_prop(mol, "CNN_VS"),
                "minimized_affinity": get_float_prop(mol, "minimizedAffinity"),
            }
        )

    return pd.DataFrame(rows)


def pb_to_df(pb_path: Path) -> pd.DataFrame:
    pb = pd.read_csv(pb_path).copy()
    if "position" in pb.columns:
        pb["sample"] = pb["position"].astype(int)
    else:
        pb["sample"] = range(len(pb))

    meta_columns = {"file", "molecule", "position", "sample"}
    metric_columns = [c for c in pb.columns if c not in meta_columns]
    pb["pb_valid"] = pb[metric_columns].all(axis=1)
    return pb[["sample", "pb_valid"]]


def format_chain_mapping(mapping: dict | None) -> str | None:
    if not mapping:
        return None
    return ",".join(f"{key}:{value}" for key, value in sorted(mapping.items()))


def ost_to_df(ost_path: Path) -> pd.DataFrame:
    data = json.loads(ost_path.read_text())

    lddt_rows: list[dict] = []
    for result in data["lddt_pli"]["full_results"]:
        sample = int(result["model_ligand"].rsplit(":", 1)[1])
        value = result.get("score")
        lddt_rows.append(
            {
                "sample": sample,
                "lddt_pli": float(value) if value is not None else 0.0,
            }
        )

    rmsd_rows: list[dict] = []
    for result in data["rmsd"]["full_results"]:
        sample = int(result["model_ligand"].rsplit(":", 1)[1])
        value = result.get("score")
        rmsd_rows.append(
            {
                "sample": sample,
                "lig_rmsd": float(value) if value is not None else 99.0,
                "lddt_lp": result.get("lddt_lp"),
                "pocket_bb_rmsd": result.get("bb_rmsd"),
                "rmsd_chain_mapping": format_chain_mapping(result.get("chain_mapping")),
            }
        )

    return pd.DataFrame(lddt_rows).merge(
        pd.DataFrame(rmsd_rows), on="sample", how="outer"
    )


def run_posebusters(
    buster: PoseBusters,
    pose_sdf: Path,
    receptor_pdb: Path,
    output_csv: Path,
) -> None:
    supplier = Chem.SDMolSupplier(str(pose_sdf), removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]
    if not molecules:
        raise ValueError(f"No readable molecules in {pose_sdf}")

    results = buster.bust(molecules, None, receptor_pdb)
    results.to_csv(output_csv, index=False)


def run_ost(
    model_protein: Path,
    model_ligands: Path,
    reference_protein: Path,
    reference_ligand: Path,
    output_json: Path,
) -> None:
    command = [
        "ost",
        "compare-ligand-structures",
        "-m",
        str(model_protein),
        "-ml",
        str(model_ligands),
        "-r",
        str(reference_protein),
        "-rl",
        str(reference_ligand),
        "--lddt-pli",
        "--rmsd",
        "-o",
        str(output_json),
        "-v",
        "0",
        "--full-results",
    ]
    subprocess.run(command, check=True)


def get_float_prop(mol: Chem.Mol, property_name: str) -> float | None:
    if not mol.HasProp(property_name):
        return None

    try:
        return float(mol.GetProp(property_name))
    except (TypeError, ValueError):
        return None


def process_complex(
    complex_dir: Path,
    selected_method: str,
    selected_exhaustiveness: int,
    overwrite: bool,
    single_num_modes: int,
    multi_num_modes: int,
    expected_conformers: int,
) -> pd.DataFrame:
    complex_id = complex_dir.name
    redock_root = complex_dir / "gnina" / "redock"
    receptor_pdb = complex_dir / "prepared_inputs" / f"{complex_id}_prepared.pdb"
    reference_ligand = complex_dir / f"{complex_id}_ligand.sdf"

    if not redock_root.is_dir():
        return pd.DataFrame()
    if not file_ok(receptor_pdb) or not file_ok(reference_ligand):
        logging.warning("Missing receptor/reference inputs for %s", complex_id)
        return pd.DataFrame()

    expected_num_modes = (
        single_num_modes if selected_method == "gnina_single" else multi_num_modes
    )

    matching_parameter_dirs: list[tuple[Path, dict[str, int | str]]] = []
    for parameter_dir in sorted(
        path for path in redock_root.iterdir() if path.is_dir()
    ):
        params = parse_parameter_dir(parameter_dir)
        if params is None:
            continue
        if int(params["exhaustiveness"]) != selected_exhaustiveness:
            continue
        if int(params["num_modes"]) != expected_num_modes:
            continue
        matching_parameter_dirs.append((parameter_dir, params))

    if not matching_parameter_dirs:
        logging.warning(
            "No matching output for %s method=%s exhaustiveness=%s",
            complex_id,
            selected_method,
            selected_exhaustiveness,
        )
        return pd.DataFrame()

    buster = PoseBusters(config="dock")
    frames: list[pd.DataFrame] = []

    for parameter_dir, params in matching_parameter_dirs:
        for seed_dir in sorted(parameter_dir.glob("seed_*")):
            seed = seed_dir.name.removeprefix("seed_")

            if selected_method == "gnina_single":
                pose_sdf = seed_dir / f"{complex_id}_docked_1.sdf"
                if not file_ok(pose_sdf):
                    logging.warning("Missing single output: %s", pose_sdf)
                    continue
            else:
                pose_sdf = seed_dir / f"{complex_id}_redock_joined.sdf"
                if overwrite or not file_ok(pose_sdf):
                    build_multi_conformer_sdf(
                        seed_dir=seed_dir,
                        output_sdf=pose_sdf,
                        expected_conformers=expected_conformers,
                        poses_per_conformer=multi_num_modes,
                    )

            pb_path = seed_dir / f"PB_{complex_id}_redock.csv"
            if overwrite or not file_ok(pb_path):
                logging.info("PoseBusters: %s %s", complex_id, parameter_dir.name)
                run_posebusters(buster, pose_sdf, receptor_pdb, pb_path)

            ost_path = seed_dir / f"OST_{complex_id}_{complex_id}_redock.json"
            if overwrite or not file_ok(ost_path):
                logging.info("OST: %s %s", complex_id, parameter_dir.name)
                run_ost(
                    model_protein=receptor_pdb,
                    model_ligands=pose_sdf,
                    reference_protein=receptor_pdb,
                    reference_ligand=reference_ligand,
                    output_json=ost_path,
                )

            pose_df = sdf_to_pose_df(pose_sdf)
            pb_df = pb_to_df(pb_path)
            ost_df = ost_to_df(ost_path)

            result = pose_df.merge(pb_df, on="sample", how="left").merge(
                ost_df, on="sample", how="left"
            )
            result["complex_id"] = complex_id
            result["method"] = selected_method
            result["dock_prot"] = "redock"
            result["exhaustiveness"] = selected_exhaustiveness
            result["box_size"] = params["box_size"]
            result["num_modes_per_run"] = int(params["num_modes"])
            result["seed"] = seed
            result["pose_sdf"] = str(pose_sdf)
            frames.append(result)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one GNINA redocking method/exhaustiveness configuration"
    )
    parser.add_argument("--root", "-r", type=Path, required=True)
    parser.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "analysis" / "parquets",
        help="Output directory for pose-level Parquet files (default: analysis/parquets).",
    )
    parser.add_argument("--cores", "-c", type=int, default=4)
    parser.add_argument(
        "--method",
        required=True,
        choices=["gnina_single", "gnina_multi"],
    )
    parser.add_argument(
        "--exhaustiveness",
        required=True,
        type=int,
        help="GNINA exhaustiveness value to analyse.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single-num-modes", type=int, default=40)
    parser.add_argument("--multi-num-modes", type=int, default=8)
    parser.add_argument("--expected-conformers", type=int, default=5)
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a CSV copy of the pose-level table.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("posebusters").setLevel(logging.WARNING)
    RDLogger.DisableLog("rdApp.*")

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    complex_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith("slurm")
        and (path / "gnina" / "redock").is_dir()
    )
    logging.info(
        "Found %d complexes for %s exhaustiveness=%d",
        len(complex_dirs),
        args.method,
        args.exhaustiveness,
    )

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = {
            executor.submit(
                process_complex,
                complex_dir,
                args.method,
                args.exhaustiveness,
                args.overwrite,
                args.single_num_modes,
                args.multi_num_modes,
                args.expected_conformers,
            ): complex_dir.name
            for complex_dir in complex_dirs
        }

        for future in as_completed(futures):
            complex_id = futures[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                logging.info("Finished %s", complex_id)
            except Exception:
                logging.exception("Failed %s", complex_id)

    final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not final.empty:
        preferred_order = [
            "complex_id",
            "dock_prot",
            "method",
            "exhaustiveness",
            "box_size",
            "num_modes_per_run",
            "seed",
            "input_conformer",
            "source_pose",
            "sample",
            "cnnscore_rank",
            "cnn_score",
            "cnn_affinity",
            "cnn_vs",
            "minimized_affinity",
            "lig_rmsd",
            "lddt_pli",
            "pb_valid",
            "lddt_lp",
            "pocket_bb_rmsd",
            "rmsd_chain_mapping",
            "pose_sdf",
        ]
        existing = [column for column in preferred_order if column in final.columns]
        extra = [column for column in final.columns if column not in preferred_order]
        final = final[existing + extra]

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_redock_exh_{args.exhaustiveness}_poses"
    parquet_path = args.outdir / f"{stem}.parquet"
    final.to_parquet(parquet_path, index=False)
    logging.info("Wrote %s", parquet_path)

    if args.write_csv:
        csv_path = args.outdir / f"{stem}.csv"
        final.to_csv(csv_path, index=False)
        logging.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
