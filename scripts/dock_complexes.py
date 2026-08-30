#!/usr/bin/env python3
"""
Simple bulk docking runner for prepared fragalysis data.

Author: Jochem Nelen (jochem.nelen@stats.ox.ac.uk)

Expected layout (root directory):
 root/
   COMPLEX1/
     COMPLEX1_ligand.sdf # (used as docking input)
     prepared_inputs/
         COMPLEX1_prepared.pdb(qt)
         COMPLEX1_ligand_prepared.sdf
   COMPLEX2/
     ...

Outputs per complex:
  COMPLEX1/method/seed/COMPLEX1_docked.sdf
  COMPLEX1/method/seed/COMPLEX1.log
"""

import subprocess
from pathlib import Path
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import time
import shlex
from typing import Optional, Union, Tuple


def compute_centroid(sdf_path: Path) -> Tuple[float, float, float]:
    """
    Compute the 3D centroid of the first conformer in an SDF file.

    Args:
        sdf_path: Path to the ligand SDF file.

    Returns:
        A tuple of (x, y, z) centroid coordinates.
    """
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        raise RuntimeError(f"RDKit could not read ligand SDF: {sdf_path}")
    conf = mol.GetConformer(0)
    c = rdMolTransforms.ComputeCentroid(conf)
    return float(c.x), float(c.y), float(c.z)


def resolve_receptor(
    protein_path: Optional[str],
    prepared_inputs: Path,
    complex_id: str,
    prot_ext: str,
) -> Tuple[Path, str]:
    """
    Resolve the receptor path and config label.

    protein_path can be:
        - None: use the default redocking receptor
        - "redock": use the default redocking receptor
        - a file path with extension: use that file directly
        - a prefix: find a matching file in prepared_inputs
    """
    if protein_path is None or protein_path == "redock":
        receptor = prepared_inputs / f"{complex_id}_prepared.{prot_ext}"
        config_label = "redock"

    else:
        spec = Path(protein_path).expanduser()

        if spec.suffix:
            if not spec.is_file():
                raise FileNotFoundError(f"Protein file not found: {spec}")

            if spec.stat().st_size == 0:
                raise FileNotFoundError(f"Protein file is empty: {spec}")

            receptor = spec.resolve()
            config_label = receptor.stem

        else:
            matches = sorted(prepared_inputs.glob(f"{protein_path}*.{prot_ext}"))

            if not matches:
                raise FileNotFoundError(
                    f"No receptor found matching {protein_path}_*.{prot_ext} "
                    f"in {prepared_inputs}"
                )

            receptor = matches[0]
            config_label = protein_path

    if not receptor.is_file():
        raise FileNotFoundError(f"Missing receptor: {receptor}")

    if receptor.stat().st_size == 0:
        raise FileNotFoundError(f"Receptor file is empty: {receptor}")

    return receptor, config_label


def run_docking_on_complex(
    folder: Path,
    complex_id: str,
    method: Path,
    exhaustiveness: int = 32,
    input_num: int = 1,
    num_modes: int = 40,
    size: Union[float, str] = 25,
    seed: int = 1,
    scoring: Optional[str] = None,
    cnn_model: Optional[str] = None,
    protein_path: Optional[str] = None,
    x_coord: Optional[float] = None,
    y_coord: Optional[float] = None,
    z_coord: Optional[float] = None,
    overwrite: bool = False,
) -> Tuple[int, Path, Path]:
    """
    Run docking for a single complex folder.

    Args:
        folder: Complex folder containing the inputs and outputs.
        complex_id: Complex identifier, usually the folder name.
        method: Path to the docking executable
        exhaustiveness: Docking exhaustiveness parameter.
        input_num: Ligand input index for cases with multiple prepared ligands.
        num_modes: Number of docking poses to request.
        size: Box size, or an autobox label string containing "autobox".
        seed: Optional random seed.
        scoring: Optional scoring function name for non-Vina methods.
        protein_path: Optional explicit protein path override.
        x_coord: Optional docking box center X.
        y_coord: Optional docking box center Y.
        z_coord: Optional docking box center Z.
        overwrite: Perform the docking calculation, even if there already is an output.

    Returns:
        A tuple of (return_code, output_directory, log_file_path).
    """

    # input files (based on your prepare script naming)
    prepared_inputs = folder / "prepared_inputs"

    selected_method = method.stem.lower()

    if selected_method == "gnina" and cnn_model not in (None, "default"):
        cnn_name = Path(cnn_model).stem
        selected_method = f"{selected_method}_{cnn_name}"

    prot_ext = "pdb"
    lig_ext = "sdf"

    receptor, config_label = resolve_receptor(
        protein_path=protein_path,
        prepared_inputs=prepared_inputs,
        complex_id=complex_id,
        prot_ext=prot_ext,
    )

    ligand_input = (
        prepared_inputs / f"{complex_id}_ligand_prepared_{input_num}.{lig_ext}"
    )
    ligand_crystal = folder / f"{complex_id}_ligand.sdf"

    if not prepared_inputs.exists():
        raise FileNotFoundError(
            f"Missing prepared inputs directory at {prepared_inputs}. Have the files been prepared?"
        )
    if not ligand_input.exists():
        raise FileNotFoundError(f"Missing ligand input: {ligand_input}")

    # Determine output path
    parts = [config_label] if config_label else []

    if x_coord is not None and y_coord is not None and z_coord is not None:
        parts.append(f"{x_coord:g}_{y_coord:g}_{z_coord:g}")

    config_label = "_".join(parts)

    out_dir = (
        folder
        / selected_method
        / config_label
        / f"exh_{exhaustiveness}_boxsize_{size.replace('_', '')}_num-mode_{num_modes}"
        / f"seed_{seed}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"{complex_id}_{input_num}.log"

    if not ligand_crystal.exists():
        raise FileNotFoundError(f"Missing crystal ligand input: {ligand_crystal}")

    out_file = out_dir / f"{complex_id}_docked_{input_num}.{lig_ext}"

    out_file_sdf = out_file.with_suffix(".sdf")
    if not overwrite and out_file_sdf.is_file() and out_file_sdf.stat().st_size > 0:
        print("Output already exists, skipping...")
        return 0, out_dir, out_dir / f"{complex_id}_{input_num}.log"

    box_args = []

    # check if we should use autobox or standard coords
    if "autobox" in size:
        box_args += ["--autobox_ligand", str(ligand_crystal)]

        # add autobox extend
        autobox_ext = size.split("_")[-1]
        if autobox_ext.isdigit():
            box_args += ["--autobox_add", str(autobox_ext)]
    else:
        # ensure input coords are all floats
        if all(isinstance(v, float) for v in (x_coord, y_coord, z_coord)):
            cx, cy, cz = x_coord, y_coord, z_coord
        else:
            cx, cy, cz = compute_centroid(ligand_crystal)

        box_args += [
            "--center_x",
            str(cx),
            "--center_y",
            str(cy),
            "--center_z",
            str(cz),
            "--size_x",
            str(size),
            "--size_y",
            str(size),
            "--size_z",
            str(size),
        ]

    cmd = [
        str(method),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand_input),
        "--out",
        str(out_file),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
    ]

    cmd += box_args

    if seed is not None:
        cmd += ["--seed", str(seed)]
    if scoring is not None:
        cmd += ["--scoring", str(scoring)]

    if "gnina" in selected_method and cnn_model not in (None, "default"):
        cnn_path = Path(cnn_model).expanduser().resolve()

        if not cnn_path.exists():
            raise FileNotFoundError(f"cnn_model path does not exist: {cnn_model}")

        if cnn_path.is_file():
            cmd += ["--cnn_model", str(cnn_path)]

        elif cnn_path.is_dir():
            model_paths = sorted(cnn_path.glob("*.pt"))

            if not model_paths:
                raise FileNotFoundError(
                    f"No .pt files found in cnn_model directory: {cnn_path}"
                )

            cmd.extend(["--cnn_model", *map(str, model_paths)])

        else:
            raise FileNotFoundError(
                f"cnn_model path is neither file nor directory: {cnn_path}"
            )

    # don't need to change the subprocess base dir
    method_cwd = None

    with log_file.open("w") as lf:
        lf.write(shlex.join(cmd) + "\n\n")
        lf.flush()

        timed_cmd = [
            "bash",
            "-c",
            "TIMEFORMAT=$'ELAPSED_SECONDS=%R\\nUSER_SECONDS=%U\\nSYSTEM_SECONDS=%S'; time \"$@\"",
            "bash",
            *cmd,
        ]

        proc = subprocess.run(
            timed_cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=method_cwd,
        )

    return proc.returncode, out_dir, log_file


def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Run docking over a folder-of-folders (fixed layout)"
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing complex subfolders",
    )
    parser.add_argument(
        "--method",
        type=Path,
        help="Path to docking executable or base directory",
        required=True,
    )
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--num-inputs", type=int, default=1)
    parser.add_argument("--num-modes", type=int, default=40)
    parser.add_argument(
        "--size",
        default=25,
        help="Box size (single value used for x,y,z) or autobox extend size",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scoring", type=str, default=None)
    parser.add_argument(
        "--cnn_model",
        type=str,
        default=None,
        help="Optional path to CNN model (gnina only). If 'default' or unset, uses gnina default",
    )
    parser.add_argument(
        "--complex",
        "-c",
        default=None,
        help="Run single complex (folder name) instead of all subfolders",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run docking even if output files already exist",
    )

    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Optional explicit protein path to use for all complexes (overrides prepared_inputs/*_prepared.*)",
    )
    parser.add_argument(
        "--x_coord", type=float, default=None, help="Optional box center X"
    )
    parser.add_argument(
        "--y_coord", type=float, default=None, help="Optional box center Y"
    )
    parser.add_argument(
        "--z_coord", type=float, default=None, help="Optional box center Z"
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    print("\n=== CPU information ===")
    subprocess.run(["lscpu"], check=False)

    print("\n=== GPU information ===")
    subprocess.run(["nvidia-smi"], check=False)

    print()

    start_time = time.time()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root folder not found: {root}")

    protein_arg = args.protein.strip() if args.protein else None

    # If coordinates are partially provided, error out early
    coords_provided = [
        args.x_coord is not None,
        args.y_coord is not None,
        args.z_coord is not None,
    ]
    if any(coords_provided) and not all(coords_provided):
        raise SystemExit(
            "You must provide all three coordinates (--x_coord, --y_coord, --z_coord) or none."
        )

    if args.complex:
        complexes = [root / args.complex]
    else:
        # run for all complexes
        complexes = sorted(
            p for p in root.iterdir() if p.is_dir() and not p.name.startswith("slurm")
        )

    print(f"Found {len(complexes)} complexes. Running docking...")

    failed = []

    success_count = 0
    attempted_count = 0

    num_inputs = args.num_inputs

    for c in complexes:
        for input_num in range(1, num_inputs + 1):
            attempted_count += 1
            cid = c.name
            print(f"-> {cid}_{input_num} ...", end="", flush=True)
            try:
                ret, out_sdf, log = run_docking_on_complex(
                    folder=c,
                    complex_id=cid,
                    method=args.method,
                    exhaustiveness=args.exhaustiveness,
                    input_num=f"{input_num}",
                    num_modes=args.num_modes,
                    size=args.size,
                    seed=args.seed,
                    scoring=args.scoring,
                    cnn_model=args.cnn_model,
                    protein_path=protein_arg,
                    x_coord=args.x_coord,
                    y_coord=args.y_coord,
                    z_coord=args.z_coord,
                    overwrite=args.overwrite,
                )
                if ret == 0:
                    success_count += 1

                    print(" OK")
                    print(f"   poses: {out_sdf}")
                    print(f"   log:   {log}")
                else:
                    print(f" FAIL (exit {ret})")
                    failed.append((cid, ret, log))

            except Exception as e:
                print(f" EXCEPTION: {e}")
                failed.append((f"{cid}_{input_num}", str(e), None))

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    total_complexes = len(complexes)
    total_tasks = attempted_count

    success_pct = (success_count / total_tasks * 100.0) if total_tasks > 0 else 0.0

    print(
        f"Summary: complexes={total_complexes}, "
        f"tasks={total_tasks}, "
        f"succeeded={success_count} ({success_pct:.1f}%), "
        f"failed={len(failed)}"
    )
    print(f"Elapsed time: {elapsed_str}")

    if failed:
        print(f"{len(failed)} failures:")
        for fid, ret_or_err, log in failed:
            print(f" - {fid}: {ret_or_err}")


if __name__ == "__main__":
    main()
