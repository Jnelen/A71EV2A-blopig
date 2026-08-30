#!/usr/bin/env python3
"""
Prepare Fragalysis complex directories.

Author: Jochem Nelen (jochem.nelen@stats.ox.ac.uk)

Description:
 - protonate receptor using pdbfixer
 - protonate ligand using OpenBabel
 - generate conformers with RDKit
 - apo mode: prepare a target protein and align it into each prepared complex
 - crystal_crossdock option: supply a csv indicating which structures were part of an initial fragment screen, and use those for a more realistic cross-docking scenario
Notes:
 - Expects files named: <complex_id>_delig-desolv.pdb and <complex_id>_ligand.sdf in each complex folder.
 - apo and crystal_crossdock mode expects the regular receptor preparations to already exist.
 - This protocol has been partially adapted from https://github.com/inductive-bio/strong-docking-baseline/tree/main
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import argparse
import os
import subprocess
from typing import Optional, Tuple, Callable

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

from tqdm import tqdm

from pdbfixer import PDBFixer
from openmm.app import ForceField
from openmm.app import PDBFile
from openmm import unit

from prody import confProDy, parsePDB, matchAlign, writePDB

ff = ForceField("amber19-all.xml")


def chains_to_keep(
    fixer,
    ligand_in: Path,
    cutoff_nm: float = 0.6,  # 6 Å
    chain_mode: str = "nearby",
) -> list[str]:
    """
    Return chain IDs according to chain_mode:
    - single: keep the chain with the most ligand-contacting residues
    - nearby: keep all chains within cutoff_nm of the ligand
    - all: keep all chains
    """
    chain_ids = [chain.id for chain in fixer.topology.chains()]

    if chain_mode == "all" or len(chain_ids) <= 1:
        return chain_ids

    mol = Chem.MolFromMolFile(str(ligand_in), removeHs=True)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Failed to read ligand: {ligand_in}")

    conf = mol.GetConformer()

    lig_coords = (
        np.array(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )
        * 0.1
    )  # Å -> nm

    chain_stats = []

    for chain in fixer.topology.chains():
        contact_residues = 0
        min_dist = float("inf")

        for residue in chain.residues():
            atom_indices = [
                atom.index
                for atom in residue.atoms()
                if atom.element is not None and atom.element.atomic_number != 1
            ]

            if not atom_indices:
                continue

            prot_coords = np.array(
                [
                    fixer.positions[i].value_in_unit(unit.nanometer)
                    for i in atom_indices
                ],
                dtype=float,
            )

            dists = np.linalg.norm(
                lig_coords[:, None, :] - prot_coords[None, :, :],
                axis=2,
            )

            residue_min = float(np.min(dists))
            min_dist = min(min_dist, residue_min)

            if residue_min <= cutoff_nm:
                contact_residues += 1

        if min_dist < float("inf"):
            chain_stats.append((chain.id, contact_residues, min_dist))

    if chain_mode == "single":
        return [
            max(
                chain_stats,
                key=lambda item: (item[1], -item[2], item[0]),
            )[0]
        ]

    if chain_mode == "nearby":
        keep_chain_ids = [
            chain_id for chain_id, _, min_dist in chain_stats if min_dist <= cutoff_nm
        ]

        if not keep_chain_ids:
            raise ValueError("No chains found within 6 Å of ligand")

        return keep_chain_ids

    raise ValueError(f"Unknown chain mode: {chain_mode}")


def protonate_receptor_and_ligand(
    folder: Path,
    complex_id: str,
    chain_mode: str = "single",
) -> None:
    """
    Protonate and fix the receptor (with PDBFixer) and ligand (with obabel).
    """
    output_dir = folder / "prepared_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_in = folder / f"{complex_id}_ligand.sdf"
    ligand_out = output_dir / f"{complex_id}_ligand_H.sdf"

    if not ligand_in.exists():
        raise FileNotFoundError(f"Ligand input not found: {ligand_in}")

    subprocess.run(
        ["obabel", str(ligand_in), "-O", str(ligand_out), "-p", "7.4"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    protein_in = folder / f"{complex_id}_delig-desolv.pdb"
    protein_out = output_dir / f"{complex_id}_prepared.pdb"

    if not protein_in.exists():
        raise FileNotFoundError(f"Protein input not found: {protein_in}")

    fixer = PDBFixer(filename=str(protein_in))
    chain_ids = [chain.id for chain in fixer.topology.chains()]

    if len(chain_ids) > 1:
        keep_chain_ids = chains_to_keep(
            fixer,
            ligand_in,
            cutoff_nm=0.6,
            chain_mode=chain_mode,
        )

        if chain_mode == "nearby" and len(keep_chain_ids) > 1:
            print(
                f"[warning] {protein_in} retained more than one protein chain "
                "and may require further manual inspection or curation."
            )

        fixer.removeChains(
            chainIds=[cid for cid in chain_ids if cid not in keep_chain_ids]
        )

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    fixer.addMissingHydrogens(pH=7.4, forcefield=ff)

    with open(protein_out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def generate_conformers(
    folder: Path,
    complex_id: str,
    num_confs: int = 1,
    seed: int = 1,
) -> None:
    """
    Generate conformers for the ligand using RDKit and write SDF files.
    """
    output_dir = folder / "prepared_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_h_sdf = output_dir / f"{complex_id}_ligand_H.sdf"
    if not ligand_h_sdf.exists():
        raise FileNotFoundError(f"Protonated ligand not found: {ligand_h_sdf}")

    mol = Chem.MolFromMolFile(str(ligand_h_sdf), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit failed to read molecule from {ligand_h_sdf}")

    Chem.SanitizeMol(mol)
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed)
    AllChem.UFFOptimizeMoleculeConfs(mol)

    for i, conf in enumerate(mol.GetConformers(), start=1):
        cid = conf.GetId()
        sdf_out = output_dir / f"{complex_id}_ligand_prepared_{i}.sdf"
        with Chem.SDWriter(str(sdf_out)) as writer:
            writer.write(mol, confId=cid)


def process_complex(
    complex_path: Path,
    num_confs: int,
    seed: int,
    chain_mode: str,
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single complex folder.
    """
    if not complex_path.is_dir():
        raise NotADirectoryError(f"{complex_path} is not a directory")

    complex_id = complex_path.name

    try:
        protonate_receptor_and_ligand(
            complex_path,
            complex_id,
            chain_mode=chain_mode,
        )
        generate_conformers(
            complex_path,
            complex_id,
            num_confs=num_confs,
            seed=seed,
        )
        return (complex_id, True, None)
    except Exception as exc:
        return (complex_id, False, str(exc))


def prepare_apo_protein(pdb_in: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Prepare a single apo protein PDB:
    - read with PDBFixer
    - add hydrogens at pH 7.4
    - write <stem>_prepared.pdb
    """
    if not pdb_in.exists():
        raise FileNotFoundError(f"Protein input not found: {pdb_in}")

    if pdb_in.suffix.lower() != ".pdb":
        raise ValueError(f"Expected a .pdb file, got: {pdb_in}")

    if output_dir is None:
        output_dir = pdb_in.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_out = output_dir / f"{pdb_in.stem}_prepared.pdb"

    fixer = PDBFixer(filename=str(pdb_in))

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    fixer.addMissingHydrogens(pH=7.4, forcefield=ff)

    with open(pdb_out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    return pdb_out


def align_pdb_to_reference(
    mobile_pdb: Path,
    reference_pdb: Path,
    output_pdb: Path,
) -> Path:
    """
    Align mobile_pdb onto reference_pdb using ProDy.
    """
    if not mobile_pdb.exists():
        raise FileNotFoundError(f"Mobile PDB not found: {mobile_pdb}")
    if not reference_pdb.exists():
        raise FileNotFoundError(f"Reference PDB not found: {reference_pdb}")

    mobile = parsePDB(str(mobile_pdb))
    reference = parsePDB(str(reference_pdb))

    mobile = matchAlign(mobile, reference)[0]

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    writePDB(str(output_pdb), mobile)
    return output_pdb


def process_apo_alignment(
    complex_dir: Path,
    apo_prepared: Path,
    output_name: Optional[str] = None,
) -> Tuple[str, bool, Optional[str]]:
    """
    Align the apo structure into one complex folder.
    Returns (complex_id, ok, reason).
    """
    if not complex_dir.is_dir():
        return (complex_dir.name, False, f"{complex_dir} is not a directory")

    complex_id = complex_dir.name
    prepared_dir = complex_dir / "prepared_inputs"
    ref_pdb = prepared_dir / f"{complex_id}_prepared.pdb"

    if not ref_pdb.exists():
        return (complex_id, False, f"Missing reference PDB: {ref_pdb}")

    if not output_name:
        output_name = f"{apo_prepared.stem}_aligned.pdb"

    out_pdb = prepared_dir / output_name

    try:
        align_pdb_to_reference(
            mobile_pdb=apo_prepared,
            reference_pdb=ref_pdb,
            output_pdb=out_pdb,
        )
        return (complex_id, True, None)
    except Exception as exc:
        return (complex_id, False, str(exc))


def iter_complex_dirs(root: Path, complex_name: Optional[str] = None) -> list[Path]:
    """
    Return either a single complex directory or all complex directories under root.
    """
    if complex_name is not None:
        return [root / complex_name]

    return [p for p in sorted(root.iterdir()) if p.is_dir() and "slurm" not in p.name]


def run_parallel(
    items: list[Path],
    worker: Callable[[Path], Tuple[str, bool, Optional[str]]],
    desc: str,
    num_workers: int,
) -> list[Tuple[str, str]]:
    """
    Run worker(item) in parallel and print only errors.
    Returns [(complex_id, reason), ...].
    """
    failures: list[Tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(worker, item) for item in items]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
            unit="complex",
        ):
            try:
                cid, ok, reason = fut.result()
            except Exception as exc:
                cid, ok, reason = (
                    "<unknown>",
                    False,
                    f"Unhandled exception in worker: {exc}",
                )

            if not ok:
                failures.append((cid, reason or "Unknown error"))
                print(f"[error] {cid}: {reason}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare complexes for docking. Input should be the aligned_files/ directory."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Root directory containing one folder per complex.",
    )
    parser.add_argument(
        "--num-confs",
        "-n",
        type=int,
        default=5,
        help="Number of conformers to generate.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=1,
        help="Random seed for conformer generation.",
    )
    parser.add_argument(
        "--complex",
        "-c",
        type=str,
        default=None,
        help="Process only a single complex folder name.",
    )
    parser.add_argument(
        "--num-workers",
        "-j",
        type=int,
        default=None,
        help="Number of parallel worker processes to use (defaults to CPU count).",
    )
    parser.add_argument(
        "--apo-pdb",
        type=Path,
        default=None,
        help="Path to the apo protein PDB to prepare and align.",
    )
    parser.add_argument(
        "--fragment-crossdock",
        type=Path,
        default=None,
        help="Automatically identify and prepare the best suitable crossdock structure based on a supplied csv containing which complexes were part of an initial fragment screen.",
    )
    parser.add_argument(
        "--chain-mode",
        choices=["single", "nearby", "all"],
        default="single",
        help=(
            "Protein chain handling: "
            "'single' keeps the chain with most ligand contacts, "
            "'nearby' keeps all chains within 6 Å of the ligand, "
            "'all' keeps all chains (default: nearby)."
        ),
    )
    args = parser.parse_args()

    confProDy(verbosity="none")

    root: Path = args.input_path.expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")

    num_workers = args.num_workers or (os.cpu_count() or 1)
    chain_mode = args.chain_mode

    if args.apo_pdb is not None:
        if not root.is_dir():
            parser.error(
                "--apo-pdb expects input_path to be the dataset root directory"
            )

        apo_prepared = prepare_apo_protein(args.apo_pdb, output_dir=args.apo_pdb.parent)
        targets = iter_complex_dirs(root, args.complex)

        if not targets:
            return

        worker = partial(process_apo_alignment, apo_prepared=apo_prepared)
        failures = run_parallel(
            targets,
            worker=worker,
            desc="Aligning apo structure",
            num_workers=num_workers,
        )

        if failures:
            fail_file = root.parent / "failed_apo_alignments.tsv"
            with fail_file.open("w") as fh:
                fh.write("complex_id\treason\n")
                for cid, reason in failures:
                    safe_reason = reason.replace("\n", " ").replace("\t", " ")
                    fh.write(f"{cid}\t{safe_reason}\n")
        return

    if args.fragment_crossdock:
        fragment_df = pd.read_csv(args.fragment_crossdock)
        fragment_list = fragment_df.loc[
            (fragment_df["fragment_screen"].astype(bool))
            & (fragment_df["pb_valid"].astype(bool))
            & (~fragment_df["artefact"].astype(bool)),
            "complex_name",
        ].tolist()

        fragment_ids = []
        fragment_fps = []

        num_fragment_crossdock = []

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        for fragment_id in fragment_list:
            mol_path = root / fragment_id / f"{fragment_id}_ligand.sdf"
            fragment_mol = Chem.MolFromMolFile(str(mol_path))

            if fragment_mol is None:
                continue

            fragment_ids.append(fragment_id)
            fragment_fps.append(gen.GetFingerprint(fragment_mol))

        targets = iter_complex_dirs(root, args.complex)

        if not targets:
            return

        # loop through all complexes and find most similar fragment
        for complex_path in tqdm(targets):
            complex_id = complex_path.name

            complex_mol = Chem.MolFromMolFile(complex_path / f"{complex_id}_ligand.sdf")
            complex_fp = gen.GetFingerprint(complex_mol)

            sims = DataStructs.BulkTanimotoSimilarity(complex_fp, fragment_fps)

            best_idx = max(range(len(sims)), key=sims.__getitem__)

            num_fragment_crossdock.append(
                [complex_id, fragment_ids[best_idx], sims[best_idx]]
            )

            fragment_path = (
                root
                / fragment_ids[best_idx]
                / "prepared_inputs"
                / f"{fragment_ids[best_idx]}_prepared.pdb"
            )
            output_name = f"fragment_crossdock_{fragment_ids[best_idx]}_{sims[best_idx]:.2f}_aligned.pdb"

            process_apo_alignment(complex_path, fragment_path, output_name)

        df = pd.DataFrame(
            num_fragment_crossdock,
            columns=[
                "complex_id",
                "most_similar_fragment",
                "ECFP4_Tanimoto_Similarity",
            ],
        )
        fragment_csv_path = root.parent / "fragment_crossdock_data.csv"
        df.to_csv(fragment_csv_path, index=False)
        print(f"Wrote fragment_crossdock details to {str(fragment_csv_path)}")
        return

    if args.complex:
        targets = iter_complex_dirs(root, args.complex)
        if not targets:
            return

        worker = partial(
            process_complex,
            num_confs=args.num_confs,
            seed=args.seed,
            chain_mode=chain_mode,
        )
        failures = run_parallel(
            targets,
            worker=worker,
            desc="Processing complex",
            num_workers=1,
        )

        if failures:
            fail_file = root.parent / "failed_complexes.tsv"
            with fail_file.open("w") as fh:
                fh.write("complex_id\treason\n")
                for cid, reason in failures:
                    safe_reason = reason.replace("\n", " ").replace("\t", " ")
                    fh.write(f"{cid}\t{safe_reason}\n")
        return

    targets = iter_complex_dirs(root)
    if not targets:
        return

    worker = partial(
        process_complex,
        num_confs=args.num_confs,
        seed=args.seed,
        chain_mode=chain_mode,
    )
    failures = run_parallel(
        targets,
        worker=worker,
        desc="Processing complexes",
        num_workers=num_workers,
    )

    if failures:
        fail_file = root.parent / "failed_complexes.tsv"
        with fail_file.open("w") as fh:
            fh.write("complex_id\treason\n")
            for cid, reason in failures:
                safe_reason = reason.replace("\n", " ").replace("\t", " ")
                fh.write(f"{cid}\t{safe_reason}\n")


if __name__ == "__main__":
    main()
