#!/usr/bin/env python3
"""
YAML-driven SLURM launcher for dock_complexes.py.

Author: Jochem Nelen (jochem.nelen@stats.ox.ac.uk)

Reads a project configuration file, builds one SLURM job per requested
seed/protein/method/preset combination, and either prints the generated
sbatch script (--dry-run) or submits it with sbatch.

"""

import yaml
import subprocess
import sys
from pathlib import Path
import argparse
import textwrap
import shlex

from typing import Optional, Sequence

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
{sbatch_lines}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err

{command}
"""

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = SCRIPT_DIR / "docking_config"
DOCK_SCRIPT = SCRIPT_DIR / "dock_complexes.py"

DOCKING_ROOT = SCRIPT_DIR.parent


def die(msg: str) -> None:
    """
    Exit the program with a human-readable error message.

    Args:
        msg: Error message to display.
    """
    sys.exit(f"ERROR: {msg}")


def resolve_methods(cfg: dict) -> dict:
    """
    Validate and normalize the per-method configuration.

    Each method must define:
        - executable
        - profile

    The following fields are optional:
        - singularity
        - scoring

    Args:
        cfg: Top-level parsed YAML configuration.

    Returns:
        A mapping from method name to resolved method configuration.
    """
    methods = cfg.get("methods")
    if not isinstance(methods, dict):
        die("'methods' must be defined as a mapping")

    resolved = {}
    for name, m in methods.items():
        exe = m.get("executable")
        if not exe:
            die(f"Method '{name}' missing 'executable'")

        p = resolve_software_relative_path(exe)
        if not p.exists():
            die(f"Executable for '{name}' not found: {p}")

        resolved[name] = {
            "executable": str(p),
            "profile": m.get("profile"),
            "singularity": m.get("singularity"),
            "scoring": m.get("scoring"),
            "cnn_model": m.get("cnn_model"),
        }

    return resolved


def resolve_config_paths(config_args: Optional[Sequence[Path]]) -> list[Path]:
    """
    Resolve YAML configuration files from command-line arguments.

    If no configuration paths are provided, this defaults to reading all
    *.yaml files in the docking_config directory. If an argument is a
    directory, all *.yaml files directly inside that directory are used. If
    an argument is a file, it is validated and used as a single configuration
    file.

    Directory expansion is non-recursive.

    Args:
        config_args: Optional sequence of configuration paths provided via the
            command line. Each path may refer to either a YAML file or a
            directory containing YAML files. If None, defaults to
            docking_config.

    Returns:
        A list of absolute, resolved paths to YAML configuration files.

    Raises:
        SystemExit: If a provided path does not exist, is not a file when a file
            is expected, or if a directory contains no *.yaml files.
    """
    if config_args is None:
        config_args = [DEFAULT_CONFIG_DIR]

    paths: list[Path] = []

    for item in config_args:
        item = item.expanduser()

        if item.is_dir():
            matches = sorted(item.glob("*.yaml"))

            if not matches:
                die(f"No .yaml files found in directory: {item}")

            paths.extend(path.resolve() for path in matches)

        else:
            path = item.resolve()

            if not path.exists():
                die(f"Config not found: {path}")

            if not path.is_file():
                die(f"Config path is not a file: {path}")

            paths.append(path)

    return paths


def resolve_software_relative_path(path: str) -> Path:
    """
    Resolve a path defined in a YAML config in a relative way.

    Absolute paths are returned unchanged. Relative paths are interpreted
    relative to DOCKING_ROOT, which is inferred from the location of this launcher
    script.

    Args:
        path: Path string from the YAML configuration.

    Returns:
        Absolute, resolved path.
    """
    p = Path(path).expanduser()

    if not p.is_absolute():
        p = DOCKING_ROOT / p

    return p.resolve()


def validate_center(
    center: Optional[Sequence[float]],
) -> Optional[tuple[float, float, float]]:
    """
    Args:
        center: None, or a sequence of exactly three numeric values.

    Returns:
        None if no center was provided, otherwise a 3-tuple of floats.
    """
    if center is None or center == []:
        return None

    if not isinstance(center, (list, tuple)):
        die("'center' must be a list or tuple of exactly 3 numbers, e.g. [x, y, z]")

    if len(center) != 3:
        die(f"'center' must contain exactly 3 values, got {len(center)}")

    if not all(isinstance(c, (int, float)) for c in center):
        die("'center' values must all be integers or floats")

    return (float(center[0]), float(center[1]), float(center[2]))


def render_sbatch_lines(sbatch: dict[str, object]) -> str:
    """
    Render SBATCH directives from a dictionary.

    Args:
        sbatch: Mapping of SBATCH option names to values.

    Returns:
        A newline-separated string of SBATCH directives.
    """
    lines = []
    for key, value in sbatch.items():
        if value is None:
            continue
        lines.append(f"#SBATCH --{key}={value}")
    return "\n".join(lines)


def build_command(
    root,
    method_name,
    method_cfg,
    seed,
    protein_path,
    cfg,
    preset_name=None,
    exhaustiveness=None,
    center=None,
    cnn_model=None,
    overwrite=False,
):
    """
    Build the command that will run inside the SLURM job.

    Args:
        root: Project root passed through to dock_complexes.py.
        method_name: Name of the selected docking method.
        method_cfg: Resolved per-method configuration.
        seed: Random seed for the job.
        protein_path: Optional absolute protein path; omitted for redock.
        cfg: Full configuration mapping.
        preset_name: Preset name for gnina/smina jobs.
        center: Optional 3D center coordinates.
        overwrite: Run docking even if output files already exist.

    Returns:
        Command as a list of arguments, suitable for shlex.join().
    """
    defaults = cfg["defaults"]

    cmd = [
        "python",
        "-u",
        str(DOCK_SCRIPT),
        str(root),
        "--method",
        method_cfg["executable"],
        "--seed",
        str(seed),
    ]

    preset = cfg["presets"][preset_name]

    cmd += [
        "--num-inputs",
        str(preset["num_inputs"]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--size",
        str(defaults["box_size"]),
        "--num-modes",
        str(preset["num_modes"]),
    ]

    scoring = method_cfg.get("scoring")
    if scoring:
        cmd += ["--scoring", str(scoring)]

    if method_name == "gnina" and cnn_model not in (None, "default"):
        cmd += ["--cnn_model", str(cnn_model)]

    if center:
        x, y, z = center
        cmd += [
            "--x_coord",
            str(x),
            "--y_coord",
            str(y),
            "--z_coord",
            str(z),
        ]

    if overwrite:
        cmd += ["--overwrite"]

    if protein_path:
        cmd += ["--protein", protein_path]

    return cmd


def main() -> None:
    """
    CLI entry point for submitting docking jobs.

    Reads a YAML config, resolves executables and optional settings, builds
    per-job SBATCH scripts, and either prints them (--dry-run) or submits
    them with sbatch.
    """
    parser = argparse.ArgumentParser("Submit docking jobs")
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more YAML config files or directories. "
            "Directories are expanded to *.yaml files. "
            "Defaults to docking_config/*.yaml."
        ),
    )

    args = parser.parse_args()

    if not DOCK_SCRIPT.is_file():
        die(f"dock_complexes.py not found next to launcher script: {DOCK_SCRIPT}")

    root = args.root.resolve()
    if not root.exists():
        die(f"Root directory not found: {root}")

    config_paths = resolve_config_paths(args.config)

    jobs_launched = 0

    for config_path in config_paths:
        print(f"\n=== Processing config: {config_path} ===")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            die(f"Config did not parse as a mapping: {config_path}")

        methods = resolve_methods(cfg)
        profiles = cfg["profiles"]
        defaults = cfg["defaults"]

        center = validate_center(cfg.get("center"))

        seeds = defaults["seeds"]
        proteins = defaults["protein_paths"]

        for seed in seeds:
            for protein_cfg in proteins:
                if protein_cfg == "redock":
                    protein_path = "redock"
                    protein_name = "redock"
                else:
                    protein_path = str(protein_cfg).strip()
                    protein_name = Path(protein_path).stem

                for method_name, method_cfg in methods.items():
                    cnn_models = method_cfg.get("cnn_model")

                    if method_name == "gnina":
                        if cnn_models is None:
                            cnn_models = [None]
                        elif isinstance(cnn_models, list):
                            cnn_models = cnn_models
                        else:
                            cnn_models = [cnn_models]
                    else:
                        cnn_models = [None]

                    profile_name = method_cfg.get("profile")
                    if profile_name not in profiles:
                        die(f"Profile '{profile_name}' not found in {config_path}")

                    profile = profiles[profile_name]

                    preset_names = list(cfg["presets"])

                    for cnn_model in cnn_models:
                        for preset_name in preset_names:
                            exhaustiveness_values = defaults["exhaustiveness"]

                            for exhaustiveness in exhaustiveness_values:
                                if method_name == "gnina" and cnn_model not in (
                                    None,
                                    "default",
                                ):
                                    method_label = (
                                        f"{method_name}_{Path(cnn_model).stem}"
                                    )
                                else:
                                    method_label = method_name

                                name_parts = [method_label, protein_name]

                                if preset_name:
                                    name_parts.insert(2, preset_name)
                                if exhaustiveness is not None:
                                    name_parts.append(f"exh{exhaustiveness}")

                                name_parts.append(f"seed{seed}")

                                job_name = "_".join(name_parts)

                                job_dir = (
                                    root / "slurm_jobs" / protein_name / method_label
                                )
                                log_dir = (
                                    root / "slurm_logs" / protein_name / method_label
                                )

                                if preset_name:
                                    job_dir /= preset_name
                                    log_dir /= preset_name

                                if exhaustiveness is not None:
                                    job_dir /= f"exh_{exhaustiveness}"
                                    log_dir /= f"exh_{exhaustiveness}"

                                sbatch_lines = render_sbatch_lines(profile)

                                cmd = build_command(
                                    root=root,
                                    method_name=method_name,
                                    method_cfg=method_cfg,
                                    seed=seed,
                                    protein_path=protein_path,
                                    cfg=cfg,
                                    preset_name=preset_name,
                                    exhaustiveness=exhaustiveness,
                                    center=center,
                                    cnn_model=cnn_model,
                                    overwrite=args.overwrite,
                                )

                                singularity = method_cfg.get("singularity")
                                if singularity:
                                    singularity = str(
                                        resolve_software_relative_path(singularity)
                                    )
                                    bind_paths = {
                                        root.resolve(),
                                        DOCKING_ROOT.resolve(),
                                    }

                                    bind_arg = ",".join(
                                        f"{p}:{p}" for p in sorted(bind_paths, key=str)
                                    )

                                    cmd = [
                                        "singularity",
                                        "exec",
                                        "--nv",
                                        "--bind",
                                        bind_arg,
                                        "--pwd",
                                        str(DOCKING_ROOT.resolve()),
                                        singularity,
                                    ] + cmd

                                command_str = shlex.join(cmd)

                                script = SBATCH_TEMPLATE.format(
                                    job_name=job_name,
                                    sbatch_lines=sbatch_lines,
                                    log_dir=str(log_dir.resolve()),
                                    command=command_str,
                                )

                                path = job_dir / f"{job_name}.sbatch"

                                if args.dry_run:
                                    print(f"[DRY-RUN] sbatch {path}")
                                    print(script)
                                else:
                                    job_dir.mkdir(parents=True, exist_ok=True)
                                    log_dir.mkdir(parents=True, exist_ok=True)
                                    path.write_text(textwrap.dedent(script))
                                    subprocess.run(["sbatch", str(path)], check=True)
                                    print(f"Submitted {job_name}")

                                jobs_launched += 1

    print(f"\nTotal jobs launched: {jobs_launched}")


if __name__ == "__main__":
    main()
