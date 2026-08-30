#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Analysis job settings
#
# Adjust these for your Slurm environment. Leave PARTITION or CLUSTERS empty
# if they are not required.
# -----------------------------------------------------------------------------
CPUS=8
MEM="8G"
TIME="24:00:00"
PARTITION=""
CLUSTERS=""
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyse_docking.py"
CONFIG_DIR="${SCRIPT_DIR}/docking_config"
OUTDIR="${REPO_ROOT}/analysis/parquets"

DRY_RUN=false
OVERWRITE=false

usage() {
    cat <<EOF
Usage: $0 [options] <dataset-root>

Options:
  --dry-run          Print the jobs without submitting them.
  --overwrite        Recalculate joined poses, PoseBusters and OST outputs.
  --outdir DIR       Pose-table output directory (default: analysis/parquets).
  -h, --help         Show this help message.
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --outdir)
            [[ $# -ge 2 ]] || { echo "ERROR: --outdir requires a value" >&2; exit 1; }
            OUTDIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -* )
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ne 1 ]]; then
    usage >&2
    exit 1
fi

ROOT="$(realpath "${POSITIONAL[0]}")"
mkdir -p "${OUTDIR}" "${REPO_ROOT}/analysis/logs"

if [[ ! -d "${ROOT}" ]]; then
    echo "ERROR: Dataset root does not exist: ${ROOT}" >&2
    exit 1
fi
if [[ ! -f "${ANALYSIS_SCRIPT}" ]]; then
    echo "ERROR: Analysis script not found: ${ANALYSIS_SCRIPT}" >&2
    exit 1
fi
if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "ERROR: Config directory not found: ${CONFIG_DIR}" >&2
    exit 1
fi

mapfile -t CONFIGURATIONS < <(
    python - "${CONFIG_DIR}" <<'PYCONF'
from pathlib import Path
import sys
import yaml

config_dir = Path(sys.argv[1])
for config_path in sorted(config_dir.glob("*.yaml")):
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "gnina" not in config.get("methods", {}):
        continue

    defaults = config.get("defaults", {})
    presets = config.get("presets", {})
    for preset_name, preset in presets.items():
        num_inputs = int(preset["num_inputs"])
        num_modes = int(preset["num_modes"])
        for exhaustiveness in defaults.get("exhaustiveness", []):
            print(preset_name, num_inputs, num_modes, int(exhaustiveness), sep="\t")
PYCONF
)

if [[ ${#CONFIGURATIONS[@]} -eq 0 ]]; then
    echo "ERROR: No GNINA analysis configurations found in ${CONFIG_DIR}" >&2
    exit 1
fi

echo "Found ${#CONFIGURATIONS[@]} analysis configurations."

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS=$'\t' read -r PRESET_NAME NUM_INPUTS NUM_MODES EXHAUSTIVENESS <<< "${configuration}"

    ANALYSIS_ARGS=(
        --root "${ROOT}"
        --outdir "${OUTDIR}"
        --cores "${CPUS}"
        --exhaustiveness "${EXHAUSTIVENESS}"
    )

    if [[ "${NUM_INPUTS}" -eq 1 ]]; then
        METHOD="gnina_single"
        ANALYSIS_ARGS+=(--method "${METHOD}" --single-num-modes "${NUM_MODES}")
    else
        METHOD="gnina_multi"
        ANALYSIS_ARGS+=(
            --method "${METHOD}"
            --multi-num-modes "${NUM_MODES}"
            --expected-conformers "${NUM_INPUTS}"
        )
    fi

    if "${OVERWRITE}"; then
        ANALYSIS_ARGS+=(--overwrite)
    fi

    JOB_NAME="analyse_${PRESET_NAME}_exh${EXHAUSTIVENESS}"
    SBATCH_ARGS=(
        --job-name="${JOB_NAME}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
        --output="${REPO_ROOT}/analysis/logs/${JOB_NAME}_%j.out"
        --error="${REPO_ROOT}/analysis/logs/${JOB_NAME}_%j.err"
    )
    [[ -n "${PARTITION}" ]] && SBATCH_ARGS+=(--partition="${PARTITION}")
    [[ -n "${CLUSTERS}" ]] && SBATCH_ARGS+=(--clusters="${CLUSTERS}")

    COMMAND=(python "${ANALYSIS_SCRIPT}" "${ANALYSIS_ARGS[@]}")
    WRAPPED_COMMAND="$(printf '%q ' "${COMMAND[@]}")"

    if "${DRY_RUN}"; then
        printf '[DRY-RUN] sbatch'
        printf ' %q' "${SBATCH_ARGS[@]}"
        printf ' --wrap=%q\n' "${WRAPPED_COMMAND}"
    else
        sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAPPED_COMMAND}"
        echo "Submitted ${JOB_NAME}"
    fi
done