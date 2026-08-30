# A71EV2A GNINA sampling benchmark

This repository contains the code, processed results, and reproducible workflow accompanying the Blopig post: [**The cost of a better pose: balancing GNINA sampling and runtime**](https://www.blopig.com/blog/2026/09/the-cost-of-a-better-pose-balancing-gnina-sampling-and-runtime/)

The blog post presents the motivation, results, and interpretation of the benchmark, while this repository provides the code and data needed to reproduce and extend the analysis.

Using the [OpenBind EV-A71 2A protease dataset](https://www.biorxiv.org/content/10.64898/2026.08.27.747600), we investigate how GNINA pose-recovery performance changes when additional compute is spent on:

- Increased exhaustiveness
- Repeated docking with different random seeds
- Docking from multiple ligand starting conformers

The repository supports three levels of reproducibility:

1. **Reproduce the figures directly** from the processed outputs included in this repository.
2. **Reanalyse the docking results** using the complete precomputed GNINA output archive available from [Zenodo](https://doi.org/10.5281/zenodo.22211565).
3. **Reproduce the full workflow from scratch** starting from the original OpenBind data release.

## Installation

Clone the repository:

```bash
git clone https://github.com/Jnelen/A71EV2A-blopig.git
cd A71EV2A-blopig
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate openbind-docking
```

## Reproduce the figures

The processed pose-level analysis tables, runtime summaries, and complex annotations used for the final figures are included in this repository.

To reproduce all figures:

```bash
python scripts/plot_figures.py
```

The figures are written to:

```text
figures/
├── gnina_sampling_main_strategy.png
├── gnina_sampling_main_strategy_full.png
├── gnina_ensemble_size_curves.png
└── gnina_runtime_pareto_front.png
```

`gnina_sampling_main_strategy.png` contains the Top-1 and Top-25 panels used in the blog post, while `gnina_sampling_main_strategy_full.png` additionally includes the Top-40 result.

The corresponding numerical summaries are written to:

```text
figures/csvs/
```

The processed inputs used by the plotting script are stored under:

```text
analysis/
├── metadata/
│   └── annotated_complexes.csv
├── parquets/
│   └── gnina_*_redock_exh_*_poses.parquet
└── runtime/
    ├── gnina_redock_runtime_invocations.csv
    └── gnina_redock_runtime_summary.csv
```

No docking calculations are required for figure reproduction.

## Reanalyse the precomputed docking results

The complete GNINA docking outputs used for this analysis are archived on [Zenodo](https://doi.org/10.5281/zenodo.22211565):

Download and extract the precomputed dataset with:

```bash
wget https://zenodo.org/records/22211566/files/A71EV2A_blopig_data.zip
unzip A71EV2A_blopig_data.zip && rm A71EV2A_blopig_data.zip
```

The archive contains an `A71EV2A_data/` directory with the complex-level data required by the analysis, together with the prepared inputs and GNINA docking outputs:

```text
A71EV2A_data/
├── annotated_complexes.csv
└── data/
    ├── <complex_id>/
    │   ├── ...
    │   ├── prepared_inputs/
    │   └── gnina/
    └── ...
```

The docking poses can then be reanalysed using the supplied SLURM launcher.

Before submitting jobs, check the SLURM settings near the top of `scripts/submit_analysis_jobs.sh` and adjust fields such as the partition, cluster, memory, and wall time for your computing environment.

By default, the analysis reuses any existing PoseBusters and OpenStructure intermediate results found within the complex directories. This makes rerunning the analysis much faster when the docking outputs have already been analysed.

```bash
./scripts/submit_analysis_jobs.sh A71EV2A_data/data/
```

This submits one CPU SLURM job for each configured sampling/exhaustiveness combination. The pose-level outputs are written to:

```text
analysis/parquets/
```

To redo the analysis from scratch, including the PoseBusters and OpenStructure checks, use:

```bash
./scripts/submit_analysis_jobs.sh --overwrite A71EV2A_data/data/
```

The generated SLURM jobs can also be inspected without submitting them:

```bash
./scripts/submit_analysis_jobs.sh --dry-run A71EV2A_data/data/
```

### Collect runtime statistics

GNINA invocation times are stored in the docking log files. Recreate the runtime summaries with:

```bash
python scripts/collect_runtimes.py \
    --root A71EV2A_data/data/
```

The runtime outputs are written to:

```text
analysis/runtime/
```

For multi-run strategies, runtime is reported as the sum of the component GNINA invocation times. These values therefore represent total compute rather than necessarily elapsed wall-clock time, since independent calculations may be run in parallel.

Once pose analysis and runtime collection are complete, regenerate the figures with:

```bash
python scripts/plot_figures.py
```

## Reproduce the docking workflow from scratch

To reproduce the full workflow, including input preparation and GNINA docking, start from the clean OpenBind EV-A71 2A protease dataset available from Zenodo.

Download and extract the dataset with:

```bash
wget https://zenodo.org/records/20798528/files/A71EV2A_data.zip
unzip A71EV2A_data.zip && rm A71EV2A_data.zip
```

The following commands assume the extracted dataset is available at:

```text
A71EV2A_data/data/
```

### Prepare docking inputs

Prepare the receptors and ligand conformers using:

```bash
python scripts/prepare_inputs.py \
    A71EV2A_data/data/ \
    --num-workers 8
```

Prepared inputs are written within each complex directory.

### Set up GNINA

GNINA must be available before running the docking calculations.

Instructions for installing GNINA and, if required, using the supplied Singularity definition are provided in: [software/README.md](https://github.com/Jnelen/A71EV2A-blopig/blob/main/software/README.md).

The calculations reported here used GNINA v1.3.2.

### Configure the docking calculations

The supplied docking configurations are located in: [scripts/docking_config/](https://github.com/Jnelen/A71EV2A-blopig/tree/main/scripts/docking_config)

They define the sampling strategies and exhaustiveness values used in this analysis.

The SLURM settings should be adjusted for the local computing environment before submission. In particular, site-specific fields such as the partition and cluster may need to be set.

The benchmark compares:

- one ligand starting conformer with five independent random seeds; and
- five ligand starting conformers using a single random seed.

Both strategies were evaluated at exhaustiveness values of 8, 16, and 32.

### Submit docking jobs

Submit the configured GNINA calculations with:

```bash
python scripts/submit_docking_jobs.py \
    A71EV2A_data/data/
```

To inspect the generated jobs without submitting them:

```bash
python scripts/submit_docking_jobs.py \
    A71EV2A_data/data/ \
    --dry-run
```

After docking has completed, continue with:

```bash
./scripts/submit_analysis_jobs.sh A71EV2A_data/data/

python scripts/collect_runtimes.py \
    --root A71EV2A_data/data/

python scripts/plot_figures.py
```

## Benchmark definition

The analysis evaluates GNINA redocking against the cognate crystallographic receptor structure.

The final benchmark excludes fragment-screen structures, crystallographic artefacts, and complexes whose experimental ligand fails the ground-truth PoseBusters validity check.

Strict pose-recovery success requires:

- ligand RMSD ≤ 2 Å;
- LDDT-PLI ≥ 0.8; and
- PoseBusters validity.

Three sampling strategies are compared.

### Single run

One ligand starting conformer and one GNINA random seed.

Five independent seed runs were performed, and the benchmark-wide success rate is averaged across these replicate runs.

### 5 seeds

Five independent GNINA runs are performed from the same ligand starting conformer.

The top eight poses from each run are pooled to give 40 poses and globally reranked using CNNscore.

### 5 conformers

Five independently generated ligand starting conformers are docked using the same GNINA random seed.

Eight poses from each conformer are pooled to give 40 poses and globally reranked using CNNscore.

## Repository structure

```text
A71EV2A-blopig/
├── analysis/
│   ├── metadata/
│   ├── parquets/
│   └── runtime/
├── figures/
│   └── csvs/
├── scripts/
│   ├── docking_config/
│   ├── prepare_inputs.py
│   ├── dock_complexes.py
│   ├── submit_docking_jobs.py
│   ├── analyse_docking.py
│   ├── submit_analysis_jobs.sh
│   ├── collect_runtimes.py
│   └── plot_figures.py
├── software/
│   └── README.md
├── environment.yml
├── LICENSE
└── README.md
```

## Data availability

The clean OpenBind EV-A71 2A protease dataset used to reproduce the docking calculations from scratch is available from [Zenodo](https://zenodo.org/records/20798527). To download directly:

```bash
wget https://zenodo.org/records/20798528/files/A71EV2A_data.zip
```

The complete dataset containing the prepared inputs and GNINA docking outputs used in this analysis is also available from [Zenodo](https://doi.org/10.5281/zenodo.22211565). To download directly:

```bash
wget https://zenodo.org/records/22211566/files/A71EV2A_blopig_data.zip
```

The processed pose-analysis [tables](https://github.com/Jnelen/A71EV2A-blopig/tree/main/figures/csvs) and [runtime summaries](https://github.com/Jnelen/A71EV2A-blopig/tree/main/analysis/runtime) required to reproduce the final [figures](https://github.com/Jnelen/A71EV2A-blopig/tree/main/figures) are included directly in this repository.

## Citation

If you use the underlying OpenBind dataset, please cite the corresponding OpenBind data release:

[https://www.biorxiv.org/content/10.64898/2026.08.27.747600](https://www.biorxiv.org/content/10.64898/2026.08.27.747600)
