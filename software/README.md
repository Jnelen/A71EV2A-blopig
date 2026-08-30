# Software Setup

This directory contains installation instructions for third-party software used in the OpenBind docking benchmarks.

These tools are **not included** in the repository and must be installed separately.

---

## Expected directory structure

After installation, the directory structure should look similar to:

```text
software/
├── gnina/
│   ├── gnina
│   └── gnina_singularity.sif          # optional
```

---

## Optional Singularity / Apptainer environments

Optional Singularity containers is available for GNINA.

These containers do **not** contain the GNINA executable themselve.  
Instead, they provide reproducible runtime dependency environments for the locally installed software.

The containers are primarily intended for:

* HPC environments
* CUDA compatibility
* avoiding dependency conflicts
* improving reproducibility across systems

The paths to both the software executables and optional Singularity images should be configured in:

```text
docking_config.yaml
```

---

## [GNINA](https://link.springer.com/article/10.1186/s13321-025-00973-x)

**Recommended version:** `v1.3.2`  

### Installation

Create the software directory:

```bash
mkdir -p software/gnina
```

Download the precompiled binary:

```bash
wget https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2

mv gnina.1.3.2 software/gnina/gnina
chmod +x software/gnina/gnina
```

### Optional Singularity dependency container

```bash
singularity pull software/gnina/gnina_singularity.sif \
    oras://ghcr.io/jnelen/gnina_singularity:v1
```

---

## Notes

* Ensure all binaries are executable
* Use consistent software versions for reproducibility
* Relative paths in `docking_config.yaml` are resolved relative to the repository root
* Refer to each tool’s official documentation for advanced configuration options