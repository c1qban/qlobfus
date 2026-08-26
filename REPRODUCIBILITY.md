# Reproducibility Guide

This document describes the command sequence used to reproduce the paper-style experiments from the repository configuration files.

## Environment Setup

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with:

```bash
source .venv/bin/activate
```

Install Clang/LLVM and ensure the compiler tools are available:

```bash
clang --version
opt --version
```

On Windows, either add LLVM to `PATH` or update compiler paths in config files such as:

- `configs/benchmark/project_codenet_subset_10x5_prepare.yaml`
- `configs/env/local.example.yaml`

Tigress is optional and only required for Tigress baseline experiments.

## CLI Entry Point

Use the unified CLI:

```bash
python -m api.cli --help
```

The main subcommands are:

```bash
python -m api.cli prepare-dataset --help
python -m api.cli train --help
python -m api.cli run-baselines --help
python -m api.cli evaluate --help
python -m api.cli report --help
```

## Smoke Workflow

Use the small-subset configs for a quick sanity run:

```bash
python -m api.cli prepare-dataset --config configs/benchmark/project_codenet_subset_10x5_prepare.yaml
python -m api.cli run-baselines --config configs/benchmark/baselines_smoke.example.yaml
```

The selected YAML config controls input paths, compiler settings, filtering rules, harness settings, and output locations.

## Main Paper Protocol

The main protocol is represented by the `protocol_v5` experiment configs.

### Input Preparation

Prepare the strict Project CodeNet C protocol:

```bash
python -m api.cli prepare-dataset --config configs/benchmark/project_codenet_c_500x4_prepare_strict.yaml
```

### Policy Training

Train Maskable PPO seeds:

```bash
python -m api.cli train --config configs/experiments/protocol_v5_mppo_seed1.yaml
python -m api.cli train --config configs/experiments/protocol_v5_mppo_seed7.yaml
python -m api.cli train --config configs/experiments/protocol_v5_mppo_seed13.yaml
python -m api.cli train --config configs/experiments/protocol_v5_mppo_seed21.yaml
python -m api.cli train --config configs/experiments/protocol_v5_mppo_seed42.yaml
```

### Internal Baselines

Run random and rule-based internal baselines:

```bash
python -m api.cli run-baselines --config configs/experiments/protocol_v5_final_test_internal_baselines.yaml
```

### Tigress Baseline

Run the optional Tigress baseline after installing Tigress and updating the relevant tool path/config:

```bash
python -m api.cli run-baselines --config configs/experiments/protocol_v5_final_test_tigress_baseline.yaml
```

### Summaries

The repository includes convenience PowerShell scripts for paper summaries:

```powershell
.\scripts\run_paper_smoke.ps1
.\scripts\run_paper_summaries.ps1
```

## Artifact Metadata

For paper submission, publish a frozen release with:

- Git tag, for example `v1.0-paper-artifact`
- Python version
- OS version
- Clang/LLVM version
- Tigress version, if used
- dependency lock file, if available
- exact commands used to produce each paper table

## Known Limitations

- Semantic preservation is empirically checked by compilation, harness execution, differential execution, and fuzzing; it is not a formal equivalence proof.
- Source-level transformations may be simplified by compiler optimization.
- Tigress has its own license, installation process, compatibility profile, and failure modes.
