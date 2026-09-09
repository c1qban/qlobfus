# Semantics-Preserving Code Obfuscation with Reinforcement Learning

This repository contains the research artifact for **Semantics-Preserving Code Obfuscation with Reinforcement Learning**.

QLObfus models source-level C obfuscation as a constrained sequential decision problem. A policy chooses an obfuscation operator, a concrete program location, and a parameterization; each step is checked by compilation, executable test harnesses, differential execution, and lightweight fuzzing. The system compares learned policies against internal random/rule-based baselines and an optional Tigress baseline.

## What Is Included

- `api/`: unified command-line entry point.
- `obfuscator/`: C source and Clang/LLVM-aware transformation operators.
- `rl/`: Gymnasium environment and Maskable PPO training integration.
- `verifier/`: compilation, differential execution, fuzzing, and manifest utilities.
- `eval/`: LLVM, obfuscation, resilience, and reporting metrics.
- `scripts/`: input preparation, training, evaluation, baselines, summaries, and paper-support utilities.
- `configs/`: benchmark, policy, reward, and experiment configurations used by the artifact.
- `tests/`: unit and integration tests for the research pipeline.
- `docs/`: paper draft and supporting research notes intended for public release.

See `REPRODUCIBILITY.md` for the recommended experiment workflow.

## Requirements

Recommended environment:

- Python 3.10 or newer
- Clang/LLVM tools available on `PATH`, or configured in experiment YAML files
- A C compiler compatible with the evaluated programs
- Optional: Tigress for the external obfuscator baseline

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

For a local Windows LLVM installation, update the compiler path in the relevant config, for example:

```yaml
verification:
  compile:
    compiler: "C:\\Program Files\\LLVM\\bin\\clang.exe"
```

## Command-Line Interface

The unified CLI is exposed through `api.cli`:

```bash
python -m api.cli --help
```

Available commands include:

```bash
python -m api.cli prepare-dataset --config configs/benchmark/project_codenet_subset_10x5_prepare.yaml
python -m api.cli train --config configs/experiments/protocol_v5_mppo_seed1.yaml
python -m api.cli run-baselines --config configs/experiments/protocol_v5_final_test_internal_baselines.yaml
python -m api.cli evaluate --config configs/experiments/protocol_v5_final_test_internal_baselines.yaml
python -m api.cli report --help
```

Most scripts can also be run directly as Python modules, for example:

```bash
python -m scripts.train --config configs/experiments/protocol_v5_mppo_seed1.yaml
python -m scripts.run_baselines --config configs/experiments/protocol_v5_final_test_internal_baselines.yaml
```

## Minimal Smoke Workflow

1. Place a small Project CodeNet-style C corpus at the path expected by the selected config.
2. Adjust compiler paths in `configs/benchmark/project_codenet_subset_10x5_prepare.yaml`.
3. Prepare manifests and harness metadata:

```bash
python -m api.cli prepare-dataset --config configs/benchmark/project_codenet_subset_10x5_prepare.yaml
```

4. Run a small baseline or policy experiment:

```bash
python -m api.cli run-baselines --config configs/benchmark/baselines_smoke.example.yaml
```

## Paper-Artifact Workflow

The main paper protocol uses Project CodeNet C programs with problem-level holdout splits, harness-clean filtering, multi-seed Maskable PPO training, frozen checkpoint evaluation, internal baselines, optional Tigress baselines, cross-compiler validation, deobfuscation attacks, and summary generation.

Start with `REPRODUCIBILITY.md`, then inspect:

- `configs/experiments/protocol_v5_mppo_seed1.yaml`
- `configs/experiments/protocol_v5_final_test_internal_baselines.yaml`
- `configs/experiments/protocol_v5_final_test_tigress_baseline.yaml`
- `scripts/run_paper_smoke.ps1`
- `scripts/run_paper_summaries.ps1`

## Citation

If you use this repository, please cite the associated paper and this artifact. A machine-readable citation template is provided in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.

Third-party corpora and external tools, including Project CodeNet and Tigress, are governed by their own licenses and are not relicensed by this repository.
