$ErrorActionPreference = "Stop"

Write-Host "[1/5] Regenerating corrected summary..."
python -m scripts.apply_experiment_corrections

Write-Host "[2/5] Regenerating paper figures..."
python -m scripts.plot_results

Write-Host "[3/5] Plotting PPO training curves..."
python -m scripts.plot_training_curves

Write-Host "[4/5] Auditing harness quality..."
python -m scripts.audit_harness_quality --include-excluded

Write-Host "[5/5] Writing reproducibility manifest..."
python -m scripts.write_repro_manifest

Write-Host "Paper summary artifacts are up to date."
