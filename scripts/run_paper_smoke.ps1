$ErrorActionPreference = "Stop"

Write-Host "[1/4] Running unit tests..."
python -m unittest discover -s tests -p "test*.py" -v

Write-Host "[2/4] Auditing formal_eval_30_v2 harnesses..."
python -m scripts.audit_harness_quality --include-excluded

Write-Host "[3/4] Regenerating corrected summary..."
python -m scripts.apply_experiment_corrections

Write-Host "[4/4] Writing reproducibility manifest..."
python -m scripts.write_repro_manifest

Write-Host "Paper smoke check complete."
