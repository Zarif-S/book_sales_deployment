# MLflow Metadata Backup & Restore

This folder contains scripts to backup and restore MLflow experiment metadata.

## Quick Usage

### Backup after pipeline runs:
```bash
# Backup current MLflow metadata
poetry run python mlflow/backup_metadata.py

# List available backups
poetry run python mlflow/backup_metadata.py --list
```

### Recommended workflow:
```bash
# 1. Run your pipeline
poetry run python pipelines/zenml_pipeline.py

# 2. Backup metadata immediately after
poetry run python mlflow/backup_metadata.py
```

## What gets backed up:

✅ **Preserved (Important)**:
- Experiment definitions and names
- Run metrics (MAE, RMSE, MAPE, accuracy, etc.)
- Run parameters (n_trials, ISBN, configuration)
- Model registry metadata (versions, stages)
- Tags and run information
- Performance tracking over time

❌ **Not backed up (Can regenerate)**:
- Model artifacts (large .pkl files)
- Model binaries
- Training data artifacts

## Storage:

- **Local**: `mlflow/backups/` folder
- **Cloud**: `gs://book-sales-deployment-artifacts/mlflow-metadata-backups/`

## If you lose MLflow server data:

1. **Model artifacts are safe** - they're in GCS permanently
2. **Metadata is backed up** - restore from JSON files
3. **You can view performance history** - metrics preserved in backups
4. **Models can be re-registered** - from existing GCS artifacts

## Backup frequency:

**Recommended**: After each significant pipeline run
- Development: Weekly
- Production: After every pipeline run

## File structure:

```
mlflow/backups/
├── experiments_20250815_232928.json    # Experiment definitions
├── runs_20250815_232928.json          # Runs with metrics/params
├── models_20250815_232928.json        # Model registry metadata
└── backup_summary_20250815_232928.json # Backup info
```

## Restore process (if needed):

If your MLflow server loses data:

1. **Don't panic** - model artifacts are safe in GCS
2. **Check backups**: `python mlflow/backup_metadata.py --list`
3. **Manual restore**: Load JSON files and re-register models
4. **Re-run pipeline**: Models will be detected and reused from GCS

## Why this approach:

- **Lightweight**: Only backs up metadata (KB not GB)
- **Fast**: Backup takes seconds
- **Comprehensive**: Preserves experiment tracking history
- **Cost-effective**: No duplicate model storage
- **Recovery-friendly**: Can rebuild from backups + GCS artifacts

## Cloud SQL alternative:

If you want fully persistent metadata:
- **Cost**: ~$7-25/month for Cloud SQL
- **Benefit**: No backup needed, automatic persistence
- **Setup**: Update MLflow server to use Cloud SQL backend

Current setup with backups is **free** and sufficient for most use cases.