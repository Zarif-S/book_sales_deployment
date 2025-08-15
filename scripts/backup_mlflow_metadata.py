#!/usr/bin/env python3
"""
MLflow Metadata Backup Script

Exports experiment metadata to JSON files in GCS bucket for persistence.
Run this after each pipeline run to backup experiment tracking data.
"""

import mlflow
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def backup_mlflow_metadata(
    mlflow_uri: str = "https://mlflow-tracking-server-1076639696283.europe-west2.run.app",
    backup_dir: str = "data/mlflow_backups"
):
    """Export all MLflow experiments, runs, and models to JSON files"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Create backup directory
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🔄 Backing up MLflow metadata from: {mlflow_uri}")
    
    # 1. Export experiments
    experiments = client.search_experiments()
    experiments_data = []
    
    for exp in experiments:
        exp_data = {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "creation_time": exp.creation_time,
            "last_update_time": exp.last_update_time
        }
        experiments_data.append(exp_data)
        
        print(f"  📁 Experiment: {exp.name} (ID: {exp.experiment_id})")
    
    # Save experiments
    exp_file = backup_path / f"experiments_{timestamp}.json"
    with open(exp_file, 'w') as f:
        json.dump(experiments_data, f, indent=2)
    
    # 2. Export all runs
    all_runs_data = []
    
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1000
        )
        
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
            all_runs_data.append(run_data)
        
        print(f"  🏃 Exported {len(runs)} runs from {exp.name}")
    
    # Save runs
    runs_file = backup_path / f"runs_{timestamp}.json"
    with open(runs_file, 'w') as f:
        json.dump(all_runs_data, f, indent=2)
    
    # 3. Export registered models
    models = client.search_registered_models()
    models_data = []
    
    for model in models:
        # Get all versions for this model
        versions = client.search_model_versions(f"name='{model.name}'")
        
        model_data = {
            "name": model.name,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "description": model.description,
            "versions": []
        }
        
        for version in versions:
            version_data = {
                "version": version.version,
                "stage": version.current_stage,
                "description": version.description,
                "creation_timestamp": version.creation_timestamp,
                "last_updated_timestamp": version.last_updated_timestamp,
                "source": version.source,
                "run_id": version.run_id,
                "status": version.status
            }
            model_data["versions"].append(version_data)
        
        models_data.append(model_data)
        print(f"  🏷️  Model: {model.name} ({len(versions)} versions)")
    
    # Save models
    models_file = backup_path / f"models_{timestamp}.json"
    with open(models_file, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    # 4. Create summary
    summary = {
        "backup_timestamp": timestamp,
        "mlflow_uri": mlflow_uri,
        "experiments_count": len(experiments_data),
        "runs_count": len(all_runs_data),
        "models_count": len(models_data),
        "files": {
            "experiments": str(exp_file),
            "runs": str(runs_file),
            "models": str(models_file)
        }
    }
    
    summary_file = backup_path / f"backup_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Backup completed:")
    print(f"   📊 {len(experiments_data)} experiments")
    print(f"   🏃 {len(all_runs_data)} runs")
    print(f"   🏷️  {len(models_data)} models")
    print(f"   📁 Saved to: {backup_path}")
    
    # 5. Upload to GCS (optional)
    try:
        import subprocess
        gcs_path = f"gs://book-sales-deployment-artifacts/mlflow-backups/{timestamp}/"
        
        cmd = f"gsutil -m cp -r {backup_path}/* {gcs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ☁️  Uploaded to: {gcs_path}")
        else:
            print(f"   ⚠️  GCS upload failed: {result.stderr}")
    except Exception as e:
        print(f"   ⚠️  GCS upload not available: {e}")
    
    return summary

if __name__ == "__main__":
    backup_mlflow_metadata()