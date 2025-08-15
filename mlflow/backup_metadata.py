#!/usr/bin/env python3
"""
MLflow Metadata Backup - Lightweight Version

Backs up only experiment metadata (runs, metrics, params) - NOT model artifacts.
Model artifacts stay in GCS and can be regenerated if needed.
This preserves your experiment tracking history and performance metrics.

Usage:
    python mlflow/backup_metadata.py
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

def backup_metadata_only(
    mlflow_uri: str = "https://mlflow-tracking-server-1076639696283.europe-west2.run.app",
    backup_dir: str = "mlflow/backups"
):
    """Export MLflow experiment metadata (no model artifacts) to JSON files"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Create backup directory
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🔄 Backing up MLflow metadata (experiments, runs, metrics)")
    print(f"   Source: {mlflow_uri}")
    print(f"   Target: {backup_path}")
    
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
        
    print(f"  📁 Found {len(experiments_data)} experiments")
    
    # 2. Export all runs with metrics and params
    all_runs_data = []
    total_runs = 0
    
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1000
        )
        
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "experiment_name": exp.name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                # Store artifact URI for reference but don't backup artifacts
                "artifact_uri": run.info.artifact_uri
            }
            all_runs_data.append(run_data)
        
        total_runs += len(runs)
        if len(runs) > 0:
            print(f"  🏃 {exp.name}: {len(runs)} runs")
    
    # 3. Export model registry metadata (without model files)
    models = client.search_registered_models()
    models_data = []
    
    for model in models:
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
        print(f"  🏷️  {model.name}: {len(versions)} versions")
    
    # Save all data
    backup_files = {}
    
    # Experiments
    exp_file = backup_path / f"experiments_{timestamp}.json"
    with open(exp_file, 'w') as f:
        json.dump(experiments_data, f, indent=2)
    backup_files["experiments"] = str(exp_file)
    
    # Runs (contains metrics, params, tags)
    runs_file = backup_path / f"runs_{timestamp}.json"
    with open(runs_file, 'w') as f:
        json.dump(all_runs_data, f, indent=2)
    backup_files["runs"] = str(runs_file)
    
    # Model registry
    models_file = backup_path / f"models_{timestamp}.json"
    with open(models_file, 'w') as f:
        json.dump(models_data, f, indent=2)
    backup_files["models"] = str(models_file)
    
    # Create summary
    summary = {
        "backup_timestamp": timestamp,
        "backup_date": datetime.now().isoformat(),
        "mlflow_uri": mlflow_uri,
        "stats": {
            "experiments": len(experiments_data),
            "runs": len(all_runs_data),
            "models": len(models_data)
        },
        "files": backup_files,
        "note": "Metadata only - model artifacts remain in original GCS locations"
    }
    
    summary_file = backup_path / f"backup_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Metadata backup completed:")
    print(f"   📊 {len(experiments_data)} experiments")
    print(f"   🏃 {len(all_runs_data)} runs")
    print(f"   🏷️  {len(models_data)} models")
    print(f"   📁 Saved to: {backup_path}")
    
    # Upload to GCS
    try:
        import subprocess
        gcs_path = f"gs://book-sales-deployment-artifacts/mlflow-metadata-backups/{timestamp}/"
        
        cmd = f"gsutil -m cp {backup_path}/*_{timestamp}.json {gcs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ☁️  Uploaded to: {gcs_path}")
            print(f"   💾 Total files: {len(backup_files) + 1} JSON files")
        else:
            print(f"   ⚠️  GCS upload failed: {result.stderr}")
    except Exception as e:
        print(f"   ⚠️  GCS upload not available: {e}")
    
    print(f"\n📋 What's backed up:")
    print(f"   ✅ Experiment definitions and metadata")
    print(f"   ✅ Run metrics (MAE, RMSE, MAPE, etc.)")
    print(f"   ✅ Run parameters (n_trials, ISBN, etc.)")
    print(f"   ✅ Model registry versions and stages")
    print(f"   ✅ Tags and run information")
    print(f"   ❌ Model artifacts (stay in GCS, can regenerate)")
    
    return summary

def list_backups(backup_dir: str = "mlflow/backups"):
    """List available backups"""
    backup_path = Path(backup_dir)
    
    if not backup_path.exists():
        print(f"📁 No backups found in {backup_dir}")
        return
    
    summaries = list(backup_path.glob("backup_summary_*.json"))
    
    if not summaries:
        print(f"📁 No backup summaries found in {backup_dir}")
        return
    
    print(f"📋 Available backups in {backup_dir}:")
    
    for summary_file in sorted(summaries, reverse=True):
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            
            timestamp = summary["backup_timestamp"]
            stats = summary["stats"]
            date = summary.get("backup_date", "Unknown date")
            
            print(f"   📅 {timestamp} ({date})")
            print(f"      📊 {stats['experiments']} experiments, {stats['runs']} runs, {stats['models']} models")
            
        except Exception as e:
            print(f"   ⚠️  Could not read {summary_file}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup MLflow metadata")
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--uri", default="https://mlflow-tracking-server-1076639696283.europe-west2.run.app", 
                       help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    if args.list:
        list_backups()
    else:
        backup_metadata_only(mlflow_uri=args.uri)