#!/usr/bin/env python3
"""
Upload MLflow ARIMA Models to Google Cloud Storage for Vertex AI Deployment

This script downloads models from your MLflow registry and uploads them to GCS
in a format that can be deployed using Vertex AI's pre-built containers.

Usage:
    python 02_upload_models_to_gcs.py --upload-all
    python deploy/02_upload_models_to_gcs.py --model-name arima_book_9780722532935
    python deploy/02_upload_models_to_gcs.py --list-models
"""

import argparse
import logging
import os
import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.statsmodels
    from google.cloud import storage
    import joblib
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.info("Run: poetry install")
    exit(1)


class ModelUploader:
    """Upload MLflow models to GCS for Vertex AI deployment."""

    def __init__(self,
                 project_id: str = "upheld-apricot-468313-e0",
                 bucket_name: str = "book-sales-deployment-artifacts",
                 mlflow_uri: str = "https://mlflow-tracking-server-1076639696283.europe-west2.run.app"):
        """Initialize the uploader."""
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.mlflow_uri = mlflow_uri

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_uri)

        # Initialize GCS client
        self.gcs_client = storage.Client(project=project_id)
        self.bucket = self.gcs_client.bucket(bucket_name)

        logger.info(f"Initialized uploader:")
        logger.info(f"  Project: {project_id}")
        logger.info(f"  Bucket: {bucket_name}")
        logger.info(f"  MLflow: {mlflow_uri}")

    def list_available_models(self) -> List[Dict]:
        """List all available ARIMA models in MLflow registry."""
        try:
            models = []
            logger.info("📋 Checking MLflow registry for ARIMA models...")

            for model in mlflow.search_registered_models():
                if model.name.startswith("arima_book_"):
                    latest_version = model.latest_versions[0] if model.latest_versions else None
                    models.append({
                        'name': model.name,
                        'version': latest_version.version if latest_version else 'N/A',
                        'stage': latest_version.current_stage if latest_version else 'N/A',
                        'isbn': model.name.replace("arima_book_", ""),
                        'source': latest_version.source if latest_version else None,
                        'run_id': latest_version.run_id if latest_version else None
                    })

            logger.info(f"Found {len(models)} ARIMA models in registry")
            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def download_model_from_mlflow(self, model_name: str, version: str = "latest") -> Optional[str]:
        """Download model from MLflow to /outputs/models/mlflow-downloads directory."""
        try:
            logger.info(f"📥 Downloading {model_name} v{version} from MLflow...")

            # Create mlflow-downloads directory to separate from training pipeline outputs
            outputs_dir = Path(__file__).parent.parent / "outputs" / "models" / "mlflow-downloads" / model_name
            outputs_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = str(outputs_dir)
            model_path = os.path.join(temp_dir, "model")

            # Download model
            model_uri = f"models:/{model_name}/{version}"
            downloaded_model = mlflow.statsmodels.load_model(model_uri)

            # Save in scikit-learn compatible format for Vertex AI
            # We'll save as joblib pickle which Vertex AI pre-built containers can load
            joblib_model_path = os.path.join(temp_dir, "model.joblib")
            try:
                joblib.dump(downloaded_model, joblib_model_path)
                logger.info(f"✅ Model saved as joblib: {joblib_model_path}")
            except Exception as joblib_error:
                logger.warning(f"Failed to save as joblib: {joblib_error}")

                # Save as regular pickle backup in outputs/models folder only if joblib fails
                isbn = model_name.replace("arima_book_", "")
                outputs_dir = Path(__file__).parent.parent / "outputs" / "models"
                outputs_dir.mkdir(parents=True, exist_ok=True)

                pickle_model_path = outputs_dir / f"{model_name}_model.pkl"
                with open(pickle_model_path, 'wb') as f:
                    pickle.dump(downloaded_model, f)
                logger.info(f"✅ Model saved as pickle backup: {pickle_model_path}")

                # Also create a copy in temp dir for deployment
                temp_pickle_path = os.path.join(temp_dir, "model.pkl")
                with open(temp_pickle_path, 'wb') as f:
                    pickle.dump(downloaded_model, f)

            # Get model metadata
            try:
                client = mlflow.tracking.MlflowClient()
                # If version is "latest", get the actual latest version number
                if version == "latest":
                    registered_model = client.get_registered_model(model_name)
                    if registered_model.latest_versions:
                        actual_version = registered_model.latest_versions[0].version
                    else:
                        raise ValueError(f"No versions found for model {model_name}")
                else:
                    actual_version = version
                
                model_version = client.get_model_version(model_name, actual_version)

                # Save metadata
                metadata = {
                    "model_name": model_name,
                    "version": version,
                    "stage": model_version.current_stage,
                    "creation_timestamp": model_version.creation_timestamp,
                    "last_updated_timestamp": model_version.last_updated_timestamp,
                    "source": model_version.source,
                    "run_id": model_version.run_id,
                    "tags": model_version.tags,
                    "isbn": model_name.replace("arima_book_", ""),
                    "model_type": "SARIMA",
                    "framework": "statsmodels",
                    "deployment_format": "joblib"
                }

                metadata_path = os.path.join(temp_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            except Exception as meta_error:
                logger.warning(f"Could not save metadata: {meta_error}")

            # Create a simple prediction script for Vertex AI
            self._create_prediction_script(temp_dir, model_name)

            logger.info(f"✅ Model downloaded to: {temp_dir}")
            return temp_dir

        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return None

    def _create_prediction_script(self, model_dir: str, model_name: str):
        """Create a prediction script for Vertex AI pre-built container."""
        isbn = model_name.replace("arima_book_", "")

        prediction_script = f'''#!/usr/bin/env python3
"""
Prediction script for Vertex AI pre-built container.
Model: {model_name}
ISBN: {isbn}
"""

import joblib
import pickle
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Load the model
try:
    model = joblib.load("/opt/ml/model/model.joblib")
except:
    with open("/opt/ml/model/model.pkl", "rb") as f:
        model = pickle.load(f)

def predict(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make predictions for Vertex AI."""
    predictions = []

    for instance in instances:
        try:
            # Parse input
            steps = instance.get("steps", 12)
            confidence_intervals = instance.get("return_confidence_intervals", False)
            confidence_level = instance.get("confidence_level", 0.95)

            # Make forecast
            if confidence_intervals:
                forecast_result = model.get_forecast(steps=steps)
                forecast_values = forecast_result.predicted_mean.tolist()

                # Get confidence intervals
                conf_int = forecast_result.conf_int(alpha=1-confidence_level)
                lower_bounds = conf_int.iloc[:, 0].tolist()
                upper_bounds = conf_int.iloc[:, 1].tolist()

                prediction = {{
                    "forecast": forecast_values,
                    "confidence_intervals": {{
                        "lower": lower_bounds,
                        "upper": upper_bounds,
                        "confidence_level": confidence_level
                    }},
                    "steps": steps,
                    "model_name": "{model_name}",
                    "isbn": "{isbn}"
                }}
            else:
                forecast = model.forecast(steps=steps)
                forecast_values = forecast.tolist() if hasattr(forecast, 'tolist') else [float(forecast)]

                prediction = {{
                    "forecast": forecast_values,
                    "steps": steps,
                    "model_name": "{model_name}",
                    "isbn": "{isbn}"
                }}

            predictions.append(prediction)

        except Exception as e:
            predictions.append({{
                "error": str(e),
                "model_name": "{model_name}",
                "isbn": "{isbn}"
            }})

    return predictions

if __name__ == "__main__":
    # Test prediction
    test_instances = [{{"steps": 4}}]
    result = predict(test_instances)
    print(json.dumps(result, indent=2, default=str))
'''

        script_path = os.path.join(model_dir, "predictor.py")
        with open(script_path, 'w') as f:
            f.write(prediction_script)

        logger.info(f"Created prediction script: {script_path}")

    def upload_model_to_gcs(self, model_dir: str, model_name: str) -> Optional[str]:
        """Upload model directory to GCS."""
        try:
            isbn = model_name.replace("arima_book_", "")
            gcs_model_path = f"models/{model_name}/latest"

            logger.info(f"📤 Uploading {model_name} to gs://{self.bucket_name}/{gcs_model_path}")

            # Collect all files first to show total progress
            all_files = []
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, model_dir)
                    gcs_file_path = f"{gcs_model_path}/{relative_path}"
                    all_files.append((local_file_path, gcs_file_path))

            # Upload all files with progress bar
            uploaded_files = []
            with tqdm(total=len(all_files), desc=f"Uploading {model_name}",
                     unit="file", ncols=100) as pbar:
                for local_file_path, gcs_file_path in all_files:
                    try:
                        # Get file size for more detailed progress
                        file_size = os.path.getsize(local_file_path)
                        pbar.set_postfix({
                            'file': os.path.basename(local_file_path),
                            'size': f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f}MB"
                        })

                        blob = self.bucket.blob(gcs_file_path)
                        blob.upload_from_filename(local_file_path)
                        uploaded_files.append(gcs_file_path)

                        pbar.update(1)
                        logger.info(f"  ✅ Uploaded: {gcs_file_path}")
                    except Exception as file_error:
                        logger.error(f"  ❌ Failed to upload {gcs_file_path}: {file_error}")
                        pbar.update(1)

            # Create a model artifact URI for Vertex AI
            model_uri = f"gs://{self.bucket_name}/{gcs_model_path}"

            logger.info(f"✅ Model uploaded successfully!")
            logger.info(f"   GCS URI: {model_uri}")
            logger.info(f"   Files uploaded: {len(uploaded_files)}/{len(all_files)}")

            return model_uri

        except Exception as e:
            logger.error(f"Failed to upload model {model_name}: {e}")
            return None

    def upload_single_model(self, model_name: str, version: str = "latest") -> Optional[str]:
        """Upload a single model to GCS."""
        # Download from MLflow
        model_dir = self.download_model_from_mlflow(model_name, version)
        if not model_dir:
            return None

        try:
            # Upload to GCS
            model_uri = self.upload_model_to_gcs(model_dir, model_name)
            return model_uri
        finally:
            # Clean up temporary directory
            if model_dir and os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                logger.info(f"🧹 Cleaned up temporary directory: {model_dir}")

    def upload_all_models(self) -> Dict[str, str]:
        """Upload all available models to GCS."""
        models = self.list_available_models()
        if not models:
            logger.warning("No models found for upload")
            return {}

        results = {}
        logger.info(f"🚀 Starting upload of {len(models)} models to GCS...")

        # Use progress bar for overall model upload progress
        with tqdm(total=len(models), desc="Overall Progress",
                 unit="model", ncols=100, position=0) as overall_pbar:

            for i, model in enumerate(models):
                model_name = model['name']
                version = model['version']
                isbn = model['isbn']

                overall_pbar.set_postfix({'current': f"ISBN {isbn}"})
                logger.info(f"📦 Processing {model_name} v{version} ({i+1}/{len(models)})")

                model_uri = self.upload_single_model(model_name, version)

                if model_uri:
                    results[model_name] = model_uri
                    logger.info(f"✅ {model_name} uploaded successfully")
                else:
                    logger.error(f"❌ {model_name} upload failed")

                overall_pbar.update(1)

        # Summary
        success_count = len(results)
        total_count = len(models)
        logger.info(f"📊 Upload complete: {success_count}/{total_count} models uploaded")

        if results:
            logger.info("📋 Uploaded models:")
            for model_name, uri in results.items():
                logger.info(f"  📖 {model_name}: {uri}")

        return results

    def list_uploaded_models(self) -> List[str]:
        """List models already uploaded to GCS."""
        try:
            prefix = "models/"
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter="/")

            model_paths = []
            for blob in blobs:
                if blob.name.endswith("/"):
                    continue
                model_paths.append(f"gs://{self.bucket_name}/{blob.name}")

            # Get unique model directories
            model_dirs = set()
            for path in model_paths:
                # Extract model directory (e.g., models/arima_book_123/latest/)
                parts = path.replace(f"gs://{self.bucket_name}/", "").split("/")
                if len(parts) >= 3 and parts[0] == "models":
                    model_dir = f"gs://{self.bucket_name}/{'/'.join(parts[:3])}"
                    model_dirs.add(model_dir)

            logger.info(f"Found {len(model_dirs)} uploaded models in GCS")
            return sorted(list(model_dirs))

        except Exception as e:
            logger.error(f"Failed to list uploaded models: {e}")
            return []


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Upload MLflow models to GCS for Vertex AI deployment")

    parser.add_argument("--upload-all", action="store_true", help="Upload all available models")
    parser.add_argument("--model-name", help="Upload specific model by name")
    parser.add_argument("--model-version", default="latest", help="Model version to upload")
    parser.add_argument("--list-models", action="store_true", help="List available models in MLflow")
    parser.add_argument("--list-uploaded", action="store_true", help="List models already uploaded to GCS")

    parser.add_argument("--project-id", default="upheld-apricot-468313-e0", help="GCP project ID")
    parser.add_argument("--bucket-name", default="book-sales-deployment-artifacts", help="GCS bucket name")
    parser.add_argument("--mlflow-uri",
                       default="https://mlflow-tracking-server-1076639696283.europe-west2.run.app",
                       help="MLflow tracking server URI")

    args = parser.parse_args()

    if not any([args.upload_all, args.model_name, args.list_models, args.list_uploaded]):
        parser.print_help()
        return

    # Initialize uploader
    try:
        uploader = ModelUploader(
            project_id=args.project_id,
            bucket_name=args.bucket_name,
            mlflow_uri=args.mlflow_uri
        )
    except Exception as e:
        logger.error(f"Failed to initialize uploader: {e}")
        logger.info("Make sure you're authenticated with: gcloud auth login")
        return

    # Execute requested action
    if args.list_models:
        models = uploader.list_available_models()
        if models:
            logger.info(f"📋 Available models in MLflow ({len(models)}):")
            for model in models:
                logger.info(f"  📖 {model['name']} v{model['version']} ({model['stage']}) - ISBN: {model['isbn']}")
        else:
            logger.info("No ARIMA models found in MLflow registry")

    if args.list_uploaded:
        uploaded = uploader.list_uploaded_models()
        if uploaded:
            logger.info(f"📋 Models uploaded to GCS ({len(uploaded)}):")
            for model_uri in uploaded:
                logger.info(f"  📦 {model_uri}")
        else:
            logger.info("No models found in GCS")

    if args.upload_all:
        logger.info("🚀 Starting upload of all models...")
        results = uploader.upload_all_models()

        if results:
            logger.info("🎉 All uploads completed!")
            logger.info("Next step: Deploy models to endpoints using 03_deploy_to_vertex_endpoints.py")
        else:
            logger.warning("No models were uploaded")

    elif args.model_name:
        logger.info(f"🚀 Uploading single model: {args.model_name}")
        model_uri = uploader.upload_single_model(args.model_name, args.model_version)

        if model_uri:
            logger.info(f"✅ Model uploaded: {model_uri}")
            logger.info("Next step: Deploy to endpoint using 03_deploy_to_vertex_endpoints.py")
        else:
            logger.error(f"❌ Failed to upload {args.model_name}")


if __name__ == "__main__":
    main()
