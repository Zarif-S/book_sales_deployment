# MLOps Deployment Guide - Book Sales ARIMA Pipeline

This guide contains step-by-step instructions for deploying the complete MLOps pipeline with ZenML, Vertex AI, and MLflow using GCS bucket data storage.

## 🏗️ Architecture Overview

### Production MLOps Stack

| Component | Technology | Purpose | URL |
|-----------|------------|---------|-----|
| **Pipeline Orchestration** | ZenML | ML pipeline management & artifact lineage | https://zenml-server-1076639696283.europe-west2.run.app |
| **Cloud Execution** | Google Vertex AI Pipelines | Serverless, scalable pipeline execution | GCP Console |
| **Experiment Tracking** | MLflow (Cloud Run) | Model versioning & metrics tracking | https://mlflow-tracking-server-1076639696283.europe-west2.run.app |
| **Data Storage** | Google Cloud Storage | Raw data & artifact storage | `gs://book-sales-deployment-artifacts` |
| **Container Registry** | Google Artifact Registry | Docker image management | `europe-west2-docker.pkg.dev/upheld-apricot-468313-e0/zenml-book-sales-artifacts` |

### Framework Responsibilities

**ZenML:**
- Pipeline step orchestration and caching decisions
- Artifact lineage and data provenance tracking
- Stack management (orchestrator + storage + registry)
- Local development framework

**Vertex AI Pipelines:**
- Managed infrastructure and auto-scaling
- Container orchestration in cloud
- System-level monitoring and resource allocation
- Integration with GCP services

**MLflow (Remote):**
- Centralized experiment tracking across all environments
- Model registry with versioning (e.g., `arima_book_9780241003008 v18`)
- Performance metrics storage (MAE, RMSE, MAPE)
- Model artifact management and deployment staging

**Google Cloud Storage:**
- Raw CSV data: `gs://book-sales-deployment-artifacts/raw_data/`
- Pipeline artifacts: Intermediate data between steps
- MLflow backend: Model binaries and metadata
- Artifact versioning: Historical model archives

### Production Workflow
```
Local Development → ZenML Server → Vertex AI → MLflow Dashboard
       ↓                ↓              ↓            ↓
   Pipeline Code    Orchestration   Execution    Results
```

## 🎯 Quick Start (If Everything is Already Set Up)

```bash
# 1. Authenticate with ZenML (7-day token)
zenml login https://zenml-server-1076639696283.europe-west2.run.app

# 2. Set the Vertex AI stack
zenml stack set vertex_stack

# 3. Verify data exists in GCS
gsutil ls gs://book-sales-deployment-artifacts/raw_data/

# 4. Run the pipeline
python pipelines/zenml_pipeline.py
```

## 📋 Prerequisites

- Google Cloud CLI installed and authenticated (`gcloud auth login`)
- Docker Desktop running
- Python 3.10+ with virtual environment activated
- ZenML installed: `pip install zenml[server]`

## 🔧 Complete Setup (From Scratch)

### 1. Google Cloud Project Setup

**Set your project:**
```bash
gcloud config set project upheld-apricot-468313-e0
```

**Enable required APIs:**
```bash
gcloud services enable cloudbuild.googleapis.com --project=upheld-apricot-468313-e0
gcloud services enable aiplatform.googleapis.com --project=upheld-apricot-468313-e0
gcloud services enable artifactregistry.googleapis.com --project=upheld-apricot-468313-e0
gcloud services enable storage.googleapis.com --project=upheld-apricot-468313-e0
```

### 2. Service Account Creation & Permissions

**Create service account:**
```bash
gcloud iam service-accounts create zenml-pipeline-runner \
    --display-name="ZenML Pipeline Runner" \
    --description="Service account for ZenML pipelines on Vertex AI"
```

**Grant service account permissions:**
```bash
# Core Vertex AI permissions
gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="serviceAccount:zenml-pipeline-runner@upheld-apricot-468313-e0.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Storage permissions  
gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="serviceAccount:zenml-pipeline-runner@upheld-apricot-468313-e0.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Service Account User (for Vertex AI to impersonate)
gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="serviceAccount:zenml-pipeline-runner@upheld-apricot-468313-e0.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
```

### 3. Container Registry Setup

**Create Artifact Registry repository:**
```bash
gcloud artifacts repositories create zenml-book-sales-artifacts \
    --repository-format=docker \
    --location=europe-west2 \
    --description="ZenML pipeline container images"
```

**Configure Docker authentication:**
```bash
gcloud auth configure-docker europe-west2-docker.pkg.dev
```

### 4. GCS Bucket & Data Setup

**Create GCS bucket:**
```bash
gsutil mb -p upheld-apricot-468313-e0 -c STANDARD -l europe-west2 gs://book-sales-deployment-artifacts
```

**Upload raw data to GCS:**
```bash
# Upload your raw data files to the bucket
gsutil cp data/raw/ISBN_data.csv gs://book-sales-deployment-artifacts/raw_data/
gsutil cp data/raw/UK_weekly_data.csv gs://book-sales-deployment-artifacts/raw_data/
```

**Verify data upload:**
```bash
gsutil ls gs://book-sales-deployment-artifacts/raw_data/
```

### 5. MLflow Remote Server Setup

**Deploy MLflow tracking server:**
```bash
gcloud run deploy mlflow-tracking-server \
    --image docker.io/python:3.10-slim \
    --platform managed \
    --region europe-west2 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --port 5000 \
    --command "/bin/bash" \
    --args "-c","pip install mlflow[extras] google-cloud-storage && mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///tmp/mlflow.db --default-artifact-root gs://book-sales-deployment-artifacts/mlflow --serve-artifacts"
```

### 6. ZenML Server Authentication

**Generate Long-Lived API Token:**
```bash
# Get temporary JWT token from web interface first, then:
curl -X 'GET' \
  'https://zenml-server-1076639696283.europe-west2.run.app/api/v1/api_token?token_type=generic&expires_in=604800' \
  -H 'accept: application/json' \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE"

# Use the returned token:
zenml login https://zenml-server-1076639696283.europe-west2.run.app --api-key
# Paste the 7-day token when prompted

# Optional: Save as environment variable
export ZENML_API_KEY="your-7-day-token-here"
echo 'export ZENML_API_KEY="your-token"' >> ~/.zshrc
```

**Add Vertex AI Service Agent permissions:**
```bash
# Add required IAM roles for Vertex AI Service Agent
gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="serviceAccount:service-1076639696283@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="serviceAccount:service-1076639696283@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="serviceAccount:service-1076639696283@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" \
    --role="roles/ml.admin"

# Add Cloud Build permissions for user
gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="user:zarifshafiei@gmail.com" \
    --role="roles/cloudbuild.builds.editor"
```

### 7. ZenML Stack Components

**Create stack components (if they don't exist):**
```bash
# Create GCS artifact store
zenml artifact-store register gcs_store \
    --flavor=gcp \
    --path=gs://book-sales-deployment-artifacts

# Create Vertex AI orchestrator  
zenml orchestrator register vertex_orchestrator \
    --flavor=vertex \
    --project=upheld-apricot-468313-e0 \
    --location=europe-west2 \
    --service_account=zenml-pipeline-runner@upheld-apricot-468313-e0.iam.gserviceaccount.com

# Create container registry
zenml container-registry register gcp_registry \
    --flavor=gcp \
    --uri=europe-west2-docker.pkg.dev/upheld-apricot-468313-e0/zenml-book-sales-artifacts

# Create and set stack
zenml stack register vertex_stack \
    -o vertex_orchestrator \
    -a gcs_store \
    -c gcp_registry

zenml stack set vertex_stack
```

## ✅ Verification Commands

**Check all systems are working:**
```bash
# Check ZenML connection
zenml status

# Check ZenML stack configuration  
zenml stack describe vertex_stack

# Verify GCS data exists  
gsutil ls gs://book-sales-deployment-artifacts/raw_data/

# Test GCS bucket access
gsutil ls gs://book-sales-deployment-artifacts/

# Check container registry
gcloud artifacts repositories list --location=europe-west2

# Test MLflow server accessibility
curl -I https://mlflow-tracking-server-1076639696283.europe-west2.run.app

# Verify Google Cloud APIs are enabled
gcloud services list --enabled --filter="name:aiplatform.googleapis.com OR name:artifactregistry.googleapis.com OR name:cloudbuild.googleapis.com OR name:storage.googleapis.com"

# Check recent container images
gcloud artifacts docker images list europe-west2-docker.pkg.dev/upheld-apricot-468313-e0/zenml-book-sales-artifacts --limit=3
```

## 🐛 Common Issues & Fixes

### Issue: Pip Dependency Conflicts
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
zenml 0.84.2 requires click<8.1.8,>=8.0.1, but you have click 8.1.8 which is incompatible.
```

**Fix:** Update Docker settings in `pipelines/zenml_pipeline.py`:
```python
docker_settings = DockerSettings(
    requirements=[
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "gcsfs>=2024.2.0",
        "google-cloud-storage>=2.10.0",
        "gdown>=5.2.0",
        "openpyxl>=3.1.2",
        "pmdarima>=2.0.4",
        "optuna>=3.0.0",
        "mlflow>=2.3.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "click<8.1.8"  # Add this line
    ],
    parent_image="zenmldocker/zenml:0.84.2-py3.10"
)
```

### Issue: Authentication Expiry
**Fix:** Re-generate 7-day token using the API method above.

### Issue: Container Build Failures
**Fix:** Check Cloud Build permissions and ensure the container registry is accessible.

## 📊 Pipeline Configuration

### Development Mode (Current)
```python
config = get_arima_config(
    environment='development',
    n_trials=3,          # Fast iteration
    force_retrain=False  # Enable smart retraining
)
```

### Production Mode
```python
config = get_arima_config(
    environment='production',
    n_trials=100,        # Thorough optimization
    force_retrain=False  # Smart retraining enabled
)
```

## 🔗 Important URLs

- **ZenML Dashboard**: https://zenml-server-1076639696283.europe-west2.run.app
- **MLflow Tracking**: https://mlflow-tracking-server-1076639696283.europe-west2.run.app
- **GCS Bucket**: https://console.cloud.google.com/storage/browser/book-sales-deployment-artifacts
- **Vertex AI Pipelines**: https://console.cloud.google.com/vertex-ai/locations/europe-west2/pipelines
- **Container Registry**: https://console.cloud.google.com/artifacts/docker/upheld-apricot-468313-e0/europe-west2/zenml-book-sales-artifacts

## 🚀 Running Different Pipeline Modes

### Local Development (Fast)
```bash
zenml stack set default
python pipelines/zenml_pipeline.py
```

### Vertex AI Production
```bash
zenml stack set vertex_stack  
python pipelines/zenml_pipeline.py
```

### Custom Configuration
```python
# Edit pipelines/zenml_pipeline.py, modify this section:
config = get_arima_config(
    environment='development',  # or 'testing', 'production'  
    n_trials=5,                # Number of optimization trials
    force_retrain=True         # Set to False for smart retraining
)
```

## 📈 Monitoring & Results

- **Check runs**: Visit MLflow dashboard for experiment tracking
- **View logs**: Check ZenML dashboard for pipeline execution logs  
- **Model artifacts**: Stored in GCS bucket under `/mlflow/artifacts/`
- **Performance metrics**: MAE, RMSE, MAPE tracked per book in MLflow

## 🔄 Regular Maintenance

1. **Refresh ZenML authentication** every 7 days
2. **Monitor MLflow server costs** (Cloud Run charges)
3. **Clean up old container images** in Artifact Registry
4. **Review model performance** in MLflow dashboard

---

*Last updated: August 2025 - After successful MLOps pipeline deployment with remote MLflow integration*