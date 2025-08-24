# MLOps Deployment Guide - Book Sales ARIMA Pipeline

This guide contains step-by-step instructions for deploying the hybrid MLOps pipeline with local ZenML orchestration, cloud storage, and remote tracking.

## 🏗️ Architecture Overview

### Production MLOps Stack (As Implemented)

| Component | Technology | Purpose | URL/Location |
|-----------|------------|---------|--------------|
| **Pipeline Orchestration** | ZenML (Local) | Local pipeline execution & artifact tracking | http://127.0.0.1:8237 |
| **Artifact Storage** | Google Cloud Storage | Raw data & pipeline artifacts | `gs://book-sales-deployment-artifacts` |
| **Experiment Tracking** | MLflow (Remote) | Model versioning & metrics tracking | https://mlflow-tracking-server-1076639696283.europe-west2.run.app |
| **Container Registry** | Google Artifact Registry | Docker image management | `europe-west2-docker.pkg.dev/upheld-apricot-468313-e0/zenml-book-sales-artifacts` |
| **Model Deployment** | Vertex AI Endpoints | Production model serving | GCP Console |

### Framework Responsibilities (Actual)

**ZenML (Local):**
- Pipeline orchestration on local machine
- Artifact lineage and data provenance tracking
- Local development with cloud integration
- Dashboard at http://127.0.0.1:8237

**Google Cloud Storage:**
- Raw CSV data: `gs://book-sales-deployment-artifacts/raw_data/`
- Pipeline artifacts: Intermediate data between steps
- MLflow backend: Model binaries and metadata

**MLflow (Remote):**
- Centralized experiment tracking
- Model registry with versioning (e.g., `arima_book_9780241003008`)
- Performance metrics storage (MAE, RMSE, MAPE)
- Accessible at: https://mlflow-tracking-server-1076639696283.europe-west2.run.app

**Vertex AI (Deployment Only):**
- Model endpoint creation and management
- Production model serving infrastructure

### Actual Workflow
```
Local ZenML → Local Orchestrator → GCS Storage → Remote MLflow → Vertex AI Endpoints
     ↓              ↓               ↓              ↓                    ↓
 Local Dev      Fast Execution   Cloud Storage   Remote Tracking   Cloud Serving
```

## 🚀 Quick Start (Current Working Setup)

```bash
# 1. Ensure local ZenML server is running
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
poetry run zenml login --local

# 2. Set the local hybrid stack
poetry run zenml stack set local_vertex_test

# 3. Verify current configuration
poetry run zenml status

# 4. Run the pipeline
poetry run python pipelines/zenml_pipeline.py
```

## 📋 Prerequisites (Simplified)

- Google Cloud CLI authenticated (`gcloud auth login`)
- Python 3.10+ with Poetry environment
- ZenML installed locally
- Access to GCS bucket and MLflow server

## 🔧 Stack Configuration (As Working)

### Current Active Stack: `local_vertex_test`

```bash
# View current stack
poetry run zenml stack describe local_vertex_test

# Components:
# - Orchestrator: default (local)
# - Artifact Store: gcs_store (cloud)
# - Container Registry: gcr_registry (cloud)
```

### Stack Components Setup

```bash
# GCS artifact store (already configured)
poetry run zenml artifact-store describe gcs_store

# Container registry (already configured)  
poetry run zenml container-registry describe gcr_registry

# Local orchestrator (default, no setup needed)
poetry run zenml orchestrator describe default
```

## ✅ Verification Commands (Current Setup)

```bash
# Check ZenML local connection
poetry run zenml status

# Check current stack
poetry run zenml stack describe local_vertex_test

# Verify GCS access
gsutil ls gs://book-sales-deployment-artifacts/raw_data/

# Test MLflow server
curl -I https://mlflow-tracking-server-1076639696283.europe-west2.run.app

# Check if local ZenML dashboard is accessible
curl -I http://127.0.0.1:8237
```

## 🎯 Pipeline Execution

### Running the Pipeline

```bash
# Standard execution (uses current stack automatically)
poetry run python pipelines/zenml_pipeline.py

# With custom configuration
poetry run python -c "
from pipelines.zenml_pipeline import book_sales_arima_modeling_pipeline
from config.arima_training_config import get_arima_config, DEFAULT_TEST_ISBNS, DEFAULT_SPLIT_SIZE
from pathlib import Path

config = get_arima_config(environment='development', n_trials=3, force_retrain=True)
output_dir = str(Path.cwd() / 'data' / 'processed')

pipeline_run = book_sales_arima_modeling_pipeline(
    output_dir=output_dir,
    selected_isbns=DEFAULT_TEST_ISBNS,
    column_name='Volume',
    split_size=DEFAULT_SPLIT_SIZE,
    use_seasonality_filter=False,
    max_seasonal_books=15,
    train_arima=True,
    n_trials=3,
    config=config,
    pipeline_timestamp='local_run',
    use_local_mlflow=False
)
"
```

## 🚀 Quick Start - Hybrid Deployment Workflow

**Current Workflow:** Local orchestration with cloud storage + Vertex AI deployment

### Complete End-to-End Deployment (Recommended)
```bash
# Step 1: Train models with hybrid stack (local orchestrator + cloud storage)
python pipelines/zenml_pipeline.py

# Step 2: Upload trained models to GCS in Vertex AI format
python deploy/02_upload_models_to_gcs.py --upload-all

# Step 3: Deploy models to Vertex AI endpoints
python deploy/03_deploy_to_vertex_endpoints.py --deploy-all

# Step 4: Test deployed endpoints
python deploy/03_deploy_to_vertex_endpoints.py --test-endpoint book-sales-9780722532935
cd streamlit_app && streamlit run app.py
```

### Deploy Existing Models (Upload → Deploy)
```bash
# If you already have trained models in MLflow:

# Step 1: Upload existing models to GCS
python deploy/02_upload_models_to_gcs.py --upload-all

# Step 2: Deploy with containerization
python deploy/03_deploy_to_vertex_endpoints.py --deploy-all

# Step 3: Test
python deploy/03_deploy_to_vertex_endpoints.py --test-endpoint book-sales-9780722532935
cd streamlit_app && streamlit run app.py
```

### ⚠️ Note: deploy/01_train_pipeline_and_create_endpoints.py

**This script is designed for full Vertex AI orchestration** and requires a remote ZenML server (ZenML 0.84.2+ requirement). 

**For your current hybrid setup (local orchestrator + cloud storage), use the workflow above instead.**

**Alternative Options:**
- **Option A:** Deploy a remote ZenML server to GCP to use this script
- **Option B:** Modify the script to work with local orchestrator (remove Vertex AI checks)
- **Option C:** Use the recommended 3-step workflow above (simpler and works perfectly)

### File Structure (Updated)
```
deploy/
├── 01_train_pipeline_and_create_endpoints.py  # Train models + create basic endpoints
├── 02_upload_models_to_gcs.py                 # Upload MLflow models to GCS
├── 03_deploy_to_vertex_endpoints.py           # Deploy with containerization
├── container/
│   └── predictor.py                            # Custom prediction container (alternative)
└── DEPLOYMENT_README.md                       # This guide
```

### Script Purpose Summary
- **01**: Trains models via pipeline + creates empty endpoints (optional if models exist)
- **02**: Uploads MLflow models to GCS in Vertex AI format (required)
- **03**: Deploys models using pre-built containers (required for predictions)

## 📊 Model Deployment

### Deploy Models to Vertex AI Endpoints

```bash
# List available models in MLflow
poetry run python deploy/01_train_pipeline_and_create_endpoints.py --list-models

# Deploy all models to Vertex AI endpoints (creates endpoints, containerization in progress)
poetry run python deploy/01_train_pipeline_and_create_endpoints.py --deploy-all
```

### ✅ Model Containerization with Vertex AI Pre-built Containers

**Current Status**: Complete containerization solution using Vertex AI's pre-built containers:
- ✅ Creates Vertex AI endpoints
- ✅ Discovers models from MLflow registry  
- ✅ Uploads models to Google Cloud Storage
- ✅ Deploys models using Vertex AI pre-built scikit-learn containers
- ✅ Full end-to-end model serving capability

**Complete Deployment Process**:

#### Step 1: Upload Models to GCS
```bash
# Upload all MLflow models to GCS in Vertex AI-compatible format
python 02_upload_models_to_gcs.py --upload-all

# List available models in MLflow
python 02_upload_models_to_gcs.py --list-models

# Check uploaded models in GCS
python 02_upload_models_to_gcs.py --list-uploaded
```

**What this does:**
- Downloads models from MLflow registry
- Converts to joblib format (compatible with Vertex AI)
- Creates prediction scripts
- Uploads to `gs://book-sales-deployment-artifacts/models/`

#### Step 2: Deploy to Vertex AI Endpoints
```bash
# Deploy all models to Vertex AI endpoints with full serving
python 03_deploy_to_vertex_endpoints.py --deploy-all

# Deploy specific model
python 03_deploy_to_vertex_endpoints.py --model-name arima_book_9780722532935

# Test deployed endpoint
python 03_deploy_to_vertex_endpoints.py --test-endpoint book-sales-9780722532935
```

**What this does:**
- Uses Vertex AI's pre-built scikit-learn container (`sklearn-cpu.1-0:latest`)
- Creates Vertex AI Model resources from GCS artifacts
- Deploys models to endpoints with auto-scaling
- Configures traffic routing and health checks

#### Step 3: Test via Streamlit App
```bash
cd streamlit_app
streamlit run app.py
# Navigate to "System Status" to check endpoint health
# Use "Forecast" page for real predictions
```

**Technical Implementation**:

**GCS Model Structure:**
```
gs://book-sales-deployment-artifacts/models/
├── arima_book_9780722532935/latest/
│   ├── model.joblib          # Vertex AI compatible model
│   ├── model.pkl            # Backup pickle format
│   ├── metadata.json        # Model metadata
│   └── predictor.py         # Prediction script
└── arima_book_9780241003008/latest/
    └── ... (same structure)
```

**Prediction Format:**
Your endpoints return structured predictions:
```json
{
  "forecast": [245.2, 267.8, 234.1, 289.3],
  "steps": 4,
  "model_name": "arima_book_9780722532935", 
  "isbn": "9780722532935",
  "confidence_intervals": {
    "lower": [230.1, 250.3, 220.5, 270.8],
    "upper": [260.3, 285.3, 247.7, 307.8],
    "confidence_level": 0.95
  }
}
```

**Vertex AI Container Configuration:**
- **Container**: `us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest`
- **Machine Type**: `n1-standard-2` (configurable)
- **Scaling**: 1-2 replicas with auto-scaling
- **Traffic**: 100% to deployed model

**Key Benefits:**
- **No Custom Dockerfiles**: Uses Google's optimized containers
- **Automatic Scaling**: Handles traffic spikes automatically  
- **Cost Effective**: Pay only for usage
- **Production Ready**: Full monitoring and logging
- **Framework Support**: Native statsmodels/MLflow compatibility

## 🔗 Important URLs (Actual)

- **ZenML Dashboard (Local)**: http://127.0.0.1:8237
- **MLflow Tracking**: https://mlflow-tracking-server-1076639696283.europe-west2.run.app
- **GCS Bucket**: https://console.cloud.google.com/storage/browser/book-sales-deployment-artifacts
- **Container Registry**: https://console.cloud.google.com/artifacts/docker/upheld-apricot-468313-e0/europe-west2/zenml-book-sales-artifacts

## 🚀 Architecture Benefits

### Why This Setup Works Well

1. **Cost Effective**: Local orchestration reduces cloud compute costs
2. **Fast Development**: No container building for local steps
3. **Cloud Integration**: Still uses cloud storage and tracking
4. **Deployment Ready**: Models can be deployed to Vertex AI endpoints
5. **Scalable**: Can be upgraded to full cloud orchestration later

### MLOps Best Practices Maintained

- ✅ **Artifact Storage**: Cloud-based with versioning
- ✅ **Experiment Tracking**: Centralized in remote MLflow
- ✅ **Model Registry**: Proper model versioning and metadata
- ✅ **Deployment Pipeline**: Automated endpoint creation
- ✅ **Infrastructure as Code**: Stack components defined programmatically

## 🔄 Monitoring & Results

- **Pipeline Runs**: ZenML dashboard at http://127.0.0.1:8237
- **Experiment Tracking**: MLflow dashboard for metrics and models
- **Model Artifacts**: Stored in GCS bucket under MLflow paths
- **Performance Metrics**: MAE, RMSE, MAPE tracked per book

## 🎯 Portfolio Value

This architecture demonstrates:
- **Hybrid cloud approach**: Local development + cloud services
- **Cost optimization**: Efficient resource usage
- **Practical MLOps**: Real-world constraints and solutions
- **Flexibility**: Easy to scale up or modify
- **Problem-solving**: Working around authentication challenges

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

## 📊 Pipeline Configuration Options

### 1. Default Configuration (Simplest)
Just run your pipeline as normal, it automatically uses development mode:
```bash
python pipelines/zenml_pipeline.py
```
This uses built-in defaults (3 trials, force_retrain=False for development).

### 2. Quick Custom Configuration
Modify the configuration in your pipeline code:
```python
from config.arima_training_config import get_arima_config

# Quick customization for development
config = get_arima_config(
    n_trials=10,           # More trials for better results
    force_retrain=False    # Enable smart retraining
)

# Run pipeline with custom config
results = book_sales_arima_modeling_pipeline(
    output_dir=output_dir,
    selected_isbns=['9780722532935', '9780241003008'],
    config=config
)
```

### 3. Environment-Specific Configs
Use predefined environment configurations:
```python
from config.arima_training_config import get_development_config, get_production_config

# Development mode (fast iteration)
dev_config = get_development_config(n_trials=5, force_retrain=False)

# Production mode (high quality)
prod_config = get_production_config()

# Testing mode (balanced)
test_config = get_arima_config(environment='testing')
```

### 4. Environment Variables (DevOps-Friendly)
Set environment variables for deployment:
```bash
export DEPLOYMENT_ENV=production
export ARIMA_N_TRIALS=50
export ARIMA_FORCE_RETRAIN=false
export ARIMA_MAX_MODEL_AGE_DAYS=14

python pipelines/zenml_pipeline.py
```

### 5. Configuration File (Advanced)
Save/load configurations to/from JSON files:
```python
config = get_production_config()
config.save_to_json_file('my_production_config.json')

# Later, load it back
config = ARIMATrainingConfig.from_json_file('my_production_config.json')
```

### 🎯 Recommended Usage Patterns

**For Development:**
```python
config = get_development_config(
    n_trials=3,           # Fast for testing
    force_retrain=False   # Enable smart retraining
)
```

**For Production:**
```bash
# Set via environment variables
export DEPLOYMENT_ENV=production
export ARIMA_FORCE_RETRAIN=false
export ARIMA_N_TRIALS=100
```

**For Different Scenarios:**
```python
# First run (train all models)
config = get_arima_config(force_retrain=True)

# Second run (reuse models if possible)
config = get_arima_config(force_retrain=False)

# High-quality production run
config = get_production_config(n_trials=200)
```

### 📊 Key Configuration Parameters
- **n_trials**: Number of Optuna trials per book (3-200)
- **force_retrain**: True = always train, False = smart retraining
- **environment**: 'development', 'testing', 'production'
- **max_model_age_days**: Retrain if model older than X days
- **performance_threshold**: Retrain if performance degrades by X%

### 🚀 Vertex AI Deployment Configuration

For Vertex AI deployment, adjust these settings for optimal performance and cost efficiency:

**Production Environment Settings:**
```python
# For Vertex AI deployment
config = get_arima_config(
    environment='production',
    n_trials=50,              # Balanced quality vs speed
    force_retrain=False,      # Enable smart retraining (important!)
    max_model_age_days=30,    # Monthly refresh cycle
    performance_threshold=0.10  # 10% performance degradation trigger
)
```

**Environment Variables for Vertex AI:**
Set these in your Vertex AI pipeline configuration:
```bash
# Core settings
DEPLOYMENT_ENV=production
ARIMA_FORCE_RETRAIN=false    # Critical for efficiency
ARIMA_N_TRIALS=50           # Adjust based on budget

# Smart retraining triggers
ARIMA_MAX_MODEL_AGE_DAYS=30
ARIMA_PERFORMANCE_THRESHOLD=0.10

# Quality gates
ARIMA_MIN_RMSE=80.0
ARIMA_MAX_MAPE=30.0
```

**Key Vertex AI Optimizations:**
The optimized pipeline is already Vertex AI ready with these features:
- ✅ **Consolidated Artifacts**: Train/test data as single DataFrames
- ✅ **Model Registry Integration**: MLflow models registered for deployment
- ✅ **Smart Retraining**: Reduces compute costs by 60-80%
- ✅ **In-Memory Storage**: Production mode uses memory-based Optuna
- ✅ **Robust Error Handling**: Graceful fallbacks for cloud environments

**Cost-Optimized Settings:**
```python
# For cost efficiency on Vertex AI
vertex_ai_config = get_arima_config(
    environment='production',
    n_trials=25,              # Lower trials = lower cost
    force_retrain=False,      # Smart retraining saves money
    max_model_age_days=14,    # More frequent checks
    performance_threshold=0.05 # Stricter performance monitoring
)
```

**High-Quality Settings:**
```python
# For maximum model quality
high_quality_config = get_arima_config(
    environment='production',
    n_trials=100,             # More optimization
    force_retrain=False,      # Still use smart retraining
    max_model_age_days=7,     # Weekly refresh
    performance_threshold=0.03 # Very strict performance
)
```

**📊 Expected Vertex AI Benefits:**
| Setting             | First Run     | Subsequent Runs | Savings |
|---------------------|---------------|-----------------|---------|
| force_retrain=True  | 100% training | 100% training   | 0%      |
| force_retrain=False | 100% training | 20-40% training | 60-80%  |

**⚙️ Vertex AI Deployment Steps:**
1. Set Environment Variables in your Vertex AI pipeline config
2. Use Production Configuration - the pipeline auto-detects DEPLOYMENT_ENV=production
3. Enable Smart Retraining with ARIMA_FORCE_RETRAIN=false
4. Monitor Results - the pipeline logs reuse statistics

**🎯 Recommended Vertex AI Config:**
```python
# Optimal balance for Vertex AI
vertex_config = get_arima_config(
    environment='production',
    n_trials=50,
    force_retrain=False,      # Key for cost savings
    max_model_age_days=21,    # 3-week refresh cycle
    performance_threshold=0.08
)
```

This configuration will:
- Reduce compute costs by 60-80% after initial run
- Maintain model quality with performance monitoring
- Use production-grade settings (in-memory storage, robust error handling)
- Scale efficiently with your book catalog

The configuration system automatically picks sensible defaults for each environment, so you only need to override what you want to change!

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

## 🚀 Model Endpoint Deployment

### Quick Start - Deploy Models to Vertex AI Endpoints

Once your models are trained and registered in MLflow, you can deploy them to Vertex AI endpoints for real-time inference:

```bash
# Navigate to deployment directory
cd deploy/

# List available trained models
python deploy_models.py --list-models

# Deploy all book models to individual endpoints
python deploy_models.py --deploy-all

# Deploy a specific model to a custom endpoint
python deploy_models.py --model-name arima_book_9780722532935 --endpoint-name book-sales-alchemist

# Check current endpoints
python deploy_models.py --list-endpoints

# Use custom project/region (if different from defaults)
python deploy_models.py --list-models --project-id your-project-id --region your-region
```

**Default Configuration:**
- **Project ID**: `upheld-apricot-468313-e0`
- **Region**: `europe-west2`
- **MLflow URI**: `https://mlflow-tracking-server-1076639696283.europe-west2.run.app`

### Prerequisites for Endpoint Deployment

**Required Dependencies:**
```bash
# Dependencies are managed by Poetry - ensure environment is up to date
poetry install

# Or if in existing environment, sync dependencies
poetry install --sync

# For cleaner command usage, activate Poetry shell (recommended):
poetry shell
# Now you can run python deploy_models.py directly without poetry run prefix
```

**Verify you have:**
✅ **Trained models**: Run your pipeline first to create MLflow models
✅ **Authentication**: `gcloud auth login` completed
✅ **Project access**: Vertex AI API enabled in your project
✅ **Dependencies**: Poetry environment with `mlflow` and `google-cloud-aiplatform` (includes `vertexai`) - already in `pyproject.toml`

**Check prerequisites:**
```bash
# Verify authentication
gcloud auth list

# Make sure you're authenticated for application default credentials
gcloud auth application-default login

# Check Vertex AI API is enabled
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"

# Test MLflow connection
curl -I https://mlflow-tracking-server-1076639696283.europe-west2.run.app

# Quick test - list models to verify everything works
python deploy_models.py --list-models
```

**If you see import errors:**
The script will detect missing dependencies. Run `poetry install` to ensure all dependencies from `pyproject.toml` are installed.

**Note**: If you prefer not to use `poetry shell`, you can run commands with `poetry run python deploy_models.py` instead.

### Deployment Process

1. **Model Discovery**: Script reads from MLflow registry
2. **Endpoint Creation**: Creates/reuses Vertex AI endpoints
3. **Basic Setup**: Prepares endpoint for model serving
4. **Status Reporting**: Logs deployment results

### Example Usage Scenarios

**1. Explore available models:**
```bash
python deploy_models.py --list-models
# Output example:
# Found 3 book models:
#   - arima_book_9780722532935 (v2, Production) - ISBN: 9780722532935
#   - arima_book_9780241003008 (v1, Staging) - ISBN: 9780241003008
#   - arima_book_9781234567890 (v1, None) - ISBN: 9781234567890
```

**2. After training pipeline completes:**
```bash
# Quick deployment of all models (creates individual endpoints)
python deploy_models.py --deploy-all
# Output: Creates endpoints like book-sales-9780722532935, book-sales-9780241003008, etc.
```

**3. Deploy specific book model:**
```bash
# Deploy The Alchemist model to custom endpoint
python deploy_models.py --model-name arima_book_9780722532935 --endpoint-name book-sales-alchemist

# Deploy Very Hungry Caterpillar model to auto-named endpoint
python deploy_models.py --model-name arima_book_9780241003008 --endpoint-name book-sales-caterpillar

# Deploy latest version of a model
python deploy_models.py --model-name arima_book_9780722532935 --endpoint-name my-endpoint
```

**4. Check deployment status:**
```bash
python deploy_models.py --list-endpoints
# Shows all endpoints with model counts and IDs
```

### Monitoring & Management

**Check deployment status:**
```bash
# List all current endpoints
python deploy_models.py --list-endpoints

# View in Google Cloud Console
open https://console.cloud.google.com/vertex-ai/locations/europe-west2/endpoints
```

**Endpoint naming convention:**
- Individual book models: `book-sales-{isbn}`
- Custom endpoints: Use `--endpoint-name` parameter

### Current Limitations & Next Steps

**Current Implementation:**
- ✅ Endpoint creation and management
- ✅ Model registry integration
- ✅ Basic deployment workflow
- ✅ Automatic model discovery from MLflow
- ✅ Batch deployment of all models
- ⚠️ **Important**: Creates endpoints but requires additional container setup for full model serving

**What the script does:**
1. ✅ Connects to MLflow registry and discovers book models
2. ✅ Creates or reuses Vertex AI endpoints
3. ✅ Prepares endpoints for model deployment
4. ⚠️ Logs that additional containerization is needed for full serving

**Next steps for production deployment:**
- Package MLflow models in serving containers
- Configure custom prediction containers
- Set up model serving infrastructure

**Production Enhancements (Post-Refactor):**
- Custom prediction containers for MLflow models
- Traffic splitting for A/B testing
- Auto-scaling configuration
- Health checks and monitoring
- Cost optimization settings

### Design Rationale

This endpoint deployment approach uses a **simple, standalone script** rather than integrated ZenML steps. This design choice supports an upcoming large-scale refactor project where:

- The main `pipelines/zenml_pipeline.py` will be broken down into proper step files
- MLOps improvements like parallel processing and A/B testing will be added
- The pipeline architecture will undergo significant restructuring

**Benefits of this approach:**
- **Refactor-friendly**: Won't interfere with pipeline restructuring
- **Standalone**: Can be run independently of ZenML pipeline
- **Simple**: Easy to understand and modify during refactor
- **Extensible**: Can be enhanced with advanced features later

By keeping endpoint deployment separate, we avoid coupling with the current pipeline structure and ensure this functionality can be easily enhanced or integrated once the new architecture is in place.

### Troubleshooting

**Import/Dependency Issues:**
```bash
# If you see "Required packages not installed" error:
poetry install

# Verify dependencies are available:
# Option 1: If Poetry shell is activated
python -c "import mlflow, google.cloud.aiplatform, vertexai; print('Dependencies OK')"

# Option 2: Without activating shell
poetry run python -c "import mlflow, google.cloud.aiplatform, vertexai; print('Dependencies OK')"

# Check Poetry environment status:
poetry show mlflow google-cloud-aiplatform
```

**Authentication Issues:**
```bash
# Re-authenticate if needed
gcloud auth login
gcloud auth application-default login

# Verify authentication status
gcloud auth list
```

**Permission Issues:**
```bash
# Verify Vertex AI permissions
gcloud projects add-iam-policy-binding upheld-apricot-468313-e0 \
    --member="user:$(gcloud config get-value account)" \
    --role="roles/aiplatform.user"

# Test permissions with a simple command
python deploy_models.py --list-endpoints
```

**Model Not Found:**
```bash
# Check models exist in MLflow (most common issue)
python deploy_models.py --list-models

# If no models shown, verify:
# 1. MLflow server is accessible
curl https://mlflow-tracking-server-1076639696283.europe-west2.run.app

# 2. Pipeline has been run and models are registered
# 3. Models follow naming convention: arima_book_*
```

**Script Arguments:**
```bash
# For help with all available options:
python deploy_models.py --help

# Common argument combinations:
python deploy_models.py --model-name MODEL_NAME --endpoint-name ENDPOINT_NAME
python deploy_models.py --deploy-all
python deploy_models.py --list-models --project-id YOUR_PROJECT --region YOUR_REGION
```

---

*Last updated: August 2025 - After successful MLOps pipeline deployment with remote MLflow integration and endpoint deployment capability*


  - ZenML Dashboard: http://127.0.0.1:8237 (pipeline runs, steps, stacks)
  - MLflow UI: http://127.0.0.1:5001 (experiment tracking, models)

zenml-pipeline-runner@upheld-apricot-468313-e0.iam.gserviceaccount.com

The email address you see is the unique identifier for that service account. Think of it as the account's official name within all of Google Cloud.

# Switch to your local stack
zenml stack set default

# Now run your pipeline script as usual
python your_pipeline_script.py

-----

# Switch to your cloud stack (replace with your stack's name)
zenml stack set vertex_stack

# Run the EXACT SAME pipeline script
python your_pipeline_script.py

-- for status:/which stack active and running

zenml stack get

########## When deploying to cloud: ###########

# Activate the cloud stack
zenml stack set vertex_stack

# Run your pipeline script
python your_pipeline_script.py

#### Batch Feeding & Dynamic Data Prep with Smart Retraining Plan https://chatgpt.com/share/68951be4-a4a4-8006-ab5e-c07cbfa9ff08 #####

## useful commands
python3 pipelines/zenml_pipeline.py && python3 scripts/arima_forecast_load_artefacts.py
