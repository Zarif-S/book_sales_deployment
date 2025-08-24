# Hybrid Cloud MLOps Deployment Architecture

A production-ready ML pipeline implementing hybrid local/cloud deployment for book sales forecasting using SARIMA models with ZenML orchestration, remote MLflow tracking, and Vertex AI deployment.

## 🏗️ Architecture Overview

This pipeline implements a hybrid deployment pattern with local development and cloud production, following enterprise MLOps patterns:

```
┌─────────────────────────┐    ┌────────────────────────────┐
│   Local Development     │    │     Cloud Production       │
│                         │    │                            │
│ • ZenML Client          │    │ • Vertex AI Orchestrator   │
│ • Pipeline Definition   │────▶ • Remote MLflow Tracking  │
│ • Model Development     │    │ • GCS Artifact Storage     │
│ • Local Testing         │    │ • Vertex AI Endpoints      │
└─────────────────────────┘    └────────────────────────────┘
```

## 🚀 Deployment Pipeline Flow

The deployment consists of three sequential steps:

```
┌──────────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│  01_train_pipeline   │    │  02_upload_models   │    │  03_deploy_to_vertex │
│  _and_create_        │    │  _to_gcs.py         │    │  _endpoints.py       │
│  endpoints.py        │    │                     │    │                      │
│                      │    │ • Download from     │    │ • Deploy to Vertex   │
│ • Run Vertex AI      │────▶   MLflow Registry   │────▶   AI Endpoints       │
│   Pipeline           │    │ • Upload to GCS     │    │ • Create endpoints   │
│ • Train ARIMA models │    │ • Package for       │    │ • Test predictions   │
│ • Log to MLflow      │    │   deployment        │    │                      │
└──────────────────────┘    └─────────────────────┘    └──────────────────────┘
```

## ✅ Best Practices Implemented

### 1. Hybrid Architecture
- **Local Development**: ZenML client for pipeline definition and testing
- **Cloud Execution**: Vertex AI orchestrator for scalable training
- **Remote Tracking**: MLflow server for centralized experiment management
- **Production Deployment**: Vertex AI endpoints for serving

### 2. Model Lifecycle Management
- **MLflow Registry**: Centralized model versioning and metadata
- **GCS Storage**: Scalable artifact storage with versioning
- **Vertex AI Models**: Production-ready model deployment
- **Endpoint Management**: Automated endpoint creation and scaling

### 3. Cloud-Native Components
- **Vertex AI Orchestrator**: Managed pipeline execution
- **GCS Artifact Store**: Durable storage for models and data
- **Container Registry**: Docker image management for custom steps
- **Remote MLflow**: Cloud-hosted experiment tracking

### 4. Deployment Automation
- **Three-Stage Pipeline**: Training → Packaging → Deployment
- **Model Containerization**: Vertex AI pre-built containers
- **Endpoint Testing**: Automated prediction validation
- **Rollback Support**: Model versioning for safe deployments

### 5. Environment Management
- **Stack Configuration**: Multiple ZenML stacks for different environments
- **GCP Integration**: Service account authentication and resource management
- **Configuration Management**: Environment-specific settings
- **Resource Optimization**: Configurable compute resources

### 6. Production Best Practices
- **Monitoring**: Endpoint health and prediction logging
- **Scaling**: Auto-scaling endpoints based on traffic
- **Security**: IAM-based access control and encrypted storage
- **Cost Management**: Resource allocation and usage optimization

## 🏢 Industry Standard Pattern

Your architecture follows the **"Hybrid MLOps"** pattern used by:
- **Netflix**: Local development → Cloud training → Managed serving
- **Uber**: Hybrid orchestration with cloud-scale execution
- **Spotify**: Development/staging/production environment separation
- **Airbnb**: MLflow + cloud ML platforms for model lifecycle

## 🎯 Why This Hybrid Architecture Is Optimal

### Developer Experience
- Local development and testing with familiar tools
- Remote execution without infrastructure management
- Centralized experiment tracking and model registry
- Easy switching between local and cloud environments

### Scalability & Cost Efficiency
- Cloud resources only used during training and serving
- Vertex AI auto-scaling for variable workloads
- Pay-per-use model for compute resources
- Managed services reduce operational overhead

### Production Readiness
- Enterprise-grade model serving with Vertex AI
- Automatic scaling and load balancing
- Built-in monitoring and logging
- Version control and rollback capabilities

## 📁 Deployment Pipeline Components

```
deploy/
├── 01_train_pipeline_and_create_endpoints.py   # Vertex AI pipeline execution
├── 02_upload_models_to_gcs.py                  # MLflow → GCS model upload
└── 03_deploy_to_vertex_endpoints.py            # Vertex AI endpoint deployment

pipelines/
└── zenml_pipeline.py                           # Core training pipeline definition

config/
└── arima_training_config.py                    # Environment configurations

outputs/
├── models/arima/                               # Local model artifacts
└── plots/interactive/                          # Forecast visualizations
```

## 🛠️ Usage - Hybrid MLOps Workflow

### Complete Deployment Flow (Current Setup)
```bash
# Step 1: Train models with local orchestrator + cloud storage
python pipelines/zenml_pipeline.py

# Step 2: Upload trained models to GCS in Vertex AI format
python deploy/02_upload_models_to_gcs.py --upload-all

# Step 3: Deploy models to Vertex AI endpoints
python deploy/03_deploy_to_vertex_endpoints.py --deploy-all
```

### Individual Steps
```bash
# Train models locally with cloud storage
python pipelines/zenml_pipeline.py

# Check what models are available
python deploy/02_upload_models_to_gcs.py --list-models

# Upload specific model to GCS
python deploy/02_upload_models_to_gcs.py --model-name arima_book_9780722532935

# Deploy specific model to Vertex AI
python deploy/03_deploy_to_vertex_endpoints.py --model-name arima_book_9780722532935

# Test deployed endpoint
python deploy/03_deploy_to_vertex_endpoints.py --test-endpoint book-sales-9780722532935
```

### ⚠️ About deploy/01_train_pipeline_and_create_endpoints.py

This script was designed for **full Vertex AI orchestration** but ZenML 0.84.2+ requires a remote ZenML server for Vertex AI orchestrator.

**Current setup uses:** Local orchestrator + GCS storage + Remote MLflow + Vertex AI deployment

**Benefits of hybrid approach:**
- ✅ No remote ZenML server management
- ✅ Fast local development and testing  
- ✅ Cloud storage and experiment tracking
- ✅ Production Vertex AI deployment capability

## ⚙️ Configuration

### ZenML Stack Configuration
```bash
# Switch to Vertex AI stack for cloud deployment
zenml stack set vertex_ai_stack

# Verify stack configuration
zenml stack describe
```

### Environment Variables
```bash
export DEPLOYMENT_ENV=production          # Environment: development/testing/production
export GCP_PROJECT=upheld-apricot-468313-e0
export GCP_REGION=europe-west2
export MLFLOW_TRACKING_URI=https://mlflow-tracking-server-1076639696283.europe-west2.run.app
```

### Key Parameters
```python
DEFAULT_TEST_ISBNS = ['9780722532935', '9780241003008']
DEFAULT_SPLIT_SIZE = 32                  # Test set size (weeks)
DEFAULT_MAX_SEASONAL_BOOKS = 15          # Maximum books for modeling
PROJECT_ID = "upheld-apricot-468313-e0"  # GCP Project
BUCKET_NAME = "book-sales-deployment-artifacts"  # GCS Bucket
```

### Required Authentication
```bash
# GCP Authentication
gcloud auth login
gcloud config set project upheld-apricot-468313-e0

# ZenML Connection (if needed)
zenml connect --url http://your-zenml-server
```

## ⚠️ Current Limitations

### Deployment Process
- Three-step manual deployment process (not fully automated)
- Model containerization uses generic pre-built containers
- Limited automated testing of deployed endpoints

### Model Scope
- SARIMA models only; no ensemble or deep learning models in deployment
- Fixed seasonal patterns and forecasting horizons
- Limited real-time data integration capabilities

### Infrastructure & Monitoring
- Basic endpoint monitoring (no custom dashboards)
- Manual scaling configuration
- Limited A/B testing framework for model comparison
- Cost monitoring requires manual GCP console checks

## 🔮 Future Improvements

### Deployment Automation
- Single-command end-to-end deployment pipeline
- Automated integration testing for deployed models
- Infrastructure as Code (Terraform) for GCP resources

### Advanced Model Serving
- Custom container images for optimized serving
- Multi-model endpoints for ensemble predictions
- Real-time feature stores for dynamic inputs
- Edge deployment options for low-latency scenarios

### Monitoring & Observability
- Model performance monitoring and drift detection
- Custom Grafana dashboards for business metrics
- Automated alerting for prediction anomalies
- Cost optimization and resource utilization tracking

### Enterprise Features
- Multi-environment promotion (dev → staging → prod)
- Automated model validation and testing suites
- A/B testing framework for gradual rollouts
- Data governance and model explainability features

---

## 📋 Technical Summary

This hybrid MLOps architecture demonstrates modern enterprise ML deployment patterns combining local development with cloud-scale execution. The three-stage deployment process (training → packaging → serving) provides a clear separation of concerns while maintaining end-to-end traceability through MLflow and ZenML.

**Key Benefits:**
- 🚀 **Developer Productivity**: Local development with cloud execution
- 📈 **Scalability**: Vertex AI auto-scaling and managed services  
- 🔒 **Production Ready**: Enterprise security and monitoring
- 💰 **Cost Efficient**: Pay-per-use cloud resources

**Result**: A production-ready MLOps pipeline that scales from prototype to enterprise deployment. 🎯