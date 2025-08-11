#!/bin/bash
set -e

# ZenML Server Deployment Script for Google Cloud Run
echo "🚀 Deploying ZenML Server to Google Cloud Run..."

PROJECT_ID="upheld-apricot-468313-e0"
SERVICE_NAME="zenml-server"
REGION="europe-west2"  # Same region as your other resources
IMAGE="gcr.io/zenml-io/zenml-server:latest"

echo "📋 Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Service: $SERVICE_NAME" 
echo "  Region: $REGION"
echo "  Image: $IMAGE"

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID

# Create a secret for ZenML server credentials
echo "🔐 Setting up authentication secret..."
ZENML_SECRET_KEY=$(openssl rand -hex 32)
echo -n "$ZENML_SECRET_KEY" | gcloud secrets create zenml-secret-key --data-file=- --project=$PROJECT_ID 2>/dev/null || echo "Secret already exists"

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image=$IMAGE \
  --platform=managed \
  --region=$REGION \
  --project=$PROJECT_ID \
  --allow-unauthenticated \
  --port=8080 \
  --memory=2Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --set-env-vars="ZENML_STORE_TYPE=sql,ZENML_STORE_URL=sqlite:///zenml.db" \
  --set-secrets="/secrets/secret-key=zenml-secret-key:latest"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform=managed --region=$REGION --project=$PROJECT_ID --format="value(status.url)")

echo ""
echo "✅ ZenML Server deployed successfully!"
echo "🌐 Server URL: $SERVICE_URL"
echo ""
echo "📝 Next steps:"
echo "1. Connect to your ZenML server:"
echo "   zenml connect --url $SERVICE_URL"
echo ""
echo "2. Create a default user (first login):"
echo "   zenml login --username admin --password your-secure-password"
echo ""
echo "3. Test the connection:"
echo "   zenml stack list"
echo ""