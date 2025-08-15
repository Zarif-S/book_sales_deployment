#!/usr/bin/env python3
"""
Deploy Models to Vertex AI Endpoints using Pre-built Containers

This script deploys models from GCS to Vertex AI endpoints using 
Vertex AI's pre-built scikit-learn container which supports statsmodels.

Usage:
    python 03_deploy_to_vertex_endpoints.py --deploy-all
    python 03_deploy_to_vertex_endpoints.py --model-name arima_book_9780722532935
    python 03_deploy_to_vertex_endpoints.py --list-endpoints
"""

import argparse
import logging
import time
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from google.cloud import aiplatform
    import vertexai
    from google.cloud import storage
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.info("Run: poetry install")
    exit(1)


class VertexAIModelDeployer:
    """Deploy models to Vertex AI endpoints using pre-built containers."""
    
    def __init__(self, 
                 project_id: str = "upheld-apricot-468313-e0", 
                 region: str = "europe-west2",
                 bucket_name: str = "book-sales-deployment-artifacts"):
        """Initialize the deployer."""
        self.project_id = project_id
        self.region = region
        self.bucket_name = bucket_name
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        aiplatform.init(project=project_id, location=region)
        
        # Initialize GCS client
        self.gcs_client = storage.Client(project=project_id)
        self.bucket = self.gcs_client.bucket(bucket_name)
        
        logger.info(f"Initialized deployer:")
        logger.info(f"  Project: {project_id}")
        logger.info(f"  Region: {region}")
        logger.info(f"  Bucket: {bucket_name}")
    
    def list_uploaded_models(self) -> List[Dict[str, str]]:
        """List models available in GCS for deployment."""
        try:
            models = []
            prefix = "models/"
            
            # List model directories
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            model_dirs = set()
            
            for blob in blobs:
                parts = blob.name.split("/")
                if len(parts) >= 3 and parts[0] == "models" and parts[1].startswith("arima_book_"):
                    model_name = parts[1]
                    model_version = parts[2]
                    model_dirs.add((model_name, model_version))
            
            for model_name, version in model_dirs:
                # Check if model has required files
                model_path = f"models/{model_name}/{version}"
                required_files = ["model.joblib", "metadata.json"]
                
                has_all_files = True
                for required_file in required_files:
                    blob_path = f"{model_path}/{required_file}"
                    if not self.bucket.blob(blob_path).exists():
                        has_all_files = False
                        break
                
                if has_all_files:
                    models.append({
                        "name": model_name,
                        "version": version,
                        "gcs_uri": f"gs://{self.bucket_name}/{model_path}",
                        "isbn": model_name.replace("arima_book_", "")
                    })
            
            logger.info(f"Found {len(models)} deployable models in GCS")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list uploaded models: {e}")
            return []
    
    def create_or_get_endpoint(self, endpoint_name: str) -> aiplatform.Endpoint:
        """Create a new endpoint or get existing one."""
        try:
            # Try to get existing endpoint
            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
            if endpoints:
                endpoint = endpoints[0]
                logger.info(f"📍 Using existing endpoint: {endpoint_name}")
                return endpoint

            # Create new endpoint
            logger.info(f"🆕 Creating new endpoint: {endpoint_name}")
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                description=f"Book sales ARIMA model endpoint for {endpoint_name}",
                labels={
                    "model_type": "arima",
                    "framework": "statsmodels",
                    "purpose": "book_sales_forecasting"
                }
            )
            logger.info(f"✅ Created endpoint: {endpoint.display_name}")
            return endpoint

        except Exception as e:
            logger.error(f"Failed to create/get endpoint {endpoint_name}: {e}")
            raise
    
    def upload_model_to_vertex(self, model_name: str, gcs_uri: str) -> Optional[aiplatform.Model]:
        """Upload model from GCS to Vertex AI Model Registry."""
        try:
            logger.info(f"📤 Uploading {model_name} to Vertex AI Model Registry...")
            
            # Use pre-built scikit-learn container that supports statsmodels
            serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
            
            # Upload model
            model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=gcs_uri,
                serving_container_image_uri=serving_container_image_uri,
                description=f"ARIMA model for book sales forecasting - {model_name}",
                labels={
                    "model_type": "arima",
                    "framework": "statsmodels", 
                    "isbn": model_name.replace("arima_book_", ""),
                    "deployment_type": "vertex_ai_prebuilt"
                }
            )
            
            logger.info(f"✅ Model uploaded to Vertex AI: {model.display_name}")
            logger.info(f"   Model ID: {model.name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to upload model {model_name}: {e}")
            return None
    
    def deploy_model_to_endpoint(self, 
                                model: aiplatform.Model, 
                                endpoint: aiplatform.Endpoint,
                                machine_type: str = "n1-standard-2") -> bool:
        """Deploy model to endpoint."""
        try:
            logger.info(f"🚀 Deploying {model.display_name} to endpoint {endpoint.display_name}")
            
            # Check if model is already deployed
            deployed_models = endpoint.list_models()
            for deployed_model in deployed_models:
                if model.display_name in deployed_model.display_name:
                    logger.info(f"✅ Model {model.display_name} already deployed")
                    return True
            
            # Deploy model to endpoint
            deployed_model = model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=f"{model.display_name}_deployment",
                machine_type=machine_type,
                min_replica_count=1,
                max_replica_count=2,
                traffic_percentage=100,
                sync=True  # Wait for deployment to complete
            )
            
            logger.info(f"✅ Model deployed successfully!")
            logger.info(f"   Deployed model ID: {deployed_model.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model to endpoint: {e}")
            return False
    
    def deploy_single_model(self, model_name: str) -> bool:
        """Deploy a single model from GCS to Vertex AI endpoint."""
        try:
            # Find model in GCS
            available_models = self.list_uploaded_models()
            model_info = None
            
            for model in available_models:
                if model["name"] == model_name:
                    model_info = model
                    break
            
            if not model_info:
                logger.error(f"Model {model_name} not found in GCS")
                logger.info("Run: python 02_upload_models_to_gcs.py --upload-all")
                return False
            
            isbn = model_info["isbn"]
            gcs_uri = model_info["gcs_uri"]
            endpoint_name = f"book-sales-{isbn}"
            
            logger.info(f"📖 Deploying {model_name} (ISBN: {isbn})")
            
            # Create or get endpoint
            endpoint = self.create_or_get_endpoint(endpoint_name)
            
            # Upload model to Vertex AI
            model = self.upload_model_to_vertex(model_name, gcs_uri)
            if not model:
                return False
            
            # Deploy to endpoint
            success = self.deploy_model_to_endpoint(model, endpoint)
            
            if success:
                logger.info(f"🎉 {model_name} successfully deployed!")
                logger.info(f"   Endpoint: {endpoint_name}")
                logger.info(f"   GCS URI: {gcs_uri}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}: {e}")
            return False
    
    def deploy_all_models(self) -> Dict[str, bool]:
        """Deploy all available models from GCS to endpoints."""
        available_models = self.list_uploaded_models()
        
        if not available_models:
            logger.warning("No models found in GCS for deployment")
            logger.info("Run: python 02_upload_models_to_gcs.py --upload-all")
            return {}
        
        logger.info(f"🚀 Starting deployment of {len(available_models)} models...")
        results = {}
        
        for model_info in available_models:
            model_name = model_info["name"]
            logger.info(f"📦 Processing {model_name}...")
            
            success = self.deploy_single_model(model_name)
            results[model_name] = success
            
            if success:
                logger.info(f"✅ {model_name} deployed successfully")
            else:
                logger.error(f"❌ {model_name} deployment failed")
            
            # Small delay between deployments
            time.sleep(5)
        
        # Summary
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        logger.info(f"📊 Deployment complete: {success_count}/{total_count} models deployed")
        
        return results
    
    def list_endpoints_with_models(self) -> List[Dict]:
        """List all endpoints and their deployed models."""
        try:
            endpoints_info = []
            
            endpoints = aiplatform.Endpoint.list()
            logger.info(f"Found {len(endpoints)} total endpoints")
            
            for endpoint in endpoints:
                if endpoint.display_name.startswith("book-sales-"):
                    deployed_models = endpoint.list_models()
                    
                    endpoint_info = {
                        "name": endpoint.display_name,
                        "id": endpoint.name,
                        "created": endpoint.create_time,
                        "models_deployed": len(deployed_models),
                        "models": []
                    }
                    
                    for model in deployed_models:
                        endpoint_info["models"].append({
                            "display_name": model.display_name,
                            "model_id": model.model,
                            "traffic_percentage": model.traffic_percentage
                        })
                    
                    endpoints_info.append(endpoint_info)
            
            return endpoints_info
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []
    
    def test_endpoint_prediction(self, endpoint_name: str, test_instances: List[Dict] = None) -> Dict:
        """Test prediction on a deployed endpoint."""
        try:
            # Find endpoint
            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
            if not endpoints:
                return {"error": f"Endpoint {endpoint_name} not found"}
            
            endpoint = endpoints[0]
            
            # Check if models are deployed
            deployed_models = endpoint.list_models()
            if not deployed_models:
                return {"error": f"No models deployed to endpoint {endpoint_name}"}
            
            # Default test instances
            if test_instances is None:
                test_instances = [
                    {"steps": 4},
                    {"steps": 12, "return_confidence_intervals": True}
                ]
            
            logger.info(f"🧪 Testing endpoint {endpoint_name} with {len(test_instances)} instances")
            
            # Make prediction
            response = endpoint.predict(instances=test_instances)
            
            return {
                "success": True,
                "endpoint": endpoint_name,
                "predictions": response.predictions,
                "model_version_id": response.model_version_id,
                "deployed_model_id": response.deployed_model_id
            }
            
        except Exception as e:
            logger.error(f"Endpoint test failed: {e}")
            return {"error": str(e)}


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Deploy models to Vertex AI endpoints using pre-built containers")
    
    parser.add_argument("--deploy-all", action="store_true", help="Deploy all available models")
    parser.add_argument("--model-name", help="Deploy specific model by name")
    parser.add_argument("--list-endpoints", action="store_true", help="List all endpoints and their models")
    parser.add_argument("--list-models", action="store_true", help="List models available for deployment")
    parser.add_argument("--test-endpoint", help="Test prediction on specific endpoint")
    
    parser.add_argument("--project-id", default="upheld-apricot-468313-e0", help="GCP project ID")
    parser.add_argument("--region", default="europe-west2", help="GCP region")
    parser.add_argument("--bucket-name", default="book-sales-deployment-artifacts", help="GCS bucket name")
    
    args = parser.parse_args()
    
    if not any([args.deploy_all, args.model_name, args.list_endpoints, args.list_models, args.test_endpoint]):
        parser.print_help()
        return
    
    # Initialize deployer
    try:
        deployer = VertexAIModelDeployer(
            project_id=args.project_id,
            region=args.region,
            bucket_name=args.bucket_name
        )
    except Exception as e:
        logger.error(f"Failed to initialize deployer: {e}")
        logger.info("Make sure you're authenticated with: gcloud auth login")
        return
    
    # Execute requested action
    if args.list_models:
        models = deployer.list_uploaded_models()
        if models:
            logger.info(f"📋 Models available for deployment ({len(models)}):")
            for model in models:
                logger.info(f"  📖 {model['name']} - ISBN: {model['isbn']} - {model['gcs_uri']}")
        else:
            logger.info("No models found in GCS")
            logger.info("Upload models first: python 02_upload_models_to_gcs.py --upload-all")
    
    if args.list_endpoints:
        endpoints = deployer.list_endpoints_with_models()
        if endpoints:
            logger.info(f"📋 Book sales endpoints ({len(endpoints)}):")
            for endpoint in endpoints:
                logger.info(f"  📡 {endpoint['name']} - {endpoint['models_deployed']} models deployed")
                for model in endpoint['models']:
                    logger.info(f"    📖 {model['display_name']} ({model['traffic_percentage']}% traffic)")
        else:
            logger.info("No book sales endpoints found")
    
    if args.test_endpoint:
        result = deployer.test_endpoint_prediction(args.test_endpoint)
        if result.get("success"):
            logger.info(f"🧪 Endpoint test successful!")
            logger.info(f"   Predictions: {len(result['predictions'])}")
            for i, pred in enumerate(result['predictions']):
                logger.info(f"   Prediction {i+1}: {pred}")
        else:
            logger.error(f"🧪 Endpoint test failed: {result.get('error')}")
    
    if args.deploy_all:
        logger.info("🚀 Starting deployment of all models...")
        results = deployer.deploy_all_models()
        
        if results:
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"🎉 Deployment complete: {success_count}/{len(results)} models deployed")
            
            for model_name, success in results.items():
                status = "✅" if success else "❌"
                logger.info(f"  {status} {model_name}")
            
            if success_count > 0:
                logger.info("🧪 Test your endpoints with:")
                for model_name, success in results.items():
                    if success:
                        isbn = model_name.replace("arima_book_", "")
                        endpoint_name = f"book-sales-{isbn}"
                        logger.info(f"   python 03_deploy_to_vertex_endpoints.py --test-endpoint {endpoint_name}")
        else:
            logger.warning("No models were available for deployment")
    
    elif args.model_name:
        logger.info(f"🚀 Deploying single model: {args.model_name}")
        success = deployer.deploy_single_model(args.model_name)
        
        if success:
            isbn = args.model_name.replace("arima_book_", "")
            endpoint_name = f"book-sales-{isbn}"
            logger.info(f"✅ Model deployed successfully!")
            logger.info(f"🧪 Test with: python 03_deploy_to_vertex_endpoints.py --test-endpoint {endpoint_name}")
        else:
            logger.error(f"❌ Failed to deploy {args.model_name}")


if __name__ == "__main__":
    main()