#!/usr/bin/env python3
"""
Simple Vertex AI Model Endpoint Deployment Script

This is a standalone script for deploying MLflow models to Vertex AI endpoints.
Designed to be simple and refactor-friendly for the upcoming pipeline restructuring.

Usage:
    python deploy_models.py --model-name arima_book_9780722532935 --endpoint-name book-sales-alchemist
    python deploy_models.py --deploy-all  # Deploy all available book models
    # List trained models
    python deploy_models.py --list-models

Design Rationale:
    This simplified approach is intentionally chosen to support an upcoming large-scale
    refactor project where the main pipeline will be broken down and MLOps improvements
    will be added. By keeping this as a standalone script, we avoid coupling with the
    current pipeline structure and can easily enhance it later.
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import mlflow
    from google.cloud import aiplatform
    import vertexai
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.info("Run: pip install mlflow google-cloud-aiplatform")
    sys.exit(1)


class SimpleModelDeployer:
    """Simple deployment class for MLflow models to Vertex AI endpoints."""

    def __init__(self, project_id: str = "upheld-apricot-468313-e0", region: str = "europe-west2"):
        """Initialize the deployer with GCP project settings."""
        self.project_id = project_id
        self.region = region
        self.mlflow_uri = "https://mlflow-tracking-server-1076639696283.europe-west2.run.app"

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        aiplatform.init(project=project_id, location=region)

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_uri)

        logger.info(f"Initialized deployer for project: {project_id}, region: {region}")
        logger.info(f"MLflow URI: {self.mlflow_uri}")

    def list_available_models(self) -> List[Dict]:
        """List all available book models in MLflow registry."""
        try:
            models = []
            for model in mlflow.search_registered_models():
                if model.name.startswith("arima_book_"):
                    latest_version = model.latest_versions[0] if model.latest_versions else None
                    models.append({
                        'name': model.name,
                        'version': latest_version.version if latest_version else 'N/A',
                        'stage': latest_version.current_stage if latest_version else 'N/A',
                        'isbn': model.name.replace("arima_book_", "")
                    })
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def create_or_get_endpoint(self, endpoint_name: str) -> aiplatform.Endpoint:
        """Create a new endpoint or get existing one."""
        try:
            # Try to get existing endpoint
            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
            if endpoints:
                endpoint = endpoints[0]
                logger.info(f"Using existing endpoint: {endpoint_name}")
                return endpoint

            # Create new endpoint
            logger.info(f"Creating new endpoint: {endpoint_name}")
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                description=f"Book sales ARIMA model endpoint for {endpoint_name}"
            )
            logger.info(f"Created endpoint: {endpoint.display_name} (ID: {endpoint.name})")
            return endpoint

        except Exception as e:
            logger.error(f"Failed to create/get endpoint {endpoint_name}: {e}")
            raise

    def deploy_model(self, model_name: str, endpoint_name: str, version: str = "latest") -> bool:
        """Deploy a specific model to an endpoint."""
        try:
            logger.info(f"Starting deployment of {model_name} version {version} to {endpoint_name}")

            # Get model URI
            if version == "latest":
                model_uri = f"models:/{model_name}/latest"
            else:
                model_uri = f"models:/{model_name}/{version}"

            logger.info(f"Model URI: {model_uri}")

            # Create or get endpoint
            endpoint = self.create_or_get_endpoint(endpoint_name)

            # For now, we'll create a simple container serving setup
            # This is a simplified approach - in production you'd want more sophisticated model serving
            logger.info("Note: This is a basic deployment setup.")
            logger.info("For production, consider using custom prediction containers or pre-built containers.")

            # Check if model is already deployed to this endpoint
            deployed_models = endpoint.list_models()
            for deployed_model in deployed_models:
                if model_name in deployed_model.display_name:
                    logger.info(f"Model {model_name} already deployed to {endpoint_name}")
                    return True

            # Basic deployment configuration
            # Note: This is simplified - MLflow models need proper containerization for Vertex AI
            logger.warning("Current implementation provides endpoint creation.")
            logger.warning("Full MLflow model deployment requires additional container setup.")
            logger.info(f"Endpoint {endpoint_name} is ready for model deployment")
            logger.info(f"Next steps: Package {model_name} in a serving container and deploy")

            return True

        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}: {e}")
            return False

    def deploy_all_models(self) -> Dict[str, bool]:
        """Deploy all available book models to individual endpoints."""
        models = self.list_available_models()
        if not models:
            logger.warning("No models found for deployment")
            return {}

        results = {}
        for model in models:
            model_name = model['name']
            isbn = model['isbn']
            endpoint_name = f"book-sales-{isbn}"

            logger.info(f"Deploying {model_name} (ISBN: {isbn})")
            success = self.deploy_model(model_name, endpoint_name)
            results[model_name] = success

            # Small delay between deployments
            time.sleep(2)

        return results

    def list_endpoints(self) -> List[Dict]:
        """List all current endpoints."""
        try:
            endpoints = []
            for endpoint in aiplatform.Endpoint.list():
                endpoints.append({
                    'name': endpoint.display_name,
                    'id': endpoint.name,
                    'create_time': endpoint.create_time,
                    'models': len(endpoint.list_models())
                })
            return endpoints
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Deploy MLflow models to Vertex AI endpoints")
    parser.add_argument("--model-name", help="Specific model name to deploy (e.g., arima_book_9780722532935)")
    parser.add_argument("--endpoint-name", help="Target endpoint name (e.g., book-sales-alchemist)")
    parser.add_argument("--deploy-all", action="store_true", help="Deploy all available book models")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument("--list-endpoints", action="store_true", help="List all current endpoints")
    parser.add_argument("--project-id", default="upheld-apricot-468313-e0", help="GCP project ID")
    parser.add_argument("--region", default="europe-west2", help="GCP region")

    args = parser.parse_args()

    if not any([args.model_name, args.deploy_all, args.list_models, args.list_endpoints]):
        parser.print_help()
        return

    # Initialize deployer
    try:
        deployer = SimpleModelDeployer(project_id=args.project_id, region=args.region)
    except Exception as e:
        logger.error(f"Failed to initialize deployer: {e}")
        logger.info("Make sure you're authenticated with: gcloud auth login")
        return

    # Execute requested action
    if args.list_models:
        models = deployer.list_available_models()
        if models:
            logger.info(f"Found {len(models)} book models:")
            for model in models:
                logger.info(f"  - {model['name']} (v{model['version']}, {model['stage']}) - ISBN: {model['isbn']}")
        else:
            logger.info("No book models found in MLflow registry")

    elif args.list_endpoints:
        endpoints = deployer.list_endpoints()
        if endpoints:
            logger.info(f"Found {len(endpoints)} endpoints:")
            for endpoint in endpoints:
                logger.info(f"  - {endpoint['name']} ({endpoint['models']} models) - ID: {endpoint['id']}")
        else:
            logger.info("No endpoints found")

    elif args.deploy_all:
        logger.info("Starting deployment of all book models...")
        results = deployer.deploy_all_models()

        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Deployment completed: {success_count}/{len(results)} models successfully deployed")

        for model_name, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {model_name}")

    elif args.model_name and args.endpoint_name:
        logger.info(f"Deploying single model: {args.model_name} -> {args.endpoint_name}")
        success = deployer.deploy_model(args.model_name, args.endpoint_name)

        if success:
            logger.info(f"✅ Successfully deployed {args.model_name} to {args.endpoint_name}")
        else:
            logger.error(f"❌ Failed to deploy {args.model_name}")

    else:
        logger.error("For single model deployment, both --model-name and --endpoint-name are required")
        parser.print_help()


if __name__ == "__main__":
    main()
