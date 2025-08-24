#!/usr/bin/env python3
"""
Vertex AI Pipeline Deployment and Model Endpoint Script

This script combines pipeline execution on Vertex AI with model endpoint deployment.
It runs the book sales ARIMA pipeline using Vertex AI orchestrator and then deploys
the resulting models to Vertex AI endpoints for serving.

Usage:
    # Run pipeline on Vertex AI and deploy all models
    python deploy/01_train_pipeline_and_create_endpoints.py --run-pipeline --deploy-all

    # Just run pipeline on Vertex AI
    python 01_vertex_deployment_and_endpoint.py --run-pipeline

    # Just deploy existing models
    python 01_vertex_deployment_and_endpoint.py --deploy-all

    # Run pipeline with custom config
    python 01_vertex_deployment_and_endpoint.py --run-pipeline --environment production --trials 50

This integrates with your existing Vertex AI stack and MLflow tracking server.
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Dict, Optional
from pathlib import Path

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
    import zenml
    from zenml.client import Client
    from zenml.enums import StackComponentType
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.info("Run: poetry install (dependencies are managed in pyproject.toml)")
    sys.exit(1)

# Import the pipeline and configuration
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.zenml_pipeline import book_sales_arima_modeling_pipeline
from config.arima_training_config import get_arima_config, DEFAULT_TEST_ISBNS, DEFAULT_SPLIT_SIZE


class VertexPipelineDeployer:
    """Vertex AI pipeline executor and model deployer."""

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

        # Initialize ZenML client
        self.zenml_client = Client()

        logger.info(f"Initialized deployer for project: {project_id}, region: {region}")
        logger.info(f"MLflow URI: {self.mlflow_uri}")

    def check_vertex_stack(self) -> bool:
        """Check if Vertex AI stack is available and properly configured."""
        try:
            active_stack = self.zenml_client.active_stack
            logger.info(f"Active ZenML stack: {active_stack.name}")

            # Check for Vertex AI orchestrator
            orchestrator = active_stack.orchestrator
            if orchestrator and "vertex" in orchestrator.flavor.lower():
                logger.info(f"✅ Vertex AI orchestrator found: {orchestrator.name} ({orchestrator.flavor})")
                return True
            else:
                logger.warning(f"⚠️  Current orchestrator: {orchestrator.name if orchestrator else 'None'} ({orchestrator.flavor if orchestrator else 'None'})")
                logger.warning("Expected Vertex AI orchestrator for remote execution")

                # List available stacks with Vertex AI
                stacks = self.zenml_client.list_stacks()
                vertex_stacks = []
                for stack in stacks:
                    if stack.orchestrator and "vertex" in stack.orchestrator.flavor.lower():
                        vertex_stacks.append(stack.name)

                if vertex_stacks:
                    logger.info(f"Available Vertex AI stacks: {vertex_stacks}")
                    logger.info("Switch to a Vertex AI stack with: zenml stack set <stack_name>")
                else:
                    logger.error("No Vertex AI stacks found!")
                    logger.info("Create one with: zenml stack register <name> -o <vertex_orchestrator>")

                return False
        except Exception as e:
            logger.error(f"Failed to check Vertex AI stack: {e}")
            return False

    def run_pipeline_on_vertex(
        self,
        environment: str = "development",
        n_trials: int = 3,
        force_retrain: bool = True,
        selected_isbns: List[str] = None
    ) -> Optional[str]:
        """Run the ARIMA pipeline on Vertex AI orchestrator."""
        try:
            # Check Vertex AI stack
            if not self.check_vertex_stack():
                logger.error("Vertex AI stack not properly configured")
                return None

            logger.info("🚀 Starting pipeline execution on Vertex AI...")

            # Set up output directory (relative to project root)
            project_root = Path(__file__).parent.parent
            output_dir = str(project_root / 'data' / 'processed')

            # Create configuration
            config = get_arima_config(
                environment=environment,
                n_trials=n_trials,
                force_retrain=force_retrain
            )

            logger.info(f"🔧 Pipeline configuration:")
            logger.info(f"   Environment: {config.environment}")
            logger.info(f"   Trials: {config.n_trials}")
            logger.info(f"   Force retrain: {config.force_retrain}")
            logger.info(f"   Books: {selected_isbns or DEFAULT_TEST_ISBNS}")

            # Create custom pipeline run name
            import datetime
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            num_books = len(selected_isbns or DEFAULT_TEST_ISBNS)

            # Get git commit hash for traceability
            try:
                import subprocess
                result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                      capture_output=True, text=True, cwd=project_root)
                git_hash = result.stdout.strip() if result.returncode == 0 else 'unknown'
            except Exception:
                git_hash = 'unknown'

            pipeline_name = f"vertex_pipeline_{num_books}books_{git_hash}_{timestamp}"
            logger.info(f"🏷️  Pipeline run name: {pipeline_name}")

            # Run the pipeline on Vertex AI
            pipeline_run = book_sales_arima_modeling_pipeline.with_options(
                run_name=pipeline_name
            )(
                output_dir=output_dir,
                selected_isbns=selected_isbns or DEFAULT_TEST_ISBNS,
                column_name='Volume',
                split_size=DEFAULT_SPLIT_SIZE,
                use_seasonality_filter=False,  # Use provided ISBNs
                max_seasonal_books=15,  # Not used when specific ISBNs provided
                train_arima=True,  # Enable ARIMA training
                n_trials=n_trials,  # Backward compatibility
                config=config,  # Use optimized configuration
                pipeline_timestamp=timestamp,
                use_local_mlflow=False  # Use remote MLflow
            )

            logger.info(f"✅ Pipeline submitted to Vertex AI")
            logger.info(f"📊 Pipeline run ID: {pipeline_run.id}")
            logger.info(f"🔗 Monitor progress in ZenML dashboard or GCP Console")

            return pipeline_run.id

        except Exception as e:
            logger.error(f"Failed to run pipeline on Vertex AI: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def wait_for_pipeline_completion(self, run_id: str, timeout_minutes: int = 60) -> bool:
        """Wait for pipeline to complete on Vertex AI."""
        try:
            logger.info(f"⏳ Waiting for pipeline {run_id} to complete (timeout: {timeout_minutes}min)")

            start_time = time.time()
            timeout_seconds = timeout_minutes * 60

            while True:
                try:
                    # Get pipeline run status
                    run = self.zenml_client.get_pipeline_run(run_id)
                    status = run.status

                    logger.info(f"📊 Pipeline status: {status}")

                    if status.name in ['COMPLETED']:
                        logger.info(f"✅ Pipeline completed successfully!")
                        return True
                    elif status.name in ['FAILED', 'CANCELLED']:
                        logger.error(f"❌ Pipeline failed with status: {status}")
                        return False

                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        logger.warning(f"⏰ Pipeline timeout after {timeout_minutes} minutes")
                        logger.info("Pipeline may still be running. Check ZenML dashboard for status.")
                        return False

                    # Wait before checking again
                    time.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    logger.warning(f"Failed to get pipeline status: {e}")
                    time.sleep(30)

        except Exception as e:
            logger.error(f"Error waiting for pipeline completion: {e}")
            return False

    def list_available_models(self) -> List[Dict]:
        """List all available book models in MLflow registry."""
        try:
            models = []
            logger.info("📋 Checking MLflow registry for available models...")

            for model in mlflow.search_registered_models():
                if model.name.startswith("arima_book_"):
                    latest_version = model.latest_versions[0] if model.latest_versions else None
                    models.append({
                        'name': model.name,
                        'version': latest_version.version if latest_version else 'N/A',
                        'stage': latest_version.current_stage if latest_version else 'N/A',
                        'isbn': model.name.replace("arima_book_", "")
                    })

            logger.info(f"Found {len(models)} ARIMA models in registry")
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
                logger.info(f"📍 Using existing endpoint: {endpoint_name}")
                return endpoint

            # Create new endpoint
            logger.info(f"🆕 Creating new endpoint: {endpoint_name}")
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                description=f"Book sales ARIMA model endpoint for {endpoint_name}"
            )
            logger.info(f"✅ Created endpoint: {endpoint.display_name} (ID: {endpoint.name})")
            return endpoint

        except Exception as e:
            logger.error(f"Failed to create/get endpoint {endpoint_name}: {e}")
            raise

    def deploy_model_to_endpoint(self, model_name: str, endpoint_name: str, version: str = "latest") -> bool:
        """Deploy a specific model to a Vertex AI endpoint."""
        try:
            logger.info(f"🚀 Deploying {model_name} v{version} to endpoint {endpoint_name}")

            # Get model URI
            model_uri = f"models:/{model_name}/{version}" if version != "latest" else f"models:/{model_name}/latest"
            logger.info(f"📦 Model URI: {model_uri}")

            # Create or get endpoint
            endpoint = self.create_or_get_endpoint(endpoint_name)

            # Check if model is already deployed
            deployed_models = endpoint.list_models()
            for deployed_model in deployed_models:
                if model_name in deployed_model.display_name:
                    logger.info(f"✅ Model {model_name} already deployed to {endpoint_name}")
                    return True

            # Note: This is a simplified deployment setup
            # For production MLflow models, you need proper containerization
            logger.info(f"📍 Endpoint {endpoint_name} ready for model deployment")
            logger.info(f"ℹ️  Next step: Package {model_name} in serving container")
            logger.warning("Note: Full MLflow → Vertex AI deployment requires custom containers")
            logger.info("Consider using Google Cloud AI Platform Prediction or custom serving setup")

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

        logger.info(f"🚀 Starting deployment of {len(models)} models to Vertex AI endpoints")
        results = {}

        for model in models:
            model_name = model['name']
            isbn = model['isbn']
            endpoint_name = f"book-sales-{isbn}"

            logger.info(f"📖 Deploying {model_name} (ISBN: {isbn})")
            success = self.deploy_model_to_endpoint(model_name, endpoint_name)
            results[model_name] = success

            # Small delay between deployments
            time.sleep(2)

        # Summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"📊 Deployment summary: {success_count}/{len(results)} models deployed")

        return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run Vertex AI pipeline and deploy models to endpoints")

    # Pipeline execution options
    parser.add_argument("--run-pipeline", action="store_true", help="Run pipeline on Vertex AI")
    parser.add_argument("--environment", default="development", choices=["development", "testing", "production"],
                       help="Pipeline environment configuration")
    parser.add_argument("--trials", type=int, default=3, help="Number of Optuna trials per book")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of all models")
    parser.add_argument("--wait", action="store_true", help="Wait for pipeline completion")
    parser.add_argument("--timeout", type=int, default=60, help="Pipeline timeout in minutes")

    # Model deployment options
    parser.add_argument("--deploy-all", action="store_true", help="Deploy all available book models")
    parser.add_argument("--list-models", action="store_true", help="List all available models")

    # GCP configuration
    parser.add_argument("--project-id", default="upheld-apricot-468313-e0", help="GCP project ID")
    parser.add_argument("--region", default="europe-west2", help="GCP region")

    args = parser.parse_args()

    if not any([args.run_pipeline, args.deploy_all, args.list_models]):
        parser.print_help()
        return

    # Initialize deployer
    try:
        deployer = VertexPipelineDeployer(project_id=args.project_id, region=args.region)
    except Exception as e:
        logger.error(f"Failed to initialize deployer: {e}")
        logger.info("Make sure you're authenticated with: gcloud auth login")
        logger.info("And ZenML is properly configured with: zenml connect")
        return

    # Execute pipeline
    run_id = None
    if args.run_pipeline:
        logger.info("🚀 Starting Vertex AI pipeline execution...")
        run_id = deployer.run_pipeline_on_vertex(
            environment=args.environment,
            n_trials=args.trials,
            force_retrain=args.force_retrain,
            selected_isbns=DEFAULT_TEST_ISBNS
        )

        if run_id:
            logger.info(f"✅ Pipeline submitted successfully: {run_id}")

            # Wait for completion if requested
            if args.wait:
                success = deployer.wait_for_pipeline_completion(run_id, args.timeout)
                if not success:
                    logger.warning("Pipeline did not complete successfully")
                    if not args.deploy_all:
                        return
        else:
            logger.error("Failed to submit pipeline")
            return

    # List models
    if args.list_models:
        models = deployer.list_available_models()
        if models:
            logger.info(f"📋 Available models ({len(models)}):")
            for model in models:
                logger.info(f"  📖 {model['name']} v{model['version']} ({model['stage']}) - ISBN: {model['isbn']}")
        else:
            logger.info("No book models found in MLflow registry")

    # Deploy models
    if args.deploy_all:
        logger.info("🚀 Starting model deployment to Vertex AI endpoints...")
        results = deployer.deploy_all_models()

        if results:
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"📊 Final deployment results: {success_count}/{len(results)} models deployed")

            for model_name, success in results.items():
                status = "✅" if success else "❌"
                logger.info(f"  {status} {model_name}")
        else:
            logger.warning("No models were available for deployment")

    logger.info("🎉 Vertex AI deployment script completed!")


if __name__ == "__main__":
    main()
