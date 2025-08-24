#!/usr/bin/env python3
"""
Custom Vertex AI Prediction Container for MLflow ARIMA Models

This module implements a custom predictor for serving ARIMA models on Vertex AI.
It loads MLflow statsmodels artifacts and provides time series forecasting.

The predictor supports:
- Loading SARIMA models from MLflow registry or runs
- Time series forecasting with configurable forecast horizons
- Input validation and error handling
- Health checks and model metadata

Usage:
    This file is used inside a Docker container deployed to Vertex AI endpoints.
    The container receives prediction requests and returns forecasts.
"""

import json
import logging
import os
from typing import Dict, List, Any
import traceback

import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIMAPredictor:
    """Custom predictor for MLflow ARIMA models deployed on Vertex AI."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.model_metadata = {}
        self.model_name = None
        self.model_version = None
        self.isbn = None
        self.mlflow_uri = None
        
    def load(self, artifacts_uri: str) -> None:
        """
        Load the ARIMA model from MLflow artifacts.
        
        Args:
            artifacts_uri: MLflow model URI (models:/model_name/version or runs:/run_id/model)
        """
        try:
            logger.info(f"Loading MLflow model from: {artifacts_uri}")
            
            # Set MLflow tracking URI from environment or default
            self.mlflow_uri = os.environ.get(
                "MLFLOW_TRACKING_URI", 
                "https://mlflow-tracking-server-1076639696283.europe-west2.run.app"
            )
            mlflow.set_tracking_uri(self.mlflow_uri)
            logger.info(f"MLflow tracking URI: {self.mlflow_uri}")
            
            # Load model using MLflow
            self.model = mlflow.statsmodels.load_model(artifacts_uri)
            
            # Extract model info from artifacts_uri
            if artifacts_uri.startswith("models:/"):
                self._load_registry_metadata(artifacts_uri)
            elif artifacts_uri.startswith("runs:/"):
                self._load_run_metadata(artifacts_uri)
            else:
                logger.warning(f"Unsupported model URI format: {artifacts_uri}")
                self.model_name = "unknown"
                
            # Extract ISBN from model name if available
            if self.model_name and "arima_book_" in self.model_name:
                self.isbn = self.model_name.replace("arima_book_", "")
                
            logger.info(f"✅ Successfully loaded model: {self.model_name}")
            logger.info(f"📚 ISBN: {self.isbn}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_registry_metadata(self, model_uri: str) -> None:
        """Load metadata from MLflow model registry."""
        try:
            # Parse model URI: models:/model_name/version
            parts = model_uri.replace("models:/", "").split("/")
            if len(parts) >= 2:
                self.model_name = parts[0]
                self.model_version = parts[1]
            
            logger.info(f"Loading metadata for registry model: {self.model_name} v{self.model_version}")
            
            # Get model metadata from registry
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version(self.model_name, self.model_version)
            
            self.model_metadata = {
                "source": "registry",
                "name": self.model_name,
                "version": self.model_version,
                "stage": model_version.current_stage,
                "description": model_version.description or "",
                "tags": model_version.tags or {},
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp
            }
            
            logger.info(f"Model stage: {model_version.current_stage}")
            
        except Exception as e:
            logger.warning(f"Could not load registry metadata: {e}")
            self.model_metadata = {"source": "registry", "error": str(e)}
    
    def _load_run_metadata(self, model_uri: str) -> None:
        """Load metadata from MLflow run."""
        try:
            # Parse model URI: runs:/run_id/model
            run_id = model_uri.replace("runs:/", "").split("/")[0]
            logger.info(f"Loading metadata for run: {run_id}")
            
            # Get run metadata
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            self.model_metadata = {
                "source": "run",
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "tags": run.data.tags,
                "params": run.data.params,
                "metrics": run.data.metrics
            }
            
            # Extract model name from tags or params
            if "isbn" in run.data.tags:
                self.isbn = run.data.tags["isbn"]
                self.model_name = f"arima_book_{self.isbn}"
            elif "isbn" in run.data.params:
                self.isbn = run.data.params["isbn"]
                self.model_name = f"arima_book_{self.isbn}"
            else:
                self.model_name = f"arima_model_{run_id[:8]}"
                
            logger.info(f"Extracted model name: {self.model_name}")
            
        except Exception as e:
            logger.warning(f"Could not load run metadata: {e}")
            self.model_metadata = {"source": "run", "error": str(e)}
    
    def predict(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make predictions using the loaded ARIMA model.
        
        Args:
            instances: List of prediction instances, each containing:
                - steps: Number of forecast steps (default: 12)
                - return_confidence_intervals: Whether to return confidence intervals (default: False)
                - confidence_level: Confidence level for intervals (default: 0.95)
        
        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load() first.")
            
            logger.info(f"Making prediction with {len(instances)} instances")
            
            predictions = []
            
            for i, instance in enumerate(instances):
                try:
                    # Parse prediction parameters
                    steps = instance.get('steps', 12)
                    return_confidence_intervals = instance.get('return_confidence_intervals', False)
                    confidence_level = instance.get('confidence_level', 0.95)
                    
                    logger.info(f"Instance {i}: forecasting {steps} steps ahead")
                    
                    # Make forecast
                    if return_confidence_intervals:
                        # Get forecast with confidence intervals
                        forecast_result = self.model.get_forecast(steps=steps)
                        forecast_values = forecast_result.predicted_mean.tolist()
                        
                        # Get confidence intervals
                        conf_int = forecast_result.conf_int(alpha=1-confidence_level)
                        lower_bounds = conf_int.iloc[:, 0].tolist()
                        upper_bounds = conf_int.iloc[:, 1].tolist()
                        
                        prediction = {
                            "forecast": forecast_values,
                            "confidence_intervals": {
                                "lower": lower_bounds,
                                "upper": upper_bounds,
                                "confidence_level": confidence_level
                            }
                        }
                    else:
                        # Simple forecast without confidence intervals
                        forecast = self.model.forecast(steps=steps)
                        forecast_values = forecast.tolist() if hasattr(forecast, 'tolist') else [float(forecast)]
                        
                        prediction = {
                            "forecast": forecast_values,
                            "confidence_intervals": None
                        }
                    
                    # Add metadata
                    prediction.update({
                        "steps": steps,
                        "model_name": self.model_name,
                        "isbn": self.isbn,
                        "forecast_type": "SARIMA",
                        "model_params": self._get_model_params()
                    })
                    
                    predictions.append(prediction)
                    logger.info(f"✅ Instance {i} completed: {len(forecast_values)} forecast values")
                    
                except Exception as instance_error:
                    logger.error(f"❌ Instance {i} failed: {instance_error}")
                    predictions.append({
                        "error": str(instance_error),
                        "model_name": self.model_name,
                        "isbn": self.isbn,
                        "instance_index": i
                    })
            
            # Return response
            response = {
                "predictions": predictions,
                "model_metadata": {
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "isbn": self.isbn,
                    "mlflow_uri": self.mlflow_uri,
                    "total_instances": len(instances),
                    "successful_predictions": len([p for p in predictions if "error" not in p]),
                    "failed_predictions": len([p for p in predictions if "error" in p])
                }
            }
            
            logger.info(f"✅ Completed {len(predictions)} predictions")
            return response
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "error": str(e),
                "model_metadata": {
                    "model_name": self.model_name,
                    "isbn": self.isbn,
                    "status": "prediction_error"
                }
            }
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Extract model parameters if available."""
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'specification'):
                spec = self.model.model.specification
                return {
                    "order": getattr(spec, 'order', None),
                    "seasonal_order": getattr(spec, 'seasonal_order', None),
                    "trend": getattr(spec, 'trend', None)
                }
            elif hasattr(self.model, 'specification'):
                spec = self.model.specification
                return {
                    "order": getattr(spec, 'order', None),
                    "seasonal_order": getattr(spec, 'seasonal_order', None),
                    "trend": getattr(spec, 'trend', None)
                }
            else:
                return {"note": "Model parameters not accessible"}
        except Exception as e:
            logger.warning(f"Could not extract model parameters: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model.
        
        Returns:
            Health status information
        """
        try:
            status = {
                "status": "healthy" if self.model is not None else "unhealthy",
                "model_loaded": self.model is not None,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "isbn": self.isbn,
                "mlflow_uri": self.mlflow_uri,
                "model_type": "SARIMA"
            }
            
            # Add metadata if available
            if self.model_metadata:
                status["metadata_available"] = True
                status["metadata_source"] = self.model_metadata.get("source", "unknown")
            else:
                status["metadata_available"] = False
            
            # Test prediction if model is loaded
            if self.model is not None:
                try:
                    test_forecast = self.model.forecast(steps=1)
                    status["test_prediction"] = "success"
                    status["test_forecast_value"] = float(test_forecast.iloc[0] if hasattr(test_forecast, 'iloc') else test_forecast)
                except Exception as test_error:
                    status["test_prediction"] = "failed"
                    status["test_error"] = str(test_error)
                    status["status"] = "unhealthy"
            
            return status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": False
            }


# Global predictor instance
predictor = ARIMAPredictor()


def load_model(artifacts_uri: str) -> None:
    """Load the model (called by Vertex AI)."""
    global predictor
    predictor.load(artifacts_uri)


def predict(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Make predictions (called by Vertex AI)."""
    global predictor
    return predictor.predict(instances)


def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    global predictor
    return predictor.health_check()


if __name__ == "__main__":
    # Test the predictor locally
    import sys
    
    if len(sys.argv) > 1:
        model_uri = sys.argv[1]
        print(f"Testing predictor with model: {model_uri}")
        
        # Load model
        load_model(model_uri)
        
        # Test health check
        health = health_check()
        print(f"Health check: {json.dumps(health, indent=2)}")
        
        # Test prediction
        test_instances = [
            {"steps": 12, "return_confidence_intervals": True, "confidence_level": 0.95},
            {"steps": 4, "return_confidence_intervals": False}
        ]
        
        result = predict(test_instances)
        print(f"Prediction result: {json.dumps(result, indent=2, default=str)}")
    else:
        print("Usage: python predictor.py <model_uri>")
        print("Example: python predictor.py models:/arima_book_9780722532935/latest")
        print("Example: python predictor.py runs:/abc123def456/model")