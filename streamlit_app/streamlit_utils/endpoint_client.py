"""
Vertex AI Endpoint Client for Streamlit App

Self-contained client for calling deployed Vertex AI endpoints.
Handles date-based prediction requests and response formatting.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

try:
    from google.cloud import aiplatform
    import vertexai
    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False
    logging.warning("Google Cloud AI Platform not available - using mock responses")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VertexAIEndpointClient:
    """Client for calling Vertex AI endpoints with book sales predictions."""
    
    def __init__(self, project_id: str = "upheld-apricot-468313-e0", region: str = "europe-west2"):
        """Initialize the endpoint client."""
        self.project_id = project_id
        self.region = region
        self.endpoints_cache = {}
        
        if HAS_VERTEX_AI:
            try:
                vertexai.init(project=project_id, location=region)
                aiplatform.init(project=project_id, location=region)
                self.vertex_available = True
                logger.info(f"Vertex AI client initialized for project: {project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI: {e}")
                self.vertex_available = False
        else:
            self.vertex_available = False
        
        # Book metadata for display
        self.book_metadata = {
            "9780722532935": {
                "title": "The Alchemist",
                "author": "Paulo Coelho",
                "endpoint_name": "book-sales-9780722532935"
            },
            "9780241003008": {
                "title": "The Very Hungry Caterpillar", 
                "author": "Eric Carle",
                "endpoint_name": "book-sales-9780241003008"
            }
        }

    def get_available_books(self) -> Dict[str, Dict[str, str]]:
        """Get list of available books for prediction."""
        return self.book_metadata

    def calculate_forecast_steps(self, target_date: datetime, last_known_date: datetime = None) -> int:
        """Calculate number of weeks from last known date to target date."""
        if last_known_date is None:
            # Default to roughly end of 2023 based on your training data
            last_known_date = datetime(2023, 12, 31)
        
        time_diff = target_date - last_known_date
        weeks_ahead = max(1, int(time_diff.days / 7))
        
        logger.info(f"Calculating forecast: {weeks_ahead} weeks from {last_known_date.date()} to {target_date.date()}")
        return weeks_ahead

    def find_endpoint(self, endpoint_name: str) -> Optional[Any]:
        """Find endpoint by name."""
        if not self.vertex_available:
            return None
            
        try:
            # Check cache first
            if endpoint_name in self.endpoints_cache:
                return self.endpoints_cache[endpoint_name]
            
            # Search for endpoint
            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
            if endpoints:
                endpoint = endpoints[0]
                self.endpoints_cache[endpoint_name] = endpoint
                logger.info(f"Found endpoint: {endpoint_name}")
                return endpoint
            else:
                logger.warning(f"Endpoint not found: {endpoint_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error finding endpoint {endpoint_name}: {e}")
            return None

    def get_prediction(self, isbn: str, target_date: datetime) -> Dict[str, Any]:
        """Get sales prediction for a specific book and date."""
        if isbn not in self.book_metadata:
            return {
                "error": f"Book with ISBN {isbn} not available",
                "available_books": list(self.book_metadata.keys())
            }
        
        book_info = self.book_metadata[isbn]
        endpoint_name = book_info["endpoint_name"]
        
        # Calculate forecast steps
        forecast_steps = self.calculate_forecast_steps(target_date)
        
        if self.vertex_available:
            try:
                return self._get_real_prediction(endpoint_name, book_info, target_date, forecast_steps)
            except Exception as e:
                logger.error(f"Real prediction failed: {e}")
                return self._get_mock_prediction(book_info, target_date, forecast_steps)
        else:
            return self._get_mock_prediction(book_info, target_date, forecast_steps)

    def _get_real_prediction(self, endpoint_name: str, book_info: Dict, target_date: datetime, forecast_steps: int) -> Dict[str, Any]:
        """Get prediction from real Vertex AI endpoint."""
        endpoint = self.find_endpoint(endpoint_name)
        
        if not endpoint:
            return {
                "error": f"Endpoint {endpoint_name} not found or not deployed",
                "suggestion": "Deploy models first: python 02_upload_models_to_gcs.py --upload-all && python 03_deploy_to_vertex_endpoints.py --deploy-all"
            }
        
        # Check if models are deployed to this endpoint
        deployed_models = endpoint.list_models()
        if not deployed_models:
            return {
                "error": f"No models deployed to endpoint {endpoint_name}",
                "suggestion": "Deploy models to endpoint: python 03_deploy_to_vertex_endpoints.py --deploy-all"
            }
        
        try:
            # Prepare prediction request for the deployed ARIMA model
            # The pre-built container expects instances in this format
            instances = [{
                "steps": forecast_steps,
                "return_confidence_intervals": False,
                "confidence_level": 0.95
            }]
            
            logger.info(f"Making prediction request to {endpoint_name} with {forecast_steps} steps")
            
            # Make prediction call
            response = endpoint.predict(instances=instances)
            predictions = response.predictions
            
            if predictions and len(predictions) > 0:
                prediction_result = predictions[0]
                
                # Extract forecast values from the prediction result
                if isinstance(prediction_result, dict) and "forecast" in prediction_result:
                    forecast_values = prediction_result["forecast"]
                    # Take the last forecasted value (the target date prediction)
                    predicted_value = forecast_values[-1] if forecast_values else 0
                    
                    # Format response
                    return {
                        "isbn": prediction_result.get("isbn", book_info.get("isbn", "unknown")),
                        "title": book_info["title"],
                        "author": book_info["author"],
                        "target_date": target_date.date().isoformat(),
                        "forecast_steps": forecast_steps,
                        "predicted_sales": round(float(predicted_value), 1),
                        "confidence_level": "ARIMA model prediction",
                        "endpoint_used": endpoint_name,
                        "prediction_type": "real",
                        "full_forecast": forecast_values,
                        "model_info": {
                            "model_name": prediction_result.get("model_name", "unknown"),
                            "forecast_length": len(forecast_values) if forecast_values else 0
                        }
                    }
                else:
                    # Handle simple numeric prediction
                    predicted_value = float(prediction_result)
                    
                    return {
                        "isbn": book_info.get("isbn", "unknown"),
                        "title": book_info["title"],
                        "author": book_info["author"],
                        "target_date": target_date.date().isoformat(),
                        "forecast_steps": forecast_steps,
                        "predicted_sales": round(predicted_value, 1),
                        "confidence_level": "ARIMA model prediction",
                        "endpoint_used": endpoint_name,
                        "prediction_type": "real"
                    }
            else:
                raise ValueError("Empty prediction response from endpoint")
                
        except Exception as e:
            logger.error(f"Endpoint prediction failed: {e}")
            # Fall back to mock prediction
            return self._get_mock_prediction(book_info, target_date, forecast_steps)

    def _get_mock_prediction(self, book_info: Dict, target_date: datetime, forecast_steps: int) -> Dict[str, Any]:
        """Generate mock prediction for development/testing."""
        # Generate realistic mock predictions based on book patterns
        
        # Base predictions with some seasonality
        base_predictions = {
            "9780722532935": 450,  # The Alchemist - steady seller
            "9780241003008": 320   # Very Hungry Caterpillar - children's book
        }
        
        isbn = None
        for book_isbn, info in self.book_metadata.items():
            if info["title"] == book_info["title"]:
                isbn = book_isbn
                break
        
        base_value = base_predictions.get(isbn, 300)
        
        # Add some seasonal variation
        month = target_date.month
        seasonal_factor = 1.0
        
        if month in [11, 12]:  # Holiday season
            seasonal_factor = 1.3
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 0.8 if isbn == "9780241003008" else 1.1  # Children's books dip in summer
        
        # Add some random variation
        np.random.seed(hash(f"{isbn}_{target_date.date()}") % 2**32)
        random_factor = np.random.uniform(0.85, 1.15)
        
        predicted_value = base_value * seasonal_factor * random_factor
        
        return {
            "isbn": isbn,
            "title": book_info["title"],
            "author": book_info["author"],
            "target_date": target_date.date().isoformat(),
            "forecast_steps": forecast_steps,
            "predicted_sales": round(predicted_value, 1),
            "confidence_level": "Mock prediction for development",
            "endpoint_used": book_info["endpoint_name"],
            "prediction_type": "mock",
            "note": "This is a mock prediction. Deploy models to get real predictions."
        }

    def get_multiple_predictions(self, isbn: str, start_date: datetime, num_weeks: int = 12) -> List[Dict[str, Any]]:
        """Get predictions for multiple weeks ahead."""
        predictions = []
        
        for week in range(num_weeks):
            target_date = start_date + timedelta(weeks=week)
            prediction = self.get_prediction(isbn, target_date)
            predictions.append(prediction)
        
        return predictions

    def health_check(self) -> Dict[str, Any]:
        """Check client health and available endpoints."""
        status = {
            "vertex_ai_available": False,  # Default to False, will update based on actual checks
            "project_id": self.project_id,
            "region": self.region,
            "available_books": len(self.book_metadata),
            "endpoints_cached": len(self.endpoints_cache)
        }
        
        if self.vertex_available:
            try:
                # Try to list endpoints
                endpoints = aiplatform.Endpoint.list()
                status["total_endpoints"] = len(endpoints)
                status["book_endpoints_found"] = []
                deployed_count = 0
                
                for book_isbn, book_info in self.book_metadata.items():
                    endpoint_name = book_info["endpoint_name"]
                    found_endpoint = None
                    for ep in endpoints:
                        if ep.display_name == endpoint_name:
                            found_endpoint = ep
                            break
                    
                    deployed = found_endpoint is not None
                    if deployed:
                        # Check if endpoint has deployed models
                        try:
                            deployed_models = found_endpoint.list_models()
                            has_models = len(deployed_models) > 0
                            if has_models:
                                deployed_count += 1
                        except:
                            has_models = False
                    else:
                        has_models = False
                    
                    status["book_endpoints_found"].append({
                        "isbn": book_isbn,
                        "title": book_info["title"],
                        "endpoint": endpoint_name,
                        "deployed": deployed and has_models
                    })
                
                # Only mark as available if we have actual deployed models
                status["vertex_ai_available"] = deployed_count > 0
                status["deployed_models_count"] = deployed_count
                    
            except Exception as e:
                status["endpoint_check_error"] = str(e)
                status["vertex_ai_available"] = False
        
        return status


# Convenience function for Streamlit
def create_client() -> VertexAIEndpointClient:
    """Create and return a configured endpoint client."""
    return VertexAIEndpointClient()


# Example usage and testing
if __name__ == "__main__":
    client = create_client()
    
    # Health check
    health = client.health_check()
    print("Health Check:")
    print(json.dumps(health, indent=2))
    
    # Test prediction
    test_date = datetime.now() + timedelta(days=30)
    prediction = client.get_prediction("9780722532935", test_date)
    print("\nSample Prediction:")
    print(json.dumps(prediction, indent=2, default=str))