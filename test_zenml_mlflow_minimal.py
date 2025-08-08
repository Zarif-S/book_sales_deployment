#!/usr/bin/env python3

"""
Minimal test to verify ZenML MLflow integration is working
"""

import mlflow
from zenml import step, pipeline
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

@step(experiment_tracker="mlflow_tracker")
def test_mlflow_step() -> str:
    """Test step that logs to MLflow"""
    
    # Set the experiment
    mlflow.set_experiment("book_sales_arima_modeling")
    
    # Log some test parameters and metrics
    mlflow.log_params({
        "test_param1": "test_value",
        "test_param2": 42,
        "pipeline_type": "zenml_test"
    })
    
    mlflow.log_metrics({
        "test_metric1": 123.45,
        "test_metric2": 67.89
    })
    
    print("✅ MLflow logging completed in ZenML step")
    return "MLflow test completed"

@pipeline
def test_mlflow_pipeline():
    """Test pipeline"""
    result = test_mlflow_step()
    return result

if __name__ == "__main__":
    print("🧪 Testing ZenML MLflow integration...")
    
    # Run the test pipeline
    pipeline_instance = test_mlflow_pipeline()
    pipeline_instance.run()
    
    print("✅ Test pipeline completed")