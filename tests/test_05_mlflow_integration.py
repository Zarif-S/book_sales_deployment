#!/usr/bin/env python3
"""
Test script for ZenML + MLflow integration
"""

import os
import sys
from zenml import pipeline, step
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
import mlflow

@step(experiment_tracker="mlflow_tracker")
def test_mlflow_step() -> str:
    """Simple test step to verify MLflow integration works."""
    
    # Log some test parameters
    mlflow.log_params({
        "test_param_1": "test_value_1",
        "test_param_2": 42,
        "model_type": "ARIMA"
    })
    
    # Log some test metrics
    mlflow.log_metrics({
        "test_metric_1": 0.95,
        "test_metric_2": 1.23,
        "mae": 5.67
    })
    
    print("✅ MLflow logging test successful!")
    return "MLflow integration working"

@pipeline
def test_mlflow_pipeline() -> str:
    """Simple test pipeline to verify MLflow integration."""
    result = test_mlflow_step()
    return result

if __name__ == "__main__":
    print("🚀 Testing ZenML + MLflow Integration")
    print("=" * 50)
    
    try:
        # Run the test pipeline
        result = test_mlflow_pipeline()
        print(f"✅ Pipeline completed successfully: {result}")
        
        # Print MLflow info
        print(f"\n📊 MLflow Integration Test Results:")
        print(f"   • MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"   • Active run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'None'}")
        print(f"   • Experiment name: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name if mlflow.active_run() else 'None'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 MLflow integration test completed!")