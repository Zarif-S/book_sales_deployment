#!/usr/bin/env python3
"""
Test MLflow integration with individual book runs
Tests the new nested run architecture for book-specific model tracking
"""

import mlflow
import pandas as pd
import datetime
from zenml import pipeline, step
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def test_individual_book_runs_step() -> str:
    """Test individual MLflow runs for each book (nested architecture)"""
    
    # Set experiment name
    experiment_name = "test_individual_book_runs"
    mlflow.set_experiment(experiment_name)
    logger.info(f"🧪 MLflow experiment set to: {experiment_name}")
    
    # Log parent run parameters
    mlflow.log_params({
        "pipeline_type": "test_individual_runs",
        "total_books": 2,
        "test_purpose": "verify_nested_runs"
    })
    
    # Test books
    test_books = [
        {"isbn": "9780722532935", "title": "The Alchemist"},
        {"isbn": "9780241003008", "title": "Very Hungry Caterpillar"}
    ]
    
    # Create individual runs for each book
    for i, book in enumerate(test_books):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        book_run_name = f"test_arima_{book['isbn']}_{timestamp}_{i}"  # Add index for uniqueness
        
        logger.info(f"📖 Creating individual run for: {book['title']} ({book['isbn']})")
        
        # Create nested run for this book with unique parameters per book
        with mlflow.start_run(run_name=book_run_name, nested=True):
            # Log book-specific parameters (different per book to avoid conflicts)
            book_params = {
                "book_isbn": book['isbn'],
                "book_title": book['title'],
                "model_type": "SARIMA",
                "seasonal_period": 52
            }
            
            # Add different SARIMA parameters per book to test separation
            if book['isbn'] == "9780722532935":  # Alchemist
                book_params.update({"p": 1, "d": 0, "q": 0, "P": 2, "D": 0, "Q": 1})
            else:  # Very Hungry Caterpillar  
                book_params.update({"p": 0, "d": 2, "q": 3, "P": 2, "D": 0, "Q": 3})
                
            mlflow.log_params(book_params)
            
            # Log mock evaluation metrics
            mlflow.log_metrics({
                "mae": 100.0 + hash(book['isbn']) % 100,  # Mock but consistent per book
                "rmse": 200.0 + hash(book['isbn']) % 150,
                "mape": 10.0 + hash(book['isbn']) % 20
            })
            
            logger.info(f"✅ Created nested MLflow run: {book_run_name}")
    
    # Log parent run summary
    mlflow.log_metrics({
        "books_processed": len(test_books),
        "runs_created": len(test_books)
    })
    
    logger.info("✅ Individual book runs test completed successfully")
    return f"Successfully created {len(test_books)} individual MLflow runs"

@pipeline
def test_individual_runs_pipeline() -> str:
    """Pipeline to test individual MLflow runs architecture"""
    logger.info("🚀 Starting individual MLflow runs test pipeline")
    
    result = test_individual_book_runs_step()
    
    logger.info("✅ Individual runs test pipeline completed")
    return result

if __name__ == "__main__":
    print("🧪 Testing Individual MLflow Runs Architecture")
    print("=" * 60)
    
    try:
        result = test_individual_runs_pipeline()
        print(f"✅ Test completed successfully: {result}")
        
        print(f"\n📊 MLflow Integration Test Results:")
        print(f"   • MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"   • Test experiment: test_individual_book_runs")
        print(f"\n📋 Check Results:")
        print(f"   • ZenML Dashboard: http://127.0.0.1:8237")
        print(f"   • MLflow UI: http://127.0.0.1:5000")
        print(f"   • Look for 'test_individual_book_runs' experiment with nested runs")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Expected Results:")
    print("   ✓ One parent run with pipeline-level metrics")
    print("   ✓ Two nested runs (one per book) with individual params/metrics")
    print("   ✓ Clean separation of book-specific data")