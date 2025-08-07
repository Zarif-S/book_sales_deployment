#!/usr/bin/env python3
"""
Pure MLflow test without ZenML to verify individual runs work
"""

import mlflow
import datetime

def test_pure_mlflow_individual_runs():
    """Test MLflow individual runs without ZenML interference"""
    print("🧪 Testing Pure MLflow Individual Runs")
    print("=" * 50)
    
    # Set experiment with timestamp to ensure fresh start
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"pure_mlflow_individual_test_{timestamp}"
    mlflow.set_experiment(experiment_name)
    print(f"📁 Experiment: {experiment_name}")
    
    # Test books with truly different parameters (avoid any conflicts)
    test_books = [
        {"isbn": "9780722532935", "title": "The Alchemist", "params": {"p": 1, "d": 0, "q": 0, "P": 1, "D": 0, "Q": 1}},
        {"isbn": "9780241003008", "title": "Very Hungry Caterpillar", "params": {"p": 0, "d": 2, "q": 3, "P": 2, "D": 0, "Q": 3}}
    ]
    
    parent_run_id = None
    
    # Create parent run
    with mlflow.start_run(run_name="pure_mlflow_test_parent") as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"🔄 Parent run ID: {parent_run_id}")
        
        # Log parent parameters
        mlflow.log_params({
            "pipeline_type": "pure_mlflow_test",
            "total_books": len(test_books)
        })
        
        # Create individual runs for each book
        for book in test_books:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Add microseconds for uniqueness
            book_run_name = f"book_{book['isbn']}_{timestamp}"
            
            print(f"📖 Creating run for: {book['title']}")
            
            # Create nested run
            with mlflow.start_run(run_name=book_run_name, nested=True) as book_run:
                print(f"  🔄 Book run ID: {book_run.info.run_id}")
                
                # Log book-specific parameters one by one to debug
                book_params = {
                    "book_isbn": book['isbn'],
                    "book_title": book['title'],
                    "model_type": "SARIMA",
                    **book['params']
                }
                
                print(f"  📋 Attempting to log params: {book_params}")
                
                # Log each parameter individually to identify conflicts
                try:
                    for key, value in book_params.items():
                        print(f"    Logging {key} = {value}")
                        mlflow.log_param(key, value)
                        
                except Exception as param_error:
                    print(f"  ❌ Failed to log parameter {key}: {param_error}")
                    raise
                
                # Log mock metrics
                mlflow.log_metrics({
                    "mae": hash(book['isbn']) % 100,
                    "rmse": hash(book['isbn']) % 200,
                    "mape": hash(book['isbn']) % 30
                })
                
                print(f"  ✅ Logged parameters: {book_params}")
        
        # Log parent summary
        mlflow.log_metrics({
            "books_processed": len(test_books)
        })
    
    print(f"\n🎉 Test completed successfully!")
    print(f"📊 Check MLflow UI: http://127.0.0.1:5000")
    print(f"🔍 Look for experiment: {experiment_name}")
    print(f"📋 Expected: 1 parent run + {len(test_books)} nested book runs")

if __name__ == "__main__":
    try:
        test_pure_mlflow_individual_runs()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()