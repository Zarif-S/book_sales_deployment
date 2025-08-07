#!/usr/bin/env python3
"""
Simple MLflow test with one book only to isolate the issue
"""

import mlflow
import datetime

def simple_single_book_test():
    """Test with single book to avoid any conflicts"""
    print("🧪 Simple Single Book MLflow Test")
    print("=" * 40)
    
    # Fresh experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"single_book_test_{timestamp}"
    mlflow.set_experiment(experiment_name)
    print(f"📁 Experiment: {experiment_name}")
    
    # Single book test with unique parameter names to avoid any caching
    book = {
        "isbn": "9780722532935", 
        "title": "The Alchemist",
        "params": {"sarima_p": 1, "sarima_d": 0, "sarima_q": 0, "sarima_P": 2, "sarima_D": 0, "sarima_Q": 1}
    }
    
    with mlflow.start_run(run_name=f"single_book_{timestamp}") as run:
        print(f"🔄 Run ID: {run.info.run_id}")
        
        # Log parameters one by one
        print(f"📋 Logging parameters for: {book['title']}")
        
        # Log basic parameters first
        mlflow.log_param("book_isbn", book['isbn'])
        mlflow.log_param("book_title", book['title'])
        mlflow.log_param("model_type", "SARIMA")
        
        # Log SARIMA parameters
        for key, value in book['params'].items():
            print(f"  Logging {key} = {value}")
            mlflow.log_param(key, value)
        
        # Log metrics
        mlflow.log_metrics({
            "mae": 150.5,
            "rmse": 200.3,
            "mape": 12.1
        })
        
        print(f"✅ Successfully logged all parameters and metrics")
    
    print(f"🎉 Single book test completed!")
    return experiment_name

if __name__ == "__main__":
    try:
        experiment = simple_single_book_test()
        print(f"📊 Check MLflow UI: http://127.0.0.1:5000")
        print(f"🔍 Look for experiment: {experiment}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()