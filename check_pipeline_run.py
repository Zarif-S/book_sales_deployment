import mlflow
from datetime import datetime

def check_pipeline_run():
    client = mlflow.MlflowClient()
    
    # Check the book_sales_arima_modeling_pipeline experiment
    try:
        exp = client.get_experiment_by_name('book_sales_arima_modeling_pipeline')
        if exp:
            print(f"📁 Found experiment: {exp.name}")
            runs = client.search_runs([exp.experiment_id])
            
            for run in runs:
                start_time = datetime.fromtimestamp(run.info.start_time / 1000)
                print(f"\n🔄 Run: {run.info.run_name}")
                print(f"   Status: {run.info.status}")
                print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                print("\n📊 Parameters:")
                for key, value in run.data.params.items():
                    print(f"   {key}: {value}")
                    
                print("\n📈 Metrics:")
                for key, value in run.data.metrics.items():
                    print(f"   {key}: {value}")
                    
                print("\n🏷️  Tags:")
                for key, value in run.data.tags.items():
                    if 'book_' in key:  # Show book-specific tags
                        print(f"   {key}: {value}")
        else:
            print("❌ Experiment 'book_sales_arima_modeling_pipeline' not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_pipeline_run()