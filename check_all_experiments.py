import mlflow
from datetime import datetime

def check_all_experiments():
    client = mlflow.MlflowClient()
    
    print("=== ALL MLFLOW EXPERIMENTS ===")
    experiments = client.search_experiments()
    
    for exp in experiments:
        print(f"\n📁 Experiment: {exp.name}")
        print(f"   ID: {exp.experiment_id}")
        print(f"   Lifecycle: {exp.lifecycle_stage}")
        
        # Get runs for this experiment
        runs = client.search_runs([exp.experiment_id], max_results=10)
        print(f"   Runs: {len(runs)}")
        
        for i, run in enumerate(runs[:3]):  # Show first 3 runs
            start_time = datetime.fromtimestamp(run.info.start_time / 1000)
            print(f"     {i+1}. {run.info.run_name or run.info.run_id[:8]}")
            print(f"        Status: {run.info.status}")
            print(f"        Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show some key metrics/params
            if run.data.metrics:
                metrics = list(run.data.metrics.keys())[:3]
                print(f"        Metrics: {metrics}")
            if run.data.params:
                params = list(run.data.params.keys())[:3]
                print(f"        Params: {params}")
        
        if len(runs) > 3:
            print(f"     ... and {len(runs) - 3} more runs")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total experiments: {len(experiments)}")
    total_runs = sum(len(client.search_runs([exp.experiment_id])) for exp in experiments)
    print(f"Total runs across all experiments: {total_runs}")

if __name__ == "__main__":
    check_all_experiments()