import mlflow
from datetime import datetime, timedelta

def check_recent_runs():
    client = mlflow.MlflowClient()
    
    # Check for runs in the last hour across all experiments
    one_hour_ago = int((datetime.now() - timedelta(hours=1)).timestamp() * 1000)
    
    print("=== RECENT RUNS (Last Hour) ===")
    experiments = client.search_experiments()
    
    total_recent_runs = 0
    for exp in experiments:
        runs = client.search_runs([exp.experiment_id], max_results=50)
        recent_runs = [run for run in runs if run.info.start_time > one_hour_ago]
        
        if recent_runs:
            print(f"\n📁 {exp.name} (ID: {exp.experiment_id})")
            for run in recent_runs:
                start_time = datetime.fromtimestamp(run.info.start_time / 1000)
                print(f"  🔄 {run.info.run_name or run.info.run_id[:8]}")
                print(f"     Status: {run.info.status}")
                print(f"     Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Show metrics and params for pipeline runs
                if run.data.metrics:
                    print(f"     Metrics: {list(run.data.metrics.keys())[:5]}")
                if run.data.params:
                    print(f"     Params: {list(run.data.params.keys())[:5]}")
                if run.data.tags:
                    zenml_tags = {k: v for k, v in run.data.tags.items() if 'zenml' in k.lower()}
                    if zenml_tags:
                        print(f"     ZenML tags: {list(zenml_tags.keys())[:3]}")
            
            total_recent_runs += len(recent_runs)
    
    if total_recent_runs == 0:
        print("No recent runs found in any experiment!")
        print("This suggests the ZenML pipeline isn't creating MLflow runs at all.")
    
    print(f"\nTotal recent runs: {total_recent_runs}")

if __name__ == "__main__":
    check_recent_runs()