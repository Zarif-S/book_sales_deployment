import mlflow

def check_experiment():
    client = mlflow.MlflowClient()
    try:
        exp = client.get_experiment_by_name('book_sales_arima_modeling')
        if exp:
            print(f'Found experiment: {exp.name} (ID: {exp.experiment_id})')
            runs = client.search_runs([exp.experiment_id])
            print(f'Runs in experiment: {len(runs)}')
            if runs:
                for run in runs:
                    print(f'  - Run: {run.info.run_name or run.info.run_id} (Status: {run.info.status})')
        else:
            print('Experiment "book_sales_arima_modeling" not found')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    check_experiment()