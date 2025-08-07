from zenml.client import Client

try:
    client = Client()
    active_stack = client.active_stack
    tracker = active_stack.experiment_tracker

    # Access the tracking URI through the component's configuration
    tracking_uri = tracker.config.tracking_uri

    print("\n✅ Your MLflow Tracking URI is:")
    print(tracking_uri)
    print("\n")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your ZenML repository is initialized and the 'mlflow_stack' is active.")
