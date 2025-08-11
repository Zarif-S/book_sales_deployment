"""
Vertex AI deployment script for book sales ARIMA modeling pipeline.
This script runs the pipeline on Google Cloud Vertex AI with minimal cost for testing.
"""

import os
from config.arima_training_config import get_arima_config
from pipelines.zenml_pipeline import book_sales_arima_modeling_pipeline

def deploy_to_vertex():
    """Deploy the pipeline to Vertex AI with development configuration for cost testing."""
    
    # Set up output directory for cloud deployment
    output_dir = "gs://book-sales-deployment-artifacts/processed"
    
    # Use development configuration for cost-effective testing
    config = get_arima_config(
        environment='development',  # Dev mode - minimal resources
        n_trials=3,  # Only 3 trials to minimize cost
        force_retrain=True  # Skip retraining logic for first test
    )
    
    print(f"🧪 Testing Vertex AI deployment with minimal cost configuration:")
    print(f"   Environment: {config.environment}")
    print(f"   Trials per book: {config.n_trials}")
    print(f"   Books to test: 1 (The Alchemist)")
    print(f"   Output directory: {output_dir}")
    print(f"   Expected cost: Very low (development mode)")
    
    # Run pipeline with single book for cost testing
    results = book_sales_arima_modeling_pipeline(
        output_dir=output_dir,
        selected_isbns=['9780722532935'],  # Just The Alchemist for testing
        column_name='Volume',
        split_size=32,
        use_seasonality_filter=False,  # Use specific ISBN
        max_seasonal_books=1,
        train_arima=True,
        config=config  # Development configuration
    )
    
    print("✅ Single-book test deployment to Vertex AI completed!")
    print("💰 Check Google Cloud Console for cost breakdown")
    return results

if __name__ == "__main__":
    deploy_to_vertex()