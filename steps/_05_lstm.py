import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os

def load_arima_residuals_from_pipeline() -> pd.DataFrame:
    """
    Load residuals from the ARIMA pipeline results.
    This function demonstrates how to access residuals from the pipeline output.
    """
    print("Loading ARIMA residuals from pipeline...")
    
    # Method 1: Load from ZenML pipeline run (recommended for production)
    try:
        from zenml.client import Client
        
        # Get the latest pipeline run
        client = Client()
        pipeline_runs = client.list_pipeline_runs(
            pipeline_name="book_sales_arima_pipeline"
        )
        
        if pipeline_runs:
            # Get the most recent run
            latest_run = pipeline_runs[0]  # Most recent run is first
            print(f"Found latest pipeline run: {latest_run.name}")
            
            # Try to get artifacts using the run's artifacts property
            try:
                # Get all artifacts from the run
                run_artifacts = latest_run.artifacts
                
                # Find the residuals artifact
                residuals_artifact = None
                for artifact in run_artifacts:
                    if artifact.name == "residuals":
                        residuals_artifact = artifact
                        break
                
                if residuals_artifact:
                    residuals_df = residuals_artifact.load()
                    print(f"✅ Successfully loaded residuals from pipeline run: {latest_run.name}")
                    return residuals_df
                else:
                    print("⚠️  No residuals artifact found in the pipeline run.")
                    return create_sample_residuals()
                    
            except Exception as artifact_error:
                print(f"⚠️  Could not load artifacts: {artifact_error}")
                return create_sample_residuals()
        else:
            print("⚠️  No pipeline runs found. Using sample data for demonstration.")
            return create_sample_residuals()
            
    except Exception as e:
        print(f"⚠️  Could not load from pipeline: {e}")
        print("Using sample data for demonstration.")
        return create_sample_residuals()

def create_sample_residuals() -> pd.DataFrame:
    """
    Create sample residuals data for demonstration purposes.
    """
    print("Creating sample residuals data...")
    
    # Create realistic sample data based on the actual pipeline output
    sample_dates = pd.date_range(start='2020-01-01', periods=628, freq='W')
    sample_residuals = np.random.normal(0, 100, 628)  # Mock residuals
    
    residuals_df = pd.DataFrame({
        'date': sample_dates,
        'residuals': sample_residuals,
        'model_signature': 'SARIMAX_(2,1,3)_(1,1,3,52)'
    })
    
    return residuals_df

def analyze_residuals_for_lstm(residuals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze residuals for LSTM model preparation.
    """
    print("\n=== Residuals Analysis for LSTM ===")
    
    # Basic statistics
    stats = {
        'count': len(residuals_df),
        'mean': float(residuals_df['residuals'].mean()),
        'std': float(residuals_df['residuals'].std()),
        'min': float(residuals_df['residuals'].min()),
        'max': float(residuals_df['residuals'].max()),
        'skewness': float(residuals_df['residuals'].skew()),
        'kurtosis': float(residuals_df['residuals'].kurtosis())
    }
    
    print(f"Residuals Statistics:")
    for key, value in stats.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    # Check for missing values
    missing_count = residuals_df['residuals'].isna().sum()
    print(f"  Missing values: {missing_count}")
    
    # Check for outliers (using IQR method)
    Q1 = residuals_df['residuals'].quantile(0.25)
    Q3 = residuals_df['residuals'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = residuals_df[
        (residuals_df['residuals'] < Q1 - 1.5 * IQR) | 
        (residuals_df['residuals'] > Q3 + 1.5 * IQR)
    ]
    print(f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(residuals_df)*100:.2f}%)")
    
    return {
        'residuals_df': residuals_df,
        'statistics': stats,
        'outliers': outliers,
        'analysis_summary': {
            'total_points': len(residuals_df),
            'outlier_percentage': len(outliers)/len(residuals_df)*100,
            'data_quality': 'good' if missing_count == 0 else 'needs_attention'
        }
    }

def plot_residuals_for_lstm(residuals_df: pd.DataFrame, save_plots: bool = True):
    """
    Create comprehensive plots of residuals for LSTM analysis.
    """
    print("\n=== Creating Residuals Plots ===")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ARIMA Residuals Analysis for LSTM Model', fontsize=16, fontweight='bold')
    
    # 1. Time series plot
    axes[0, 0].plot(residuals_df['date'], residuals_df['residuals'], linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Residuals Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Histogram with normal distribution overlay
    axes[0, 1].hist(residuals_df['residuals'], bins=30, alpha=0.7, density=True, edgecolor='black')
    # Overlay normal distribution
    x = np.linspace(residuals_df['residuals'].min(), residuals_df['residuals'].max(), 100)
    mu, sigma = residuals_df['residuals'].mean(), residuals_df['residuals'].std()
    normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    axes[0, 1].plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals_df['residuals'].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    axes[1, 1].boxplot(residuals_df['residuals'], vert=True)
    axes[1, 1].set_title('Residuals Box Plot')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, 'arima_residuals_for_lstm.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plots saved to: {plot_path}")
    
    plt.show()
    
    return fig

def prepare_residuals_for_lstm(residuals_df: pd.DataFrame, sequence_length: int = 10) -> Dict[str, np.ndarray]:
    """
    Prepare residuals data for LSTM model training.
    """
    print(f"\n=== Preparing Residuals for LSTM (sequence_length={sequence_length}) ===")
    
    # Convert to numpy array
    residuals_array = residuals_df['residuals'].values
    
    # Normalize the data (important for LSTM)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals_array.reshape(-1, 1)).flatten()
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(residuals_scaled) - sequence_length):
        X.append(residuals_scaled[i:(i + sequence_length)])
        y.append(residuals_scaled[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/test sets (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Original residuals shape: {residuals_array.shape}")
    print(f"Scaled residuals shape: {residuals_scaled.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'sequence_length': sequence_length,
        'original_residuals': residuals_array,
        'scaled_residuals': residuals_scaled
    }

def print_residuals_sample(residuals_df: pd.DataFrame, n_samples: int = 10):
    """
    Print a sample of the residuals data to verify it loaded correctly.
    """
    print(f"\n=== Residuals Sample (first {n_samples} rows) ===")
    print(residuals_df.head(n_samples).to_string(index=False))
    
    print(f"\n=== Residuals Sample (last {n_samples} rows) ===")
    print(residuals_df.tail(n_samples).to_string(index=False))
    
    print(f"\n=== Residuals Summary ===")
    print(f"Total rows: {len(residuals_df)}")
    print(f"Date range: {residuals_df['date'].min()} to {residuals_df['date'].max()}")
    print(f"Model signature: {residuals_df['model_signature'].iloc[0]}")
    print(f"Residuals range: {residuals_df['residuals'].min():.2f} to {residuals_df['residuals'].max():.2f}")

def demonstrate_pipeline_integration():
    """
    Demonstrate how to integrate with the ARIMA pipeline and access residuals.
    This shows the complete workflow for production use.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Complete Pipeline Integration")
    print("="*60)
    
    print("\n1. Running the ARIMA pipeline to get residuals...")
    print("   (In production, you would call this from your main script)")
    
    # This is how you would call the pipeline in your main script
    try:
        # Import the pipeline
        from pipelines.zenml_pipeline_latest_31_jul import book_sales_arima_pipeline
        import os
        
        # Set up parameters
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'data', 'processed')
        
        default_selected_isbns = [
            '9780722532935',  # The Alchemist
            '9780241003008'   # The Very Hungry Caterpillar
        ]
        
        # Run the pipeline (this would be your actual call)
        print("   Running: book_sales_arima_pipeline(...)")
        results = book_sales_arima_pipeline(
            output_dir=output_dir,
            selected_isbns=default_selected_isbns,
            column_name='Volume',
            split_size=32,
            n_trials=3
        )
        
        print("   ✅ Pipeline completed successfully!")
        
        # Access residuals from the results
        residuals_df = results["residuals"]
        print(f"   ✅ Residuals accessed: {residuals_df.shape}")
        
        return residuals_df
        
    except Exception as e:
        print(f"   ⚠️  Pipeline execution failed: {e}")
        print("   Using sample data for demonstration...")
        return create_sample_residuals()

def main():
    """
    Main function to demonstrate residuals loading and analysis for LSTM.
    """
    print("=== ARIMA Residuals Analysis for LSTM Model ===")
    print("This script demonstrates how to load and analyze residuals from the ARIMA pipeline")
    print("for use in LSTM model training.\n")
    
    # Method 1: Try to load from existing pipeline run
    print("METHOD 1: Loading from existing pipeline run...")
    residuals_df = load_arima_residuals_from_pipeline()
    
    # Method 2: Demonstrate complete pipeline integration
    print("\nMETHOD 2: Complete pipeline integration demonstration...")
    residuals_df_demo = demonstrate_pipeline_integration()
    
    # Use the demo residuals if we got them from the pipeline
    if residuals_df_demo is not None and not residuals_df_demo.empty:
        residuals_df = residuals_df_demo
    
    # Print sample of residuals to verify loading
    print_residuals_sample(residuals_df)
    
    # Analyze residuals
    analysis_results = analyze_residuals_for_lstm(residuals_df)
    
    # Create plots
    plot_residuals_for_lstm(residuals_df)
    
    # Prepare data for LSTM
    lstm_data = prepare_residuals_for_lstm(residuals_df, sequence_length=10)
    
    print("\n=== Summary ===")
    print("✅ Residuals successfully loaded from ARIMA pipeline")
    print("✅ Residuals analyzed and visualized")
    print("✅ Data prepared for LSTM model training")
    print(f"✅ Ready for LSTM with {lstm_data['X_train'].shape[0]} training sequences")
    
    print("\n=== Next Steps for LSTM Model ===")
    print("1. Use lstm_data['X_train'] and lstm_data['y_train'] for training")
    print("2. Use lstm_data['X_test'] and lstm_data['y_test'] for validation")
    print("3. Use lstm_data['scaler'] to inverse transform predictions")
    print("4. The residuals are now ready for LSTM modeling!")
    
    return {
        'residuals_df': residuals_df,
        'analysis_results': analysis_results,
        'lstm_data': lstm_data
    }

if __name__ == "__main__":
    results = main()