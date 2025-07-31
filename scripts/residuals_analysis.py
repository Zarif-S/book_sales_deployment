import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os
from scipy import stats
import statsmodels.graphics.tsaplots as smgraphics # added temporarily


def analyze_residuals_for_lstm(residuals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze residuals for LSTM model preparation.
    """
    print("\n=== Residuals Analysis for LSTM ===")
    
    # Basic statistics
    stats_dict = {
        'count': len(residuals_df),
        'mean': float(residuals_df['residuals'].mean()),
        'std': float(residuals_df['residuals'].std()),
        'min': float(residuals_df['residuals'].min()),
        'max': float(residuals_df['residuals'].max()),
        'skewness': float(residuals_df['residuals'].skew()),
        'kurtosis': float(residuals_df['residuals'].kurtosis())
    }
    
    print(f"Residuals Statistics:")
    for key, value in stats_dict.items():
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
        'statistics': stats_dict,
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
    stats.probplot(residuals_df['residuals'].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    axes[1, 1].boxplot(residuals_df['residuals'], vert=True)
    axes[1, 1].set_title('Residuals Box Plot')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add simple test plots, temporarily
    print("\n=== Creating Simple Test Plots ===")
    
    # Create a new figure for the simple plots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Simple Residuals Test Plots', fontsize=14, fontweight='bold')
    
    # Plot the histogram of the residuals
    ax1.hist(residuals_df['residuals'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Histogram of Residuals')
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot the ACF of the residuals
    smgraphics.plot_acf(residuals_df['residuals'].dropna(), ax=ax2, lags=40)
    ax2.set_title('ACF of Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if save_plots:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, 'arima_residuals_for_lstm.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plots saved to: {plot_path}")
        
        # Save the simple test plots as well, temporarily
        simple_plot_path = os.path.join(plots_dir, 'simple_residuals_test.png')
        fig2.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
        print(f"Simple test plots saved to: {simple_plot_path}")
    
    return fig


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


def run_complete_residuals_analysis(residuals_df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
    """
    Convenient wrapper function that runs both analysis and plotting.
    """
    print("\n=== Running Complete Residuals Analysis ===")
    
    # Run analysis
    analysis_results = analyze_residuals_for_lstm(residuals_df)
    
    # Create plots
    plot_figure = plot_residuals_for_lstm(residuals_df, save_plots=save_plots)
    
    # Add plot figure to results
    analysis_results['plot_figure'] = plot_figure
    
    return analysis_results


def create_sample_residuals_for_testing() -> pd.DataFrame:
    """
    Create sample residuals data for standalone testing of analysis functions.
    """
    print("Creating sample residuals data for testing...")
    
    # Create realistic sample data
    sample_dates = pd.date_range(start='2020-01-01', periods=200, freq='W')
    sample_residuals = np.random.normal(0, 50, 200)  # Mock residuals
    
    residuals_df = pd.DataFrame({
        'date': sample_dates,
        'residuals': sample_residuals,
        'model_signature': 'SARIMAX_(2,1,3)_(1,1,3,52)_TEST'
    })
    
    return residuals_df


def load_real_residuals_from_csv(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load real residuals from CSV file saved by the ARIMA pipeline.
    """
    import os
    
    # Look for residuals CSV in the data directory
    residuals_csv_path = os.path.join(data_dir, "arima_residuals.csv")
    
    if not os.path.exists(residuals_csv_path):
        print(f"⚠️  Residuals CSV not found at: {residuals_csv_path}")
        print("📋 Checking for alternative locations...")
        
        # Check other possible locations
        alternative_paths = [
            "data/arima_residuals.csv",
            "arima_residuals.csv",
            os.path.join(data_dir, "..", "arima_residuals.csv")
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                residuals_csv_path = alt_path
                print(f"✅ Found residuals CSV at: {residuals_csv_path}")
                break
        else:
            raise FileNotFoundError(f"Residuals CSV not found. Expected at: {residuals_csv_path}")
    
    residuals_df = pd.read_csv(residuals_csv_path)
    
    # Convert date column to datetime if it's not already
    if 'date' in residuals_df.columns:
        residuals_df['date'] = pd.to_datetime(residuals_df['date'])
    
    return residuals_df


if __name__ == "__main__":
    """
    Run residuals analysis on real data from ARIMA pipeline.
    """
    print("=== Real Residuals Analysis ===")
    print("Analyzing residuals from ARIMA pipeline results.\n")
    
    try:
        # Try to load real residuals data
        print("📋 Loading real residuals from CSV...")
        residuals_df = load_real_residuals_from_csv()
        print(f"✅ Loaded {len(residuals_df)} real residuals data points")
        print(f"📅 Date range: {residuals_df['date'].min()} to {residuals_df['date'].max()}")
        print(f"🔧 Model: {residuals_df['model_signature'].iloc[0]}")
        
    except Exception as e:
        print(f"⚠️  Could not load real residuals: {e}")
        print("🧪 Falling back to sample data for demonstration...")
        residuals_df = create_sample_residuals_for_testing()
    
    # Print sample
    print_residuals_sample(residuals_df, n_samples=5)
    
    # Run complete analysis
    test_results = run_complete_residuals_analysis(residuals_df, save_plots=True)
    
    print("\n=== Analysis Complete ===")
    print(f"✅ Analyzed {test_results['analysis_summary']['total_points']} data points")
    print(f"✅ Data quality: {test_results['analysis_summary']['data_quality']}")
    print(f"✅ Outlier percentage: {test_results['analysis_summary']['outlier_percentage']:.2f}%")
    print("✅ Residuals analysis completed successfully!")
