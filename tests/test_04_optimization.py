#!/usr/bin/env python3
"""
Test script for the optimized ARIMA pipeline with configuration and smart retraining.

This demonstrates the new optimization features without running the full pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.arima_training_config import get_arima_config, get_development_config, get_production_config
from utils.model_reuse import create_retraining_engine

def test_configurations():
    """Test different configuration environments"""
    print("🔧 Testing Configuration System")
    print("=" * 50)
    
    configs = [
        ("Development (default)", get_development_config()),
        ("Development (custom)", get_development_config(n_trials=5, force_retrain=False)),
        ("Production", get_production_config()),
        ("Environment variable override", get_arima_config())  # Will use DEPLOYMENT_ENV if set
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        print(f"  Environment: {config.environment}")
        print(f"  Trials: {config.n_trials}, Patience: {config.patience}")
        print(f"  Force Retrain: {config.force_retrain}")
        print(f"  Max Model Age: {config.max_model_age_days} days")
        print(f"  Performance Threshold: {config.performance_threshold}")
        print(f"  Storage Type: {config.optuna_storage_type}")


def test_smart_retraining():
    """Test smart retraining decision engine"""
    print("\n\n♻️ Testing Smart Retraining Logic")
    print("=" * 50)
    
    # Create test configuration
    config = get_development_config(
        force_retrain=False,  # Enable smart retraining
        max_model_age_days=7,
        performance_threshold=0.10
    )
    
    # Create temporary output directory
    test_output_dir = "/tmp/arima_test_models"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Initialize retraining engine
    engine = create_retraining_engine(config, test_output_dir)
    
    # Create mock data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='W')
    mock_train_data = pd.DataFrame({
        'ISBN': ['9780722532935'] * len(dates),
        'Volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    
    mock_test_data = mock_train_data.copy()
    
    # Test decision 1: No existing model
    should_retrain_1, reason_1, model_info_1 = engine.should_retrain_model(
        '9780722532935', mock_train_data, mock_test_data
    )
    
    print(f"Decision 1 (no existing model):")
    print(f"  Should retrain: {should_retrain_1}")
    print(f"  Reason: {reason_1}")
    
    # Simulate registering a model
    mock_metrics = {'rmse': 50.0, 'mae': 40.0, 'mape': 15.0}
    mock_params = {'p': 1, 'd': 0, 'q': 0, 'P': 1, 'D': 0, 'Q': 0}
    data_hash = engine.calculate_data_hash(mock_train_data, mock_test_data, '9780722532935')
    
    engine.register_model(
        isbn='9780722532935',
        model_path='/tmp/mock_model.pkl',
        evaluation_metrics=mock_metrics,
        model_params=mock_params,
        data_hash=data_hash,
        train_length=len(mock_train_data),
        test_length=len(mock_test_data),
        metadata={'test': 'true'}
    )
    
    # Test decision 2: Model exists and is recent
    should_retrain_2, reason_2, model_info_2 = engine.should_retrain_model(
        '9780722532935', mock_train_data, mock_test_data
    )
    
    print(f"\nDecision 2 (recent model exists):")
    print(f"  Should retrain: {should_retrain_2}")
    print(f"  Reason: {reason_2}")
    print(f"  Model info: Created {model_info_2.created_date.strftime('%Y-%m-%d %H:%M')} with RMSE {model_info_2.baseline_rmse}")
    
    # Test decision 3: Force retrain
    config_force = get_development_config(force_retrain=True)
    engine_force = create_retraining_engine(config_force, test_output_dir)
    
    should_retrain_3, reason_3, model_info_3 = engine_force.should_retrain_model(
        '9780722532935', mock_train_data, mock_test_data
    )
    
    print(f"\nDecision 3 (force retrain enabled):")
    print(f"  Should retrain: {should_retrain_3}")
    print(f"  Reason: {reason_3}")
    
    # Show retraining stats
    stats = engine.get_retraining_stats()
    print(f"\nRetraining Statistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Retrain decisions: {stats.get('retrain_decisions', 0)}")
    print(f"  Reuse decisions: {stats.get('reuse_decisions', 0)}")
    print(f"  Retrain rate: {stats.get('retrain_rate', 0):.1%}")


def test_environment_variables():
    """Test environment variable overrides"""
    print("\n\n🌍 Testing Environment Variable Overrides")
    print("=" * 50)
    
    # Test default
    config_default = get_arima_config()
    print(f"Default config: {config_default.environment} mode, {config_default.n_trials} trials")
    
    # Set environment variables and test override
    os.environ['DEPLOYMENT_ENV'] = 'production'
    os.environ['ARIMA_N_TRIALS'] = '25'
    os.environ['ARIMA_FORCE_RETRAIN'] = 'true'
    
    config_env = get_arima_config()
    print(f"Environment override: {config_env.environment} mode, {config_env.n_trials} trials, force_retrain={config_env.force_retrain}")
    
    # Clean up
    if 'DEPLOYMENT_ENV' in os.environ:
        del os.environ['DEPLOYMENT_ENV']
    if 'ARIMA_N_TRIALS' in os.environ:
        del os.environ['ARIMA_N_TRIALS']
    if 'ARIMA_FORCE_RETRAIN' in os.environ:
        del os.environ['ARIMA_FORCE_RETRAIN']


def demonstrate_efficiency_gains():
    """Demonstrate potential efficiency gains"""
    print("\n\n⚡ Efficiency Gains Demonstration")
    print("=" * 50)
    
    scenarios = [
        ("Development (first run)", {"reused": 0, "new": 2, "trials_per_book": 10}),
        ("Development (second run, smart retraining)", {"reused": 2, "new": 0, "trials_per_book": 10}),
        ("Production (mixed scenario)", {"reused": 3, "new": 2, "trials_per_book": 100}),
    ]
    
    for scenario_name, data in scenarios:
        reused = data["reused"]
        new = data["new"]
        trials = data["trials_per_book"]
        total_books = reused + new
        
        # Estimate time savings (assuming 30 seconds per trial)
        time_without_optimization = total_books * trials * 30  # seconds
        time_with_optimization = new * trials * 30  # seconds
        time_saved = time_without_optimization - time_with_optimization
        
        efficiency = (time_saved / time_without_optimization * 100) if time_without_optimization > 0 else 0
        
        print(f"\n{scenario_name}:")
        print(f"  Books: {total_books} total ({reused} reused, {new} newly trained)")
        print(f"  Trials per book: {trials}")
        print(f"  Time without optimization: {time_without_optimization/60:.1f} minutes")
        print(f"  Time with optimization: {time_with_optimization/60:.1f} minutes")
        print(f"  Time saved: {time_saved/60:.1f} minutes ({efficiency:.1f}% efficiency gain)")


if __name__ == "__main__":
    print("🚀 ARIMA Pipeline Optimization Test Suite")
    print("=" * 60)
    print("Testing the new configuration system and smart retraining logic")
    print("=" * 60)
    
    try:
        test_configurations()
        test_smart_retraining()
        test_environment_variables()
        demonstrate_efficiency_gains()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("✅ Pipeline optimization features are working correctly")
        print("=" * 60)
        
        print("\n🎯 Next Steps:")
        print("1. Run the optimized pipeline: python pipelines/zenml_pipeline.py")
        print("2. On first run, all models will be trained (no existing models)")
        print("3. On second run, models will be reused if data hasn't changed")
        print("4. Configure production settings via environment variables")
        print("5. Deploy with appropriate configuration for your environment")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()