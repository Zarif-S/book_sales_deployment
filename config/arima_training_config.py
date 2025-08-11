"""
ARIMA Training Configuration for Book Sales Pipeline

This module provides environment-specific configuration for ARIMA model training
and retraining logic, enabling production-ready optimization without code changes.
"""

import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ARIMATrainingConfig:
    """Configuration for ARIMA model training and retraining logic"""

    # Optimization parameters
    n_trials: int
    patience: int
    min_trials: int
    min_improvement: float

    # Retraining logic parameters
    force_retrain: bool
    max_model_age_days: Optional[int]
    performance_threshold: Optional[float]  # Fraction increase that triggers retraining

    # Environment and storage
    environment: str
    optuna_storage_type: str  # 'sqlite' or 'memory'

    # Model quality thresholds
    min_acceptable_rmse: Optional[float]
    max_acceptable_mape: Optional[float]

    def __init__(self, environment: str = None, **kwargs):
        """Initialize configuration with environment-specific defaults and overrides"""

        # Determine environment
        self.environment = environment or os.getenv('DEPLOYMENT_ENV', 'development').lower()

        # Get base configuration for environment
        base_config = self._get_environment_config()

        # Apply any keyword argument overrides
        for key, value in kwargs.items():
            if hasattr(self, key) or key in base_config:
                base_config[key] = value

        # Set all attributes from final config
        for key, value in base_config.items():
            setattr(self, key, value)

    def _get_environment_config(self) -> Dict[str, Any]:
        """Get configuration based on environment"""

        configs = {
            'development': {
                # Fast development parameters for quick iteration
                'n_trials': int(os.getenv('ARIMA_N_TRIALS', 3)),
                'patience': int(os.getenv('ARIMA_PATIENCE', 1)),
                'min_trials': int(os.getenv('ARIMA_MIN_TRIALS', 2)),
                'min_improvement': float(os.getenv('ARIMA_MIN_IMPROVEMENT', 0.5)),
                'force_retrain': os.getenv('ARIMA_FORCE_RETRAIN', 'true').lower() == 'true',
                'max_model_age_days': None,  # Age check disabled in dev
                'performance_threshold': None,  # Performance check disabled in dev
                'optuna_storage_type': 'sqlite',  # Persist studies for debugging
                'min_acceptable_rmse': None,  # Quality gates disabled in dev
                'max_acceptable_mape': None,
            },

            'testing': {
                # More robust testing parameters
                'n_trials': int(os.getenv('ARIMA_N_TRIALS', 50)),
                'patience': int(os.getenv('ARIMA_PATIENCE', 15)),
                'min_trials': int(os.getenv('ARIMA_MIN_TRIALS', 25)),
                'min_improvement': float(os.getenv('ARIMA_MIN_IMPROVEMENT', 0.1)),
                'force_retrain': os.getenv('ARIMA_FORCE_RETRAIN', 'false').lower() == 'true',
                'max_model_age_days': int(os.getenv('ARIMA_MAX_MODEL_AGE_DAYS', 7)),  # Weekly refresh
                'performance_threshold': float(os.getenv('ARIMA_PERFORMANCE_THRESHOLD', 0.10)),  # 10% degradation
                'optuna_storage_type': 'sqlite',
                'min_acceptable_rmse': float(os.getenv('ARIMA_MIN_RMSE', 100.0)),
                'max_acceptable_mape': float(os.getenv('ARIMA_MAX_MAPE', 50.0)),
            },

            'production': {
                # Production parameters optimized for quality and efficiency
                'n_trials': int(os.getenv('ARIMA_N_TRIALS', 100)),
                'patience': int(os.getenv('ARIMA_PATIENCE', 25)),
                'min_trials': int(os.getenv('ARIMA_MIN_TRIALS', 50)),
                'min_improvement': float(os.getenv('ARIMA_MIN_IMPROVEMENT', 0.05)),
                'force_retrain': os.getenv('ARIMA_FORCE_RETRAIN', 'false').lower() == 'true',
                'max_model_age_days': int(os.getenv('ARIMA_MAX_MODEL_AGE_DAYS', 30)),  # Monthly refresh
                'performance_threshold': float(os.getenv('ARIMA_PERFORMANCE_THRESHOLD', 0.05)),  # 5% degradation
                'optuna_storage_type': 'memory',  # In-memory for production reliability
                'min_acceptable_rmse': float(os.getenv('ARIMA_MIN_RMSE', 80.0)),
                'max_acceptable_mape': float(os.getenv('ARIMA_MAX_MAPE', 30.0)),
            }
        }

        return configs.get(self.environment, configs['development'])

    def get_optuna_storage_config(self) -> Dict[str, Any]:
        """Get Optuna storage configuration for the environment"""

        if self.optuna_storage_type == 'sqlite':
            # Development/testing: Use SQLite for persistence
            storage_dir = Path.home() / "zenml_optuna_storage"
            storage_dir.mkdir(exist_ok=True)
            return {
                'storage_type': 'sqlite',
                'storage_dir': str(storage_dir),
                'load_if_exists': True
            }
        else:
            # Production: Use in-memory for reliability
            return {
                'storage_type': 'memory',
                'storage_dir': None,
                'load_if_exists': False
            }

    def log_configuration(self, logger) -> None:
        """Log current configuration for debugging and monitoring"""

        logger.info(f"🔧 ARIMA Training Configuration ({self.environment} mode)")
        logger.info(f"   Optimization: {self.n_trials} trials, patience={self.patience}, min_improvement={self.min_improvement}")
        logger.info(f"   Retraining: force={self.force_retrain}, max_age={self.max_model_age_days}d, perf_threshold={self.performance_threshold}")
        logger.info(f"   Storage: {self.optuna_storage_type}")

        if self.min_acceptable_rmse or self.max_acceptable_mape:
            logger.info(f"   Quality Gates: RMSE<{self.min_acceptable_rmse}, MAPE<{self.max_acceptable_mape}%")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for serialization"""
        return {
            'environment': self.environment,
            'n_trials': self.n_trials,
            'patience': self.patience,
            'min_trials': self.min_trials,
            'min_improvement': self.min_improvement,
            'force_retrain': self.force_retrain,
            'max_model_age_days': self.max_model_age_days,
            'performance_threshold': self.performance_threshold,
            'optuna_storage_type': self.optuna_storage_type,
            'min_acceptable_rmse': self.min_acceptable_rmse,
            'max_acceptable_mape': self.max_acceptable_mape
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ARIMATrainingConfig':
        """Create configuration from dictionary"""
        environment = config_dict.pop('environment', 'development')
        return cls(environment=environment, **config_dict)

    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> 'ARIMATrainingConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_to_json_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_arima_config(environment: str = None, **overrides) -> ARIMATrainingConfig:
    """
    Convenience function to get ARIMA training configuration.

    Args:
        environment: Environment name ('development', 'testing', 'production')
        **overrides: Any configuration parameters to override

    Returns:
        ARIMATrainingConfig instance

    Example:
        # Development with custom trials
        config = get_arima_config(n_trials=20)

        # Production with forced retraining
        config = get_arima_config(environment='production', force_retrain=True)
    """
    return ARIMATrainingConfig(environment=environment, **overrides)


# Environment-specific factory functions for convenience
def get_development_config(**overrides) -> ARIMATrainingConfig:
    """Get development configuration with optional overrides"""
    return get_arima_config(environment='development', **overrides)


def get_testing_config(**overrides) -> ARIMATrainingConfig:
    """Get testing configuration with optional overrides"""
    return get_arima_config(environment='testing', **overrides)


def get_production_config(**overrides) -> ARIMATrainingConfig:
    """Get production configuration with optional overrides"""
    return get_arima_config(environment='production', **overrides)


if __name__ == "__main__":
    # Example usage and configuration testing
    print("ARIMA Training Configuration Examples:\n")

    configs = [
        ("Development", get_development_config()),
        ("Testing", get_testing_config()),
        ("Production", get_production_config()),
        ("Custom Dev", get_development_config(n_trials=25, force_retrain=False))
    ]

    for name, config in configs:
        print(f"{name} Configuration:")
        print(f"  Environment: {config.environment}")
        print(f"  Trials: {config.n_trials}, Patience: {config.patience}")
        print(f"  Force Retrain: {config.force_retrain}")
        print(f"  Model Age Limit: {config.max_model_age_days} days")
        print(f"  Performance Threshold: {config.performance_threshold}")
        print(f"  Storage: {config.optuna_storage_type}")
        print()
