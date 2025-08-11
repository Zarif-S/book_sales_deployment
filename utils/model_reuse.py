"""
Model Reuse and Smart Retraining Logic for ARIMA Pipeline

This module provides intelligent decision-making for model retraining,
avoiding unnecessary computation while maintaining model quality.
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
import mlflow
import mlflow.statsmodels
from zenml.logger import get_logger

from config.arima_training_config import ARIMATrainingConfig

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about an existing model"""
    isbn: str
    model_path: str
    mlflow_run_id: Optional[str]
    mlflow_model_version: Optional[str]
    created_date: datetime
    baseline_rmse: float
    baseline_mae: float
    baseline_mape: float
    model_params: Dict[str, Any]
    data_hash: str
    training_data_length: int
    test_data_length: int
    metadata: Dict[str, Any]


class ModelRetrainDecisionEngine:
    """Decides whether models need retraining based on configurable criteria"""

    def __init__(self, config: ARIMATrainingConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.model_registry_path = Path(output_dir) / "model_registry.json"
        self.logger = get_logger(__name__)

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load or initialize model registry
        self.model_registry = self._load_model_registry()

    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from disk, create if doesn't exist"""
        if self.model_registry_path.exists():
            try:
                with open(self.model_registry_path, 'r') as f:
                    registry = json.load(f)
                self.logger.info(f"Loaded model registry with {len(registry.get('models', {}))} models")
                return registry
            except Exception as e:
                self.logger.warning(f"Failed to load model registry: {e}, creating new one")

        # Create new registry
        return {
            'created_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'models': {},
            'retraining_history': []
        }

    def _save_model_registry(self) -> None:
        """Save model registry to disk"""
        self.model_registry['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.model_registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
            self.logger.debug(f"Model registry saved to {self.model_registry_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")

    def calculate_data_hash(self, train_data: pd.DataFrame, test_data: pd.DataFrame, isbn: str) -> str:
        """Calculate hash of training data to detect changes"""
        try:
            # Filter data for specific book
            book_train = train_data[train_data['ISBN'] == isbn] if 'ISBN' in train_data.columns else train_data
            book_test = test_data[test_data['ISBN'] == isbn] if 'ISBN' in test_data.columns else test_data

            # Create hash from data shape, values, and date range
            train_info = {
                'shape': book_train.shape,
                'columns': list(book_train.columns),
                'date_range': [str(book_train.index.min()), str(book_train.index.max())],
                'volume_sum': float(book_train.get('Volume', pd.Series([0])).sum()),
                'volume_mean': float(book_train.get('Volume', pd.Series([0])).mean())
            }

            test_info = {
                'shape': book_test.shape,
                'date_range': [str(book_test.index.min()), str(book_test.index.max())],
                'volume_sum': float(book_test.get('Volume', pd.Series([0])).sum())
            }

            combined_info = json.dumps({
                'train': train_info,
                'test': test_info
            }, sort_keys=True, default=str)

            return hashlib.md5(combined_info.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Failed to calculate data hash for {isbn}: {e}")
            # Fallback to timestamp-based hash
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()

    def get_latest_model_info(self, isbn: str) -> Optional[ModelInfo]:
        """Get information about the latest model for a book"""
        if isbn not in self.model_registry['models']:
            return None

        model_data = self.model_registry['models'][isbn]

        try:
            return ModelInfo(
                isbn=isbn,
                model_path=model_data['model_path'],
                mlflow_run_id=model_data.get('mlflow_run_id'),
                mlflow_model_version=model_data.get('mlflow_model_version'),
                created_date=datetime.fromisoformat(model_data['created_date']),
                baseline_rmse=float(model_data['baseline_rmse']),
                baseline_mae=float(model_data['baseline_mae']),
                baseline_mape=float(model_data['baseline_mape']),
                model_params=model_data['model_params'],
                data_hash=model_data['data_hash'],
                training_data_length=model_data['training_data_length'],
                test_data_length=model_data['test_data_length'],
                metadata=model_data.get('metadata', {})
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse model info for {isbn}: {e}")
            return None

    def register_model(
        self,
        isbn: str,
        model_path: str,
        evaluation_metrics: Dict[str, float],
        model_params: Dict[str, Any],
        data_hash: str,
        train_length: int,
        test_length: int,
        mlflow_run_id: Optional[str] = None,
        mlflow_model_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new model in the registry"""

        model_entry = {
            'isbn': isbn,
            'model_path': model_path,
            'mlflow_run_id': mlflow_run_id,
            'mlflow_model_version': mlflow_model_version,
            'created_date': datetime.now().isoformat(),
            'baseline_rmse': evaluation_metrics.get('rmse', 0.0),
            'baseline_mae': evaluation_metrics.get('mae', 0.0),
            'baseline_mape': evaluation_metrics.get('mape', 0.0),
            'model_params': model_params,
            'data_hash': data_hash,
            'training_data_length': train_length,
            'test_data_length': test_length,
            'metadata': metadata or {}
        }

        self.model_registry['models'][isbn] = model_entry
        self._save_model_registry()

        self.logger.info(f"Registered model for {isbn}: RMSE={evaluation_metrics.get('rmse', 0):.2f}, "
                        f"MAPE={evaluation_metrics.get('mape', 0):.1f}%")

    def validate_existing_model(self, isbn: str, model_info: ModelInfo,
                              current_test_data: pd.DataFrame) -> Optional[float]:
        """Validate existing model performance on current data"""
        try:
            # Filter test data for this book
            book_test_data = current_test_data[current_test_data['ISBN'] == isbn].copy()
            if book_test_data.empty:
                self.logger.warning(f"No current test data found for {isbn}")
                return None

            # Load the existing model
            if model_info.mlflow_run_id and model_info.mlflow_model_version:
                try:
                    # Try to load from MLflow first
                    model_uri = f"models:/arima_book_{isbn}/{model_info.mlflow_model_version}"
                    model = mlflow.statsmodels.load_model(model_uri)
                    self.logger.debug(f"Loaded model from MLflow: {model_uri}")
                except Exception as mlflow_error:
                    self.logger.warning(f"MLflow model load failed: {mlflow_error}, trying file path")
                    model = mlflow.statsmodels.load_model(model_info.model_path)
            else:
                # Load from file path
                model = mlflow.statsmodels.load_model(model_info.model_path)

            # Prepare test data
            book_test_clean = book_test_data.drop(columns=['ISBN', 'Title', 'End Date'], errors='ignore')
            if not pd.api.types.is_datetime64_any_dtype(book_test_clean.index):
                if 'End Date' in book_test_data.columns:
                    book_test_clean.set_index(pd.to_datetime(book_test_data['End Date']), inplace=True)

            # Make predictions
            test_predictions = model.forecast(steps=len(book_test_clean))
            actual_values = book_test_clean['Volume'].values

            # Calculate RMSE
            rmse = np.sqrt(np.mean((actual_values - test_predictions.values) ** 2))

            self.logger.debug(f"Model validation for {isbn}: current RMSE={rmse:.2f}, baseline={model_info.baseline_rmse:.2f}")
            return rmse

        except Exception as e:
            self.logger.warning(f"Model validation failed for {isbn}: {e}")
            return None

    def should_retrain_model(
        self,
        isbn: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Tuple[bool, str, Optional[ModelInfo]]:
        """
        Determine if model should be retrained based on various criteria.

        Returns:
            (should_retrain: bool, reason: str, existing_model_info: Optional[ModelInfo])
        """

        # Always retrain if forced
        if self.config.force_retrain:
            return True, "force_retrain=True", None

        # Check if model exists
        existing_model = self.get_latest_model_info(isbn)
        if not existing_model:
            return True, "no_existing_model", None

        # Check if model file still exists
        if not os.path.exists(existing_model.model_path):
            return True, "model_file_missing", existing_model

        # Check model age
        if self.config.max_model_age_days is not None:
            age_days = (datetime.now() - existing_model.created_date).days
            if age_days > self.config.max_model_age_days:
                return True, f"model_too_old_{age_days}_days", existing_model

        # Check data drift
        current_data_hash = self.calculate_data_hash(train_data, test_data, isbn)
        if current_data_hash != existing_model.data_hash:
            return True, "data_drift_detected", existing_model

        # Check quality gates
        if (self.config.min_acceptable_rmse is not None and
            existing_model.baseline_rmse > self.config.min_acceptable_rmse):
            return True, f"rmse_above_threshold_{existing_model.baseline_rmse:.2f}", existing_model

        if (self.config.max_acceptable_mape is not None and
            existing_model.baseline_mape > self.config.max_acceptable_mape):
            return True, f"mape_above_threshold_{existing_model.baseline_mape:.1f}%", existing_model

        # Check performance degradation on current data
        if self.config.performance_threshold is not None:
            current_rmse = self.validate_existing_model(isbn, existing_model, test_data)
            if current_rmse is not None:
                degradation = (current_rmse - existing_model.baseline_rmse) / existing_model.baseline_rmse
                if degradation > self.config.performance_threshold:
                    return True, f"performance_degraded_{degradation:.1%}", existing_model
                else:
                    # Performance is still acceptable
                    self.logger.info(f"Model for {isbn} still performs well: "
                                   f"current RMSE {current_rmse:.2f} vs baseline {existing_model.baseline_rmse:.2f} "
                                   f"({degradation:+.1%})")

        return False, "model_still_valid", existing_model

    def load_existing_model_results(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Load results from an existing model for use in pipeline results"""
        return {
            'isbn': model_info.isbn,
            'best_params': model_info.model_params,
            'evaluation_metrics': {
                'rmse': model_info.baseline_rmse,
                'mae': model_info.baseline_mae,
                'mape': model_info.baseline_mape
            },
            'model_path': model_info.model_path,
            'train_series_length': model_info.training_data_length,
            'test_series_length': model_info.test_data_length,
            'reused_existing_model': True,
            'created_date': model_info.created_date.isoformat(),
            'mlflow_run_id': model_info.mlflow_run_id,
            'mlflow_model_version': model_info.mlflow_model_version
        }

    def log_retraining_decision(self, isbn: str, should_retrain: bool, reason: str,
                              existing_model: Optional[ModelInfo]) -> None:
        """Log retraining decision for monitoring and debugging"""

        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'isbn': isbn,
            'should_retrain': should_retrain,
            'reason': reason,
            'config_environment': self.config.environment,
            'existing_model_age_days': (
                (datetime.now() - existing_model.created_date).days
                if existing_model else None
            ),
            'existing_model_rmse': existing_model.baseline_rmse if existing_model else None
        }

        if 'retraining_history' not in self.model_registry:
            self.model_registry['retraining_history'] = []

        self.model_registry['retraining_history'].append(decision_entry)

        # Keep only last 100 decisions to prevent file growth
        if len(self.model_registry['retraining_history']) > 100:
            self.model_registry['retraining_history'] = self.model_registry['retraining_history'][-100:]

        self._save_model_registry()

        # Log to console
        if should_retrain:
            self.logger.info(f"🔄 Retraining {isbn}: {reason}")
        else:
            self.logger.info(f"♻️  Reusing model for {isbn}: {reason}")

    def get_retraining_stats(self) -> Dict[str, Any]:
        """Get statistics about retraining decisions"""
        history = self.model_registry.get('retraining_history', [])
        if not history:
            return {'total_decisions': 0}

        total_decisions = len(history)
        retrain_decisions = sum(1 for entry in history if entry['should_retrain'])
        reuse_decisions = total_decisions - retrain_decisions

        # Count reasons
        retrain_reasons = {}
        reuse_reasons = {}

        for entry in history:
            reason = entry['reason']
            if entry['should_retrain']:
                retrain_reasons[reason] = retrain_reasons.get(reason, 0) + 1
            else:
                reuse_reasons[reason] = reuse_reasons.get(reason, 0) + 1

        return {
            'total_decisions': total_decisions,
            'retrain_decisions': retrain_decisions,
            'reuse_decisions': reuse_decisions,
            'retrain_rate': retrain_decisions / total_decisions if total_decisions > 0 else 0,
            'retrain_reasons': retrain_reasons,
            'reuse_reasons': reuse_reasons,
            'last_30_days': len([
                entry for entry in history
                if datetime.fromisoformat(entry['timestamp']) > datetime.now() - timedelta(days=30)
            ])
        }


def create_retraining_engine(config: ARIMATrainingConfig, output_dir: str) -> ModelRetrainDecisionEngine:
    """Factory function to create a model retraining decision engine"""
    return ModelRetrainDecisionEngine(config, output_dir)


if __name__ == "__main__":
    # Example usage and testing
    from config.arima_training_config import get_development_config

    config = get_development_config(force_retrain=False)
    engine = create_retraining_engine(config, "/tmp/test_output")

    # Test data simulation
    test_train_data = pd.DataFrame({
        'ISBN': ['9780722532935'] * 10,
        'Volume': np.random.randint(100, 1000, 10)
    })
    test_test_data = test_train_data.copy()

    # Test decision logic
    should_retrain, reason, model_info = engine.should_retrain_model(
        '9780722532935', test_train_data, test_test_data
    )

    print(f"Should retrain: {should_retrain}, Reason: {reason}")
    print(f"Retraining stats: {engine.get_retraining_stats()}")
