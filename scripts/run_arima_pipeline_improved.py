
#### DO NOT RUN THIS ##########

#!/usr/bin/env python3
"""
Script to run the IMPROVED complete ARIMA pipeline for book sales data.

IMPROVEMENTS:
- Uses persistent Optuna storage for hyperparameter optimization
- Enables ZenML caching for faster repeated runs
- Returns structured outputs including hyperparameters
- Handles all metadata properly to avoid unhashable type errors

This script runs the full pipeline including:
- Data loading and preprocessing
- Data quality analysis
- ARIMA modeling with persistent Optuna optimization
- Structured output handling
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.zenml_pipeline_latest_31_jul import book_sales_arima_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the improved complete ARIMA pipeline."""
    try:
        logger.info("Starting IMPROVED ARIMA pipeline execution")

        # Set up output directory
        output_dir = os.path.join(project_root, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)

        # Define parameters for ARIMA modeling
        selected_isbns = [
            '9780722532935',  # The Alchemist
        ]

        logger.info(f"Using ISBNs: {selected_isbns}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("IMPROVEMENTS: Persistent Optuna storage, ZenML caching, structured outputs")

        # Run the improved pipeline
        results = book_sales_arima_pipeline(
            output_dir=output_dir,
            selected_isbns=selected_isbns,
            column_name='Volume',
            split_size=32,
            n_trials=30  # Configurable optimization trials
        )

        logger.info("IMPROVED ARIMA pipeline completed successfully!")
        logger.info("Pipeline run completed. Check the ZenML dashboard for detailed results.")

        # Note: ZenML pipelines return PipelineRunResponse objects, not dictionaries
        # The actual results are available through the ZenML dashboard and artifact store
        logger.info("Pipeline results are available in the ZenML dashboard and artifact store.")
        logger.info("You can access individual step outputs through the ZenML client.")
        logger.info("IMPROVEMENTS: Persistent Optuna storage will resume optimization in future runs.")
        logger.info("IMPROVEMENTS: ZenML caching will skip steps when inputs haven't changed.")

        logger.info("Pipeline execution completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
