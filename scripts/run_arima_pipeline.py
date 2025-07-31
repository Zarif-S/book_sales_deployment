#!/usr/bin/env python3
"""
Script to run the complete ARIMA pipeline for book sales data.

This script runs the full pipeline including:
- Data loading and preprocessing
- Data quality analysis
- ARIMA modeling with Optuna optimization
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.zenml_pipeline_with_arima import book_sales_arima_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete ARIMA pipeline."""
    try:
        logger.info("Starting ARIMA pipeline execution")
        
        # Set up output directory
        output_dir = os.path.join(project_root, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Define parameters for ARIMA modeling
        selected_isbns = [
            '9780722532935',  # The Alchemist
            '9780241003008'   # The Very Hungry Caterpillar
        ]
        
        logger.info(f"Using ISBNs: {selected_isbns}")
        logger.info(f"Output directory: {output_dir}")
        
        # Run the pipeline
        results = book_sales_arima_pipeline(
            output_dir=output_dir,
            selected_isbns=selected_isbns,
            column_name='Volume',
            split_size=32
        )
        
        logger.info("ARIMA pipeline completed successfully!")
        logger.info(f"Pipeline returned {len(results)} result items")
        
        # Log key results
        if 'arima_results' in results:
            arima_df = results['arima_results']
            logger.info(f"ARIMA results shape: {arima_df.shape}")
            
            # Check for model configuration results
            model_configs = arima_df[arima_df['result_type'] == 'model_config']
            if not model_configs.empty:
                logger.info(f"Found {len(model_configs)} model configurations")
                for _, row in model_configs.iterrows():
                    logger.info(f"  {row['parameter']}: {row['value']}")
            
            # Check for evaluation metrics
            eval_metrics = arima_df[arima_df['result_type'] == 'evaluation']
            if not eval_metrics.empty:
                logger.info(f"Found {len(eval_metrics)} evaluation metrics")
                for _, row in eval_metrics.iterrows():
                    logger.info(f"  {row['parameter']}: {row['value']}")
        
        if 'quality_report' in results:
            quality = results['quality_report']
            logger.info(f"Data quality score: {quality.get('quality_score', 'N/A')}%")
        
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