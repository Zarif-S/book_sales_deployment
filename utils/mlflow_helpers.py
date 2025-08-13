"""
MLflow Helper Utilities

This module contains MLflow-specific utility functions for model management,
artifact cleanup, and tracking operations.
"""

import os
import shutil
from zenml.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def cleanup_old_mlflow_models(max_models_per_book: int = 2) -> None:
    """
    Cleanup old MLflow model artifacts to prevent disk space issues.
    Keeps only the most recent `max_models_per_book` model.statsmodels files per book.
    """
    try:
        # Find all model.statsmodels files and extract book information
        model_files_by_book = {}

        for root, dirs, files in os.walk("mlruns"):
            for file in files:
                if file == "model.statsmodels":
                    full_path = os.path.join(root, file)
                    stat_info = os.stat(full_path)

                    # Extract book ISBN from run name if available
                    run_dir = full_path.split('/artifacts/')[0]
                    book_isbn = None

                    try:
                        # Try to get run name from tags
                        run_name_file = os.path.join(run_dir, 'tags', 'mlflow.runName')
                        if os.path.exists(run_name_file):
                            with open(run_name_file, 'r') as f:
                                run_name = f.read().strip()
                                # Extract ISBN from run name (format: book_9780123456789_Title_timestamp)
                                if 'book_' in run_name:
                                    parts = run_name.split('_')
                                    for i, part in enumerate(parts):
                                        if part == 'book' and i + 1 < len(parts):
                                            potential_isbn = parts[i + 1]
                                            if len(potential_isbn) == 13 and potential_isbn.isdigit():
                                                book_isbn = potential_isbn
                                                break
                    except Exception:
                        # If we can't extract from run name, skip this model
                        continue

                    if book_isbn:
                        if book_isbn not in model_files_by_book:
                            model_files_by_book[book_isbn] = []
                        model_files_by_book[book_isbn].append((full_path, stat_info.st_mtime, stat_info.st_size, run_dir))

        if not model_files_by_book:
            logger.info("✅ Model cleanup: No models found with identifiable book ISBNs")
            return

        total_removed = 0
        total_size_removed = 0

        # Process each book separately
        for book_isbn, book_models in model_files_by_book.items():
            # Sort by modification time (newest first)
            book_models.sort(key=lambda x: x[1], reverse=True)

            models_to_keep = len(book_models)
            models_to_remove = max(0, len(book_models) - max_models_per_book)

            logger.info(f"📚 Book {book_isbn}: Found {len(book_models)} models, keeping {min(len(book_models), max_models_per_book)}")

            if models_to_remove > 0:
                # Remove old models for this book (keep the newest max_models_per_book)
                for file_path, mod_time, size, run_dir in book_models[max_models_per_book:]:
                    try:
                        if os.path.exists(run_dir):
                            shutil.rmtree(run_dir)
                            total_removed += 1
                            total_size_removed += size
                            logger.info(f"🗑️  Removed old model for {book_isbn}: {os.path.basename(run_dir)} ({size/(1024*1024):.1f}MB)")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to remove {file_path}: {e}")

        if total_removed > 0:
            size_mb = total_size_removed / (1024 * 1024)
            logger.info(f"✅ Model cleanup completed: Removed {total_removed} old model runs ({size_mb:.1f}MB total)")
        else:
            logger.info(f"✅ Model cleanup: All models within per-book limit of {max_models_per_book}")

    except Exception as e:
        logger.warning(f"⚠️  Model cleanup failed: {e}")