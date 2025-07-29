import os
import pandas as pd
from typing import Tuple, Annotated

from zenml import step, pipeline
from zenml.logger import get_logger
from zenml import ArtifactConfig
from zenml.steps import get_step_context
from zenml.pipelines import get_pipeline_context
from zenml.integrations.pandas.materializers.pandas_materializer import PandasMaterializer

# Import your existing modules
from scripts._01_load_xml import get_steps_df, get_sleep_df
from scripts._02_preprocess_step_data import preprocess_steps_func
from scripts._03_preprocess_sleep_data import preprocess_sleep_func
from scripts._04_plots import plot_daily_steps_func, plot_daily_sleep_duration_func, plot_hourly_steps_func

logger = get_logger(__name__)

# Configure step settings to enable metadata
step_settings = {
    "enable_artifact_metadata": True,
    "enable_artifact_visualization": True,
}

# ------------------ STEPS ------------------ #

@step(
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def extract_steps(xml_path: str, extracted_dir: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="raw_steps_data")]:
    """Extract steps data from XML and return as DataFrame artifact."""
    logger.info(f"Starting extraction from {xml_path}")
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    os.makedirs(extracted_dir, exist_ok=True)
    df = get_steps_df(xml_path)

    # Save CSV for backup (optional)
    steps_csv_path = os.path.join(extracted_dir, 'steps_raw.csv')
    df.to_csv(steps_csv_path, index=False)

    # Prepare metadata with better error handling
    total_records = len(df)
    columns_list = list(df.columns) if hasattr(df, 'columns') else []
    
    date_range_start = ""
    date_range_end = ""
    if 'date' in df.columns and not df.empty and df['date'].notnull().any():
        date_range_start = str(df['date'].min())
        date_range_end = str(df['date'].max())

    metadata_dict = {
        "total_records": total_records,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "columns": columns_list,
        "file_saved_to": steps_csv_path,
        "data_shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "extraction_source": xml_path
    }

    # Log metadata with detailed debugging
    logger.info(f"Metadata to be logged: {metadata_dict}")
    
    try:
        context = get_step_context()
        context.add_output_metadata(
            output_name="raw_steps_data",
            metadata={
                "total_records": len(df),
                "date_range_start": str(df['start_date'].min()) if 'start_date' in df.columns and df['start_date'].notnull().any() else "",
                "date_range_end": str(df['end_date'].max()) if 'end_date' in df.columns and df['end_date'].notnull().any() else "",
                "columns": list(df.columns) if hasattr(df, 'columns') else [],
                "file_saved_to": steps_csv_path
            }
        )
        logger.info("Successfully added output metadata")
    except Exception as e:
        logger.error(f"Failed to add metadata: {e}")

    logger.info(f"Extracted {total_records} step records from {xml_path}")
    return df


@step(
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def preprocess_steps(steps_raw_df: pd.DataFrame, processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess steps data and return processed daily and hourly DataFrames."""
    logger.info("Starting steps preprocessing")
    os.makedirs(processed_dir, exist_ok=True)

    temp_csv = os.path.join(processed_dir, 'temp_raw_steps.csv')
    steps_raw_df.to_csv(temp_csv, index=False)

    processed_paths = preprocess_steps_func(temp_csv, processed_dir)
    daily_steps_df = pd.read_csv(processed_paths["daily"])
    hourly_steps_df = pd.read_csv(processed_paths["hourly"])

    # --- metadata ---
    metadata_dict = {
        "raw_records": len(steps_raw_df),
        "daily_records": len(daily_steps_df),
        "hourly_records": len(hourly_steps_df),
        "avg_daily_steps": float(daily_steps_df['steps'].mean()) if not daily_steps_df.empty else 0.0,
        "max_daily_steps": int(daily_steps_df['steps'].max()) if not daily_steps_df.empty else 0,
        "daily_file_path": processed_paths["daily"],
        "hourly_file_path": processed_paths["hourly"],
    }
    logger.info(f"Preprocessing metadata: {metadata_dict}")

    try:
        context = get_step_context()
        context.add_output_metadata(output_name="processed_steps_data", metadata=metadata_dict)
    except Exception as e:
        logger.error(f"Failed to add preprocessing metadata: {e}")

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    logger.info(f"Processed {len(daily_steps_df)} daily and {len(hourly_steps_df)} hourly step records")
    return daily_steps_df, hourly_steps_df


@step(enable_cache=False, **step_settings)
def plot_daily_steps(daily_steps_df: pd.DataFrame, output_dir: str) -> Annotated[str, ArtifactConfig(name="steps_plot_path")]:
    """Generate daily steps plot and return path."""
    logger.info("Generating daily steps plot")
    os.makedirs(output_dir, exist_ok=True)
    
    temp_csv = os.path.join(output_dir, 'temp_daily_steps.csv')
    daily_steps_df.to_csv(temp_csv, index=False)

    plot_path = plot_daily_steps_func(temp_csv, output_dir)
    
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    metadata_dict = {
        "plot_type": "daily_steps",
        "data_points": len(daily_steps_df),
        "plot_saved_to": plot_path,
        "plot_format": "html",
        "file_size_mb": round(os.path.getsize(plot_path) / (1024*1024), 2) if os.path.exists(plot_path) else 0
    }

    logger.info(f"Plot metadata: {metadata_dict}")
    
    try:
        context = get_step_context()
        context.add_output_metadata(
            output_name="steps_plot_path",
            metadata=metadata_dict
        )
        logger.info("Successfully added plot metadata")
    except Exception as e:
        logger.error(f"Failed to add plot metadata: {e}")

    logger.info(f"Daily steps plot saved to {plot_path}")
    return plot_path


@step(
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def extract_sleeps(xml_path: str, extracted_dir: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="raw_sleep_data")]:
    """Extract sleep data from XML and return as DataFrame artifact."""
    logger.info(f"Starting sleep extraction from {xml_path}")
    os.makedirs(extracted_dir, exist_ok=True)
    df = get_sleep_df(xml_path)

    sleep_csv_path = os.path.join(extracted_dir, 'sleep_raw.csv')
    df.to_csv(sleep_csv_path, index=False)

    # Prepare metadata safely
    sleep_types = []
    date_range_start = ""
    date_range_end = ""
    
    if 'type' in df.columns and not df.empty and df['type'].notnull().any():
        sleep_types = df['type'].unique().tolist()
    
    if 'date' in df.columns and not df.empty and df['date'].notnull().any():
        date_range_start = str(df['date'].min())
        date_range_end = str(df['date'].max())

    metadata_dict = {
        "total_sleep_records": len(df),
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "sleep_types": sleep_types,
        "file_saved_to": sleep_csv_path,
        "data_shape": f"{df.shape[0]} rows x {df.shape[1]} columns"
    }

    logger.info(f"Sleep extraction metadata: {metadata_dict}")
    
    try:
        context = get_step_context()
        context.add_output_metadata(
            output_name="raw_sleep_data",
            metadata={
                "total_sleep_records": len(df),
                "date_range_start": str(df['start_date'].min()) if 'start_date' in df.columns and df['start_date'].notnull().any() else "",
                "date_range_end": str(df['end_date'].max()) if 'end_date' in df.columns and df['end_date'].notnull().any() else "",
                "sleep_types": df['type'].unique().tolist() if 'type' in df.columns and df['type'].notnull().any() else [],
                "file_saved_to": sleep_csv_path
            }
        )
        logger.info("Successfully added sleep extraction metadata")
    except Exception as e:
        logger.error(f"Failed to add sleep extraction metadata: {e}")

    logger.info(f"Extracted {len(df)} sleep records from {xml_path}")
    return df


@step(
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer,
    enable_cache=False,
)
def preprocess_sleep(sleep_raw_df: pd.DataFrame, processed_dir: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="processed_sleep_data")]:
    """Preprocess sleep data and return processed DataFrame."""
    logger.info("Starting sleep preprocessing")
    os.makedirs(processed_dir, exist_ok=True)

    temp_csv = os.path.join(processed_dir, 'temp_raw_sleep.csv')
    sleep_raw_df.to_csv(temp_csv, index=False)

    processed_csv_path = preprocess_sleep_func(temp_csv, processed_dir)
    processed_df = pd.read_csv(processed_csv_path)
    
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    # Calculate sleep statistics safely
    avg_sleep_duration = 0.0
    min_sleep_duration = 0.0
    max_sleep_duration = 0.0
    
    if 'duration' in processed_df.columns and not processed_df.empty:
        avg_sleep_duration = float(processed_df['duration'].mean())
        min_sleep_duration = float(processed_df['duration'].min())
        max_sleep_duration = float(processed_df['duration'].max())

    metadata_dict = {
        "raw_sleep_records": len(sleep_raw_df),
        "processed_records": len(processed_df),
        "avg_sleep_duration": avg_sleep_duration,
        "min_sleep_duration": min_sleep_duration,
        "max_sleep_duration": max_sleep_duration,
        "processed_file_path": processed_csv_path,
        "sleep_duration_unit": "hours"  # Assuming your duration is in hours
    }

    logger.info(f"Sleep preprocessing metadata: {metadata_dict}")
    
    try:
        context = get_step_context()
        context.add_output_metadata(
            output_name="processed_sleep_data",
            metadata=metadata_dict
        )
        logger.info("Successfully added sleep preprocessing metadata")
    except Exception as e:
        logger.error(f"Failed to add sleep preprocessing metadata: {e}")

    logger.info(f"Processed to {len(processed_df)} daily sleep duration records")
    return processed_df


@step(enable_cache=False, **step_settings)
def plot_daily_sleep_duration(daily_sleep_df: pd.DataFrame, output_dir: str) -> Annotated[str, ArtifactConfig(name="sleep_plot_path")]:
    """Generate daily sleep duration plot and return path."""
    logger.info("Generating daily sleep duration plot")
    os.makedirs(output_dir, exist_ok=True)
    
    temp_csv = os.path.join(output_dir, 'temp_daily_sleep.csv')
    daily_sleep_df.to_csv(temp_csv, index=False)

    plot_path = plot_daily_sleep_duration_func(temp_csv, output_dir)
    
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    metadata_dict = {
        "plot_type": "daily_sleep_duration",
        "data_points": len(daily_sleep_df),
        "plot_saved_to": plot_path,
        "plot_format": "html",
        "file_size_mb": round(os.path.getsize(plot_path) / (1024*1024), 2) if os.path.exists(plot_path) else 0
    }

    logger.info(f"Sleep plot metadata: {metadata_dict}")
    
    try:
        context = get_step_context()
        context.add_output_metadata(
            output_name="sleep_plot_path",
            metadata=metadata_dict
        )
        logger.info("Successfully added sleep plot metadata")
    except Exception as e:
        logger.error(f"Failed to add sleep plot metadata: {e}")

    logger.info(f"Daily sleep plot saved to {plot_path}")
    return plot_path

@step(enable_cache=False, **step_settings)
def plot_hourly_steps(hourly_steps_df: pd.DataFrame, output_dir: str) -> Annotated[str, ArtifactConfig(name="hourly_steps_plot_path")]:
    """Generate hourly steps plot and return path."""
    logger.info("Generating hourly steps plot")
    os.makedirs(output_dir, exist_ok=True)
    
    temp_csv = os.path.join(output_dir, 'temp_hourly_steps.csv')
    hourly_steps_df.to_csv(temp_csv, index=False)

    # You will need to create this function in your plots module
    plot_path = plot_hourly_steps_func(temp_csv, output_dir)

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    metadata_dict = {
        "plot_type": "hourly_steps",
        "data_points": len(hourly_steps_df),
        "plot_saved_to": plot_path,
        "plot_format": "html",
        "file_size_mb": round(os.path.getsize(plot_path) / (1024*1024), 2) if os.path.exists(plot_path) else 0
    }

    logger.info(f"Hourly steps plot metadata: {metadata_dict}")
    
    try:
        context = get_step_context()
        context.add_output_metadata(
            output_name="hourly_steps_plot_path",
            metadata=metadata_dict
        )
        logger.info("Successfully added hourly plot metadata")
    except Exception as e:
        logger.error(f"Failed to add hourly plot metadata: {e}")

    logger.info(f"Hourly steps plot saved to {plot_path}")
    return plot_path

# ------------------ PIPELINE ------------------ #

@pipeline
def apple_health_pipeline(xml_path: str, extracted_dir: str, processed_dir: str, plot_dir: str) -> Tuple[str, str, str]:
    """Apple Health data processing pipeline with hourly steps plot."""
    logger.info(f"Running pipeline: apple_health_pipeline")

    # Steps pipeline
    steps_raw_df = extract_steps(xml_path=xml_path, extracted_dir=extracted_dir)
    daily_steps_df, hourly_steps_df = preprocess_steps(steps_raw_df=steps_raw_df, processed_dir=processed_dir)
    plot_steps_html_path = plot_daily_steps(daily_steps_df=daily_steps_df, output_dir=plot_dir)
    plot_hourly_steps_html_path = plot_hourly_steps(hourly_steps_df=hourly_steps_df, output_dir=plot_dir)

    # Sleep pipeline  
    sleep_raw_df = extract_sleeps(xml_path=xml_path, extracted_dir=extracted_dir)
    processed_sleep_df = preprocess_sleep(sleep_raw_df=sleep_raw_df, processed_dir=processed_dir)
    plot_sleep_html_path = plot_daily_sleep_duration(daily_sleep_df=processed_sleep_df, output_dir=plot_dir)

    return plot_steps_html_path, plot_hourly_steps_html_path, plot_sleep_html_path

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    project_root = os.path.expanduser('~/Documents/Projects/apple_watch_project')
    xml_path = os.path.join(project_root, 'data', 'raw', 'apple_health_export', 'export.xml')
    extracted_dir = os.path.join(project_root, 'data', 'extracted')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    plot_dir = os.path.join(project_root, 'data', 'plots')

    apple_health_pipeline(
        xml_path=xml_path,
        extracted_dir=extracted_dir,
        processed_dir=processed_dir,
        plot_dir=plot_dir
    )

    print("Pipeline completed successfully!")