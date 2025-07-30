import pytest
import pandas as pd
import os
import tempfile
import shutil
import time
import numpy as np
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
import logging

# Import the actual functions from the correct module
from steps._01_load_data import (
    get_merged_data, load_data_from_google_sheets, download_google_sheet,
    concatenate_sheets, is_file_recent, save_dataframe_as_csv,
    save_raw_data_as_csv, load_isbn_data, load_uk_weekly_data, get_data_config
)

# =============================================================================
# 1. INTEGRATION TESTS FOR get_merged_data WITH FULLY MOCKED DEPENDENCIES
# =============================================================================

class TestGetMergedDataIntegration:
    """Integration tests for get_merged_data with all external dependencies mocked."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'isbn_file_id': 'test_isbn_id_123',
            'uk_file_id': 'test_uk_id_456',
            'isbn_sheet_names': ["F - Adult Fiction", "S Adult Non-Fiction"],
            'uk_sheet_names': ["F Adult Fiction", "S Adult Non-Fiction"],
            'source_labels': ["F - Adult Fiction", "S Adult Non-Fiction"]
        }
    
    @pytest.fixture
    def sample_dataframes(self):
        """Sample DataFrames that would be loaded from sheets."""
        isbn_df = pd.DataFrame({
            'ISBN': ['978-1234567890', '978-0987654321'],
            'Title': ['Test Book 1', 'Test Book 2'],
            'Author': ['Author A', 'Author B'],
            'Source': ['F - Adult Fiction', 'S Adult Non-Fiction']
        })
        
        uk_df = pd.DataFrame({
            'Week': ['2023-W01', '2023-W02'],
            'Sales': [100, 150],
            'Rank': [1, 2],
            'Source': ['F Adult Fiction', 'S Adult Non-Fiction']
        })
        
        return isbn_df, uk_df
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('steps._01_load_data.save_raw_data_as_csv')
    @patch('steps._01_load_data.load_uk_weekly_data')
    @patch('steps._01_load_data.load_isbn_data')
    @patch('steps._01_load_data.get_data_config')
    def test_get_merged_data_success_no_force_download(
        self, mock_get_config, mock_load_isbn, mock_load_uk, mock_save_csv,
        sample_config, sample_dataframes
    ):
        """Test successful data merging without forced download."""
        isbn_df, uk_df = sample_dataframes
        
        # Setup mocks
        mock_get_config.return_value = sample_config
        mock_load_isbn.return_value = isbn_df
        mock_load_uk.return_value = uk_df
        mock_save_csv.return_value = {
            'isbn': '/path/to/ISBN_data.csv',
            'uk_weekly': '/path/to/UK_weekly_data.csv'
        }
        
        # Execute
        result = get_merged_data(force_download=False)
        
        # Verify function calls
        mock_get_config.assert_called_once()
        mock_load_isbn.assert_called_once_with(sample_config, False)
        mock_load_uk.assert_called_once_with(sample_config, False)
        
        # Verify save_csv was called with correct structure
        expected_datasets = {
            'isbn': (isbn_df, 'ISBN_data.csv', 'ISBN data'),
            'uk_weekly': (uk_df, 'UK_weekly_data.csv', 'UK weekly data')
        }
        mock_save_csv.assert_called_once_with(expected_datasets)
        
        # Verify return structure
        assert 'df_isbns' in result
        assert 'df_uk_weekly' in result
        pd.testing.assert_frame_equal(result['df_isbns'], isbn_df)
        pd.testing.assert_frame_equal(result['df_uk_weekly'], uk_df)
    
    @patch('steps._01_load_data.save_raw_data_as_csv')
    @patch('steps._01_load_data.load_uk_weekly_data')
    @patch('steps._01_load_data.load_isbn_data')
    @patch('steps._01_load_data.get_data_config')
    def test_get_merged_data_with_force_download(
        self, mock_get_config, mock_load_isbn, mock_load_uk, mock_save_csv,
        sample_config, sample_dataframes
    ):
        """Test data merging with forced download."""
        isbn_df, uk_df = sample_dataframes
        
        mock_get_config.return_value = sample_config
        mock_load_isbn.return_value = isbn_df
        mock_load_uk.return_value = uk_df
        
        get_merged_data(force_download=True)
        
        # Verify force_download=True was passed through
        mock_load_isbn.assert_called_once_with(sample_config, True)
        mock_load_uk.assert_called_once_with(sample_config, True)
    
    @patch('steps._01_load_data.save_raw_data_as_csv')
    @patch('steps._01_load_data.load_uk_weekly_data')
    @patch('steps._01_load_data.load_isbn_data')
    @patch('steps._01_load_data.get_data_config')
    def test_get_merged_data_isbn_load_failure(
        self, mock_get_config, mock_load_isbn, mock_load_uk, mock_save_csv,
        sample_config
    ):
        """Test behavior when ISBN data loading fails."""
        mock_get_config.return_value = sample_config
        mock_load_isbn.side_effect = Exception("Failed to load ISBN data")
        
        with pytest.raises(Exception, match="Failed to load ISBN data"):
            get_merged_data()
        
        # UK data should not be attempted if ISBN fails
        mock_load_uk.assert_not_called()
        mock_save_csv.assert_not_called()
    
    @patch('steps._01_load_data.save_raw_data_as_csv')
    @patch('steps._01_load_data.load_uk_weekly_data')
    @patch('steps._01_load_data.load_isbn_data')
    @patch('steps._01_load_data.get_data_config')
    def test_get_merged_data_empty_dataframes(
        self, mock_get_config, mock_load_isbn, mock_load_uk, mock_save_csv,
        sample_config
    ):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        mock_get_config.return_value = sample_config
        mock_load_isbn.return_value = empty_df
        mock_load_uk.return_value = empty_df
        
        result = get_merged_data()
        
        # Should still complete successfully
        assert 'df_isbns' in result
        assert 'df_uk_weekly' in result
        assert result['df_isbns'].empty
        assert result['df_uk_weekly'].empty


# =============================================================================
# 2. PERFORMANCE TESTS FOR LARGE DATASETS
# =============================================================================

class TestPerformance:
    """Performance tests for data processing functions with large datasets."""
    
    @pytest.fixture(params=[1000, 10000, 50000])
    def large_dataframe(self, request):
        """Generate DataFrames of varying sizes for performance testing."""
        size = request.param
        np.random.seed(42)  # For reproducible results
        
        df = pd.DataFrame({
            'id': range(size),
            'text_data': [f'Sample text data row {i}' for i in range(size)],
            'numeric_data': np.random.randn(size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size),
            'date_col': pd.date_range('2023-01-01', periods=size, freq='h')
        })
        return df, size
    
    def test_concatenate_sheets_performance(self, large_dataframe):
        """Test concatenate_sheets performance with large datasets."""
        df, size = large_dataframe
        
        # Create multiple sheets to concatenate
        sheets_data = [
            (df.copy(), f'Sheet_{i}') for i in range(5)
        ]
        
        start_time = time.time()
        result = concatenate_sheets(sheets_data)
        execution_time = time.time() - start_time
        
        # Verify correctness
        expected_rows = size * 5
        assert len(result) == expected_rows
        assert 'Source' in result.columns
        
        # Performance assertions (adjust thresholds based on your requirements)
        if size <= 1000:
            assert execution_time < 0.1  # 100ms for small datasets
        elif size <= 10000:
            assert execution_time < 1.0   # 1s for medium datasets
        else:
            assert execution_time < 5.0   # 5s for large datasets
        
        print(f"Concatenated {expected_rows} rows in {execution_time:.3f}s")
    
    def test_save_dataframe_performance(self, large_dataframe, tmp_path):
        """Test CSV saving performance with large datasets."""
        df, size = large_dataframe
        
        with patch('steps._01_load_data.ensure_raw_data_dir', return_value=str(tmp_path)):
            start_time = time.time()
            result_path = save_dataframe_as_csv(df, f'perf_test_{size}.csv', f'Performance test {size}')
            execution_time = time.time() - start_time
            
            # Verify file was created and has correct size
            assert os.path.exists(result_path)
            saved_df = pd.read_csv(result_path)
            assert len(saved_df) == size
            
            # Performance thresholds
            if size <= 1000:
                assert execution_time < 0.5
            elif size <= 10000:
                assert execution_time < 2.0
            else:
                assert execution_time < 10.0
            
            print(f"Saved {size} rows to CSV in {execution_time:.3f}s")
    
    @pytest.mark.parametrize("num_sheets", [2, 5, 10])
    def test_multiple_sheet_concatenation_scaling(self, num_sheets):
        """Test how concatenation performance scales with number of sheets."""
        base_df = pd.DataFrame({
            'col1': range(1000),
            'col2': np.random.randn(1000)
        })
        
        sheets_data = [(base_df.copy(), f'Sheet_{i}') for i in range(num_sheets)]
        
        start_time = time.time()
        result = concatenate_sheets(sheets_data)
        execution_time = time.time() - start_time
        
        assert len(result) == 1000 * num_sheets
        
        # Should scale roughly linearly
        expected_max_time = 0.01 * num_sheets  # 10ms per sheet baseline
        assert execution_time < expected_max_time
        
        print(f"Concatenated {num_sheets} sheets in {execution_time:.4f}s")


# =============================================================================
# 3. MOCK EXTERNAL DEPENDENCIES (gdown, network calls)
# =============================================================================

class TestExternalDependencies:
    """Test functions that interact with external dependencies."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for downloads."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('steps._01_load_data.gdown.download')
    @patch('steps._01_load_data.is_file_recent')
    def test_download_google_sheet_fresh_download(
        self, mock_is_recent, mock_gdown, temp_directory
    ):
        """Test downloading when no cached file exists."""
        file_id = 'test_file_id_123'
        destination = os.path.join(temp_directory, 'test_download.xlsx')
        
        # Mock: no recent file exists
        mock_is_recent.return_value = False
        mock_gdown.return_value = None  # Successful download
        
        result = download_google_sheet(file_id, destination, force_download=False)
        
        # Verify gdown was called with correct URL
        expected_url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx'
        mock_gdown.assert_called_once_with(expected_url, destination, quiet=False)
        
        assert result == destination
    
    @patch('steps._01_load_data.gdown.download')
    @patch('steps._01_load_data.is_file_recent')
    def test_download_google_sheet_use_cache(
        self, mock_is_recent, mock_gdown, temp_directory
    ):
        """Test using cached file when it's recent enough."""
        file_id = 'test_file_id_123'
        destination = os.path.join(temp_directory, 'test_download.xlsx')
        
        # Create a dummy cached file
        with open(destination, 'w') as f:
            f.write('cached content')
        
        # Mock: recent file exists
        mock_is_recent.return_value = True
        
        result = download_google_sheet(file_id, destination, force_download=False)
        
        # Should not attempt download
        mock_gdown.assert_not_called()
        assert result == destination
    
    @patch('steps._01_load_data.gdown.download')
    @patch('steps._01_load_data.is_file_recent')
    @patch('steps._01_load_data.logger')
    def test_download_google_sheet_download_failure_with_cache(
        self, mock_logger, mock_is_recent, mock_gdown, temp_directory
    ):
        """Test download failure when cached file exists."""
        file_id = 'test_file_id_123'
        destination = os.path.join(temp_directory, 'test_download.xlsx')
        
        # Create a cached file
        with open(destination, 'w') as f:
            f.write('cached content')
        
        # Mock: file not recent, download fails
        mock_is_recent.return_value = False
        mock_gdown.side_effect = Exception("Network error")
        
        result = download_google_sheet(file_id, destination, force_download=False)
        
        # Should fallback to cached file
        assert result == destination
        mock_logger.error.assert_called_once()
        mock_logger.warning.assert_called_once()
    
    @patch('steps._01_load_data.gdown.download')
    @patch('steps._01_load_data.is_file_recent')
    def test_download_google_sheet_download_failure_no_cache(
        self, mock_is_recent, mock_gdown, temp_directory
    ):
        """Test download failure when no cached file exists."""
        file_id = 'test_file_id_123'
        destination = os.path.join(temp_directory, 'nonexistent.xlsx')
        
        # Mock: no cached file, download fails
        mock_is_recent.return_value = False
        mock_gdown.side_effect = Exception("Network error")
        
        with pytest.raises(Exception, match="Download failed and no cached file available"):
            download_google_sheet(file_id, destination, force_download=False)
    
    @patch('steps._01_load_data.gdown.download')
    @patch('steps._01_load_data.is_file_recent')
    def test_download_google_sheet_force_download(
        self, mock_is_recent, mock_gdown, temp_directory
    ):
        """Test forced download ignores cache."""
        file_id = 'test_file_id_123'
        destination = os.path.join(temp_directory, 'test_download.xlsx')
        
        # Even if file would be considered recent, force download
        mock_is_recent.return_value = True
        
        download_google_sheet(file_id, destination, force_download=True)
        
        # Should still download despite recent file
        mock_gdown.assert_called_once()
    
    @patch('steps._01_load_data.load_and_concat_sheets')
    @patch('steps._01_load_data.download_google_sheet')
    @patch('steps._01_load_data.ensure_raw_data_dir')
    def test_load_data_from_google_sheets_integration(
        self, mock_ensure_dir, mock_download, mock_load_concat, temp_directory
    ):
        """Test the full load_data_from_google_sheets flow."""
        # Setup
        mock_ensure_dir.return_value = temp_directory
        destination_file = os.path.join(temp_directory, 'test.xlsx')
        mock_download.return_value = destination_file
        
        expected_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_load_concat.return_value = expected_df
        
        # Execute
        result = load_data_from_google_sheets(
            file_id='test_id',
            sheet_names=['Sheet1', 'Sheet2'],
            source_labels=['Source1', 'Source2'],
            destination_filename='test.xlsx',
            force_download=True
        )
        
        # Verify the chain of calls
        mock_ensure_dir.assert_called_once()
        mock_download.assert_called_once_with('test_id', destination_file, True)
        mock_load_concat.assert_called_once_with(
            destination_file, ['Sheet1', 'Sheet2'], ['Source1', 'Source2']
        )
        
        pd.testing.assert_frame_equal(result, expected_df)


# =============================================================================
# PYTEST CONFIGURATION AND FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def sample_large_dataset():
    """Session-scoped fixture for large dataset tests to avoid recreation."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100000),
        'value': np.random.randn(100000),
        'category': np.random.choice(['A', 'B', 'C'], 100000)
    })

# Pytest markers for test organization
pytest.mark.integration = pytest.mark.parametrize("", [])
pytest.mark.performance = pytest.mark.parametrize("", [])
pytest.mark.external = pytest.mark.parametrize("", [])

# Example usage in conftest.py or pytest.ini:
# [tool:pytest]
# markers =
#     integration: marks tests as integration tests
#     performance: marks tests as performance tests  
#     external: marks tests that mock external dependencies
#     slow: marks tests as slow (deselect with '-m "not slow"')

if __name__ == '__main__':
    # Run with: python -m pytest test_advanced.py -v
    # Or specific markers: python -m pytest test_advanced.py -m "integration" -v
    # Or exclude slow tests: python -m pytest test_advanced.py -m "not performance" -v
    pytest.main([__file__, '-v'])