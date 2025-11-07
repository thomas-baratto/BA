"""
Unit tests for data_loader.py
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from data_loader import CSVDataset, load_data


class TestCSVDataset:
    """Tests for CSVDataset class."""
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([[5.0], [6.0]])
        
        dataset = CSVDataset(data, labels)
        
        assert len(dataset) == 2
        assert dataset.data.shape == (2, 2)
        assert dataset.labels.shape == (2, 1)
    
    def test_dataset_getitem(self):
        """Test that dataset returns correct items."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([[5.0], [6.0]])
        
        dataset = CSVDataset(data, labels)
        
        item_data, item_label = dataset[0]
        assert item_data.numpy().tolist() == [1.0, 2.0]
        assert item_label.numpy().tolist() == [5.0]
    
    def test_dataset_length(self):
        """Test that dataset length is correct."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([[7.0], [8.0], [9.0]])
        
        dataset = CSVDataset(data, labels)
        
        assert len(dataset) == 3


class TestLoadData:
    """Tests for load_data function."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a temporary CSV file for testing."""
        data = {
            'Flow_well': [100, 200, 300, 400, 500],
            'Temp_diff': [10, 20, 30, 40, 50],
            'kW_well': [5, 10, 15, 20, 25],
            'Hydr_gradient': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Hydr_conductivity': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Aqu_thickness': [10, 20, 30, 40, 50],
            'Long_dispersivity': [1, 2, 3, 4, 5],
            'Trans_dispersivity': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Isotherm': [1, 2, 3, 4, 5],
            'Area': [100, 200, 300, 400, 500],
            'Iso_distance': [10, 20, 30, 40, 50],
            'Iso_width': [5, 10, 15, 20, 25]
        }
        df = pd.DataFrame(data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_load_data_basic(self, sample_csv):
        """Test basic data loading."""
        feature_cols = ['Flow_well', 'Temp_diff', 'kW_well']
        label_cols = ['Area']
        
        X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
            csv_file=sample_csv,
            feature_cols=feature_cols,
            label_cols=label_cols,
            test_size=0.2,
            plots=False
        )
        
        # Check shapes
        assert X_train.shape[1] == 3  # 3 features
        assert y_train.shape[1] == 1  # 1 label
        assert X_test.shape[1] == 3
        assert y_test.shape[1] == 1
        
        # Check train/test split ratio
        total_samples = len(X_train) + len(X_test)
        assert total_samples == 5
        assert len(X_test) == 1  # 20% of 5
    
    def test_load_data_multiple_labels(self, sample_csv):
        """Test loading with multiple labels."""
        feature_cols = ['Flow_well', 'Temp_diff']
        label_cols = ['Area', 'Iso_distance', 'Iso_width']
        
        X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
            csv_file=sample_csv,
            feature_cols=feature_cols,
            label_cols=label_cols,
            test_size=0.2,
            plots=False
        )
        
        assert y_train.shape[1] == 3  # 3 labels
        assert y_test.shape[1] == 3
    
    def test_load_data_scaling(self, sample_csv):
        """Test that data is properly scaled."""
        feature_cols = ['Flow_well', 'Temp_diff']
        label_cols = ['Area']
        
        X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
            csv_file=sample_csv,
            feature_cols=feature_cols,
            label_cols=label_cols,
            test_size=0.2,
            plots=False
        )
        
        # Check that values are in [0, 1] range after MinMaxScaler
        # Use small tolerance for floating point precision
        assert X_train.min() >= -1e-10
        assert X_train.max() <= 1 + 1e-10
        assert y_train.min() >= -1e-10
        assert y_train.max() <= 1 + 1e-10
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data(
                csv_file='nonexistent.csv',
                feature_cols=['Flow_well'],
                label_cols=['Area'],
                plots=False
            )
    
    def test_load_data_invalid_columns(self, sample_csv):
        """Test error handling for invalid column names."""
        with pytest.raises(KeyError):
            load_data(
                csv_file=sample_csv,
                feature_cols=['Invalid_Column'],
                label_cols=['Area'],
                plots=False
            )
    
    def test_load_data_with_plots(self, sample_csv):
        """Test that plots can be generated without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            
            X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
                csv_file=sample_csv,
                feature_cols=feature_cols,
                label_cols=label_cols,
                test_size=0.2,
                plots=True,
                rf=tmpdir
            )
            
            # Check that plot directory was created
            plot_dir = Path(tmpdir) / "plots" / "Area"
            assert plot_dir.exists()
            
            # Check that plots were created
            assert (plot_dir / "before_transform.png").exists()
            assert (plot_dir / "after_log_transform.png").exists()
            assert (plot_dir / "after_scaling.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
