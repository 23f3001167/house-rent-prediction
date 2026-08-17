# tests/test_main.py
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.main import (
    data_inspection,
    select_features_target,
    remove_outliers,
    feature_engineering,
    build_pipeline,
    calculate_classification_metrics,
    save_model
)

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "House_Rent_Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model_tuned.pkl"
SAMPLE_PATH = BASE_DIR / "sample" / "sample_unseen_houses.csv"


class TestDataLoading:
    """Test data loading and inspection."""
    
    def test_data_file_exists(self):
        """Test that the dataset file exists."""
        assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"
    
    def test_data_loads(self):
        """Test that data loads correctly."""
        df = pd.read_csv(DATA_PATH)
        assert df is not None
        assert len(df) > 0
        assert df.shape[1] == 12  # Should have 12 columns


class TestFeatureSelection:
    """Test feature selection functionality."""
    
    def test_feature_selection(self):
        """Test that features are selected correctly."""
        df = pd.read_csv(DATA_PATH)
        X, y = select_features_target(df)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert 'Rent' in df.columns


class TestOutlierRemoval:
    """Test outlier removal functionality."""
    
    def test_outlier_removal_percentile(self):
        """Test percentile-based outlier removal."""
        df = pd.read_csv(DATA_PATH)
        X, y = select_features_target(df)
        X_clean, y_clean = remove_outliers(X, y, method='percentile')
        
        assert X_clean is not None
        assert y_clean is not None
        assert len(X_clean) <= len(X)
        assert len(y_clean) <= len(y)
        assert len(X_clean) == len(y_clean)


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_feature_engineering(self):
        """Test that new features are created."""
        df = pd.read_csv(DATA_PATH)
        X, y = select_features_target(df)
        
        original_count = len(X.columns)
        X_engineered = feature_engineering(X)
        
        assert len(X_engineered.columns) > original_count


class TestPipeline:
    """Test the ML pipeline."""
    
    def test_pipeline_builds(self):
        """Test that pipeline builds successfully."""
        pipeline = build_pipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    def test_pipeline_trains(self):
        """Test that pipeline trains without errors."""
        df = pd.read_csv(DATA_PATH)
        X, y = select_features_target(df)
        X, y = remove_outliers(X, y, method='percentile')
        X = feature_engineering(X)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(y_test)


class TestModelLoading:
    """Test model loading and saving."""
    
    def test_model_exists(self):
        """Test that model file exists."""
        assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
    
    def test_model_loads(self):
        """Test that model loads correctly."""
        model = joblib.load(MODEL_PATH)
        assert model is not None
        assert hasattr(model, 'predict')


class TestPredictions:
    """Test prediction functionality."""
    
    def test_sample_file_exists(self):
        """Test that sample file exists."""
        assert SAMPLE_PATH.exists(), f"Sample file not found at {SAMPLE_PATH}"
    
    def test_sample_loads(self):
        """Test that sample data loads correctly."""
        df = pd.read_csv(SAMPLE_PATH)
        assert df is not None
        assert len(df) > 0
        
        required_columns = ['BHK', 'Size', 'Floor', 'Area Type', 'Area Locality',
                           'City', 'Furnishing Status', 'Tenant Preferred', 
                           'Bathroom', 'Point of Contact']
        for col in required_columns:
            assert col in df.columns
    
    def test_predictions_file_exists(self):
        """Test that predictions file was created."""
        pred_path = BASE_DIR / "sample" / "predictions_output_tuned.csv"
        assert pred_path.exists(), "Predictions file not found"
    
    def test_predictions_valid(self):
        """Test that predictions are valid numbers."""
        pred_path = BASE_DIR / "sample" / "predictions_output_tuned.csv"
        df = pd.read_csv(pred_path)
        
        assert 'Predicted_Rent' in df.columns
        assert df['Predicted_Rent'].notna().all()
        assert (df['Predicted_Rent'] > 0).all()
        assert len(df) > 0


class TestMetrics:
    """Test metric calculations."""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        y_true = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        
        metrics, y_binned, y_pred_binned = calculate_classification_metrics(y_true, y_pred)
        
        assert metrics is not None
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 1.0


class TestOutputs:
    """Test that output files are generated."""
    
    def test_outputs_folder_exists(self):
        """Test that outputs folder exists."""
        output_path = BASE_DIR / "outputs"
        assert output_path.exists(), "Outputs folder not found"
    
    def test_metrics_file_exists(self):
        """Test that metrics file exists."""
        metrics_path = BASE_DIR / "outputs" / "tuned_model_metrics.txt"
        assert metrics_path.exists(), "Metrics file not found"
    
    def test_comparison_file_exists(self):
        """Test that comparison file exists."""
        comparison_path = BASE_DIR / "outputs" / "model_comparison.txt"
        assert comparison_path.exists(), "Comparison file not found"
    
    def test_plot_files_exist(self):
        """Test that plot files exist."""
        output_path = BASE_DIR / "outputs"
        expected_plots = [
            'rent_distribution.png',
            'rent_by_bhk.png',
            'rent_by_city.png',
            'rent_vs_size.png',
            'baseline_model_actual_vs_predicted.png',
            'tuned_model_actual_vs_predicted.png',
            'model_comparison_plot.png'
        ]
        
        for plot in expected_plots:
            plot_path = output_path / plot
            assert plot_path.exists(), f"Plot file not found: {plot}"


class TestDataQuality:
    """Test data quality checks."""
    
    def test_no_missing_values(self):
        """Test that there are no missing values."""
        df = pd.read_csv(DATA_PATH)
        missing = df.isnull().sum()
        assert missing.sum() == 0, f"Missing values found: {missing[missing > 0]}"
    
    def test_rent_positive(self):
        """Test that all rent values are positive."""
        df = pd.read_csv(DATA_PATH)
        assert (df['Rent'] > 0).all(), "Some rent values are not positive"
    
    def test_size_positive(self):
        """Test that all size values are positive."""
        df = pd.read_csv(DATA_PATH)
        assert (df['Size'] > 0).all(), "Some size values are not positive"
    
    def test_bhk_range(self):
        """Test that BHK values are in reasonable range."""
        df = pd.read_csv(DATA_PATH)
        assert (df['BHK'] >= 1).all(), "Some BHK values are less than 1"
        assert (df['BHK'] <= 6).all(), "Some BHK values are greater than 6"