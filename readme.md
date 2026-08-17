# 🏠 House Rent Prediction System

This project predicts house rent prices using machine learning, specifically a Random Forest Regressor. It includes data preprocessing, feature engineering, model training, evaluation, and prediction capabilities.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technical Architecture](#technical-architecture)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Results & Visualizations](#results--visualizations)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Technical Choices](#technical-choices)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview

The **House Rent Prediction System** is an end-to-end machine learning solution that predicts house rental prices based on various property features. Built with Python and scikit-learn, it demonstrates best practices in data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

### 🎯 Key Achievements
- **R² Score**: 0.7619 (76% of variance explained)
- **MAE**: ₹8,647 (Average prediction error)
- **MAPE**: 28.53% (Average percentage error)
- **Accuracy**: 91.55% (Category prediction)
- **Cross-validation R²**: 0.6972 (±0.0237)

---

## ✨ Features

### Core Functionality
- **Data Preprocessing**: Automated cleaning, outlier removal, and feature encoding
- **Feature Engineering**: Creates 8+ derived features for better predictions
- **Model Training**: Random Forest Regressor with hyperparameter optimization
- **Evaluation**: Comprehensive metrics including MAE, RMSE, R², MAPE, and classification metrics
- **Prediction**: Real-time predictions for new/unseen properties
- **Visualization**: 12+ professional plots for analysis and presentation

### Technical Features
- **Reproducible Pipeline**: End-to-end automation from data to predictions
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameters
- **Model Persistence**: Save/load trained models using joblib
- **Unit Testing**: Comprehensive test suite with pytest
- **Documentation**: Complete code documentation and this README

---

## 📊 Dataset

### Source
[India House Rent Prediction Dataset](https://www.kaggle.com/datasets/pranavshinde36/india-house-rent-prediction) from Kaggle

### Dataset Statistics
| Attribute | Value |
|-----------|-------|
| **Total Records** | 4,746 |
| **Features** | 12 |
| **Missing Values** | 0 (Clean dataset) |
| **Target Variable** | Rent (in INR) |
| **Cities Covered** | 6 (Kolkata, Mumbai, Bangalore, Delhi, Chennai, Hyderabad) |

### Features Description
| Feature | Description | Type |
|---------|-------------|------|
| `BHK` | Number of bedrooms | Numerical |
| `Size` | Area in square feet | Numerical |
| `Floor` | Floor location (e.g., "Ground out of 2") | Categorical |
| `Area Type` | Type of area (Super Area, Carpet Area, Built Area) | Categorical |
| `Area Locality` | Specific locality name | Categorical |
| `City` | City name | Categorical |
| `Furnishing Status` | Furnished, Semi-Furnished, Unfurnished | Categorical |
| `Tenant Preferred` | Preferred tenant type | Categorical |
| `Bathroom` | Number of bathrooms | Numerical |
| `Point of Contact` | Contact person type | Categorical |
| `Posted On` | Date of posting | DateTime |
| `Rent` | Monthly rent (Target) | Numerical |

---

## 🏗️ Technical Architecture

### Pipeline Overview
```
CSV Data → Data Inspection → EDA → Feature Engineering → 
Preprocessing → Train/Test Split → Model Training → 
Hyperparameter Tuning → Evaluation → Model Saving → Predictions
```

### Technology Stack
| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Model** | Random Forest Regressor |
| **Tuning** | GridSearchCV/RandomizedSearchCV |
| **Model Persistence** | Joblib |
| **Testing** | Pytest, Pytest-Cov |
| **Environment** | Virtual Environment (venv) |

### Feature Engineering Details
| New Feature | Description | Benefit |
|-------------|-------------|---------|
| `Floor_Number` | Extracted numeric floor from "Floor" | Better floor representation |
| `Total_Floors` | Total floors in building | Height indicator |
| `Is_Ground` | Binary (1 if ground floor) | Ground floor premium |
| `Floor_Ratio` | Floor number / Total floors | Relative position |
| `Size_per_BHK` | Size / (BHK + 1) | Space efficiency |
| `Room_Bathroom_Ratio` | BHK / (Bathroom + 1) | Room distribution |
| `Log_Size` | Log transformation of Size | Normalize distribution |
| `Size_Squared` | Size² | Capture non-linearity |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Project
```bash

# Or download and extract the ZIP file
cd house-rent-prediction
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install pandas numpy matplotlib seaborn scikit-learn joblib pytest pytest-cov
```

### Step 4: Download Dataset
1. Download from [Kaggle](https://www.kaggle.com/datasets/pranavshinde36/india-house-rent-prediction)
2. Place it in `data/raw/House_Rent_Dataset.csv`

### Step 5: Verify Installation
```bash
# Test imports
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib; print('✓ All packages installed successfully!')"
```

---

## 📖 Usage Guide

### 1. Run the Complete Pipeline
```bash
# From the project root directory
python src/main.py
```

**What happens:**
1. Loads and inspects the dataset
2. Performs Exploratory Data Analysis (EDA)
3. Selects features and removes outliers
4. Engineers new features
5. Splits data into train/test sets
6. Builds and trains baseline Random Forest model
7. Performs hyperparameter tuning
8. Evaluates both models
9. Saves the best model
10. Generates predictions for sample houses
11. Creates visualizations and metrics reports

### 2. Make Predictions on New Data

**Option A: Using the saved model**
```python
# predict.py
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/random_forest_model_tuned.pkl')

# Load your data
new_data = pd.read_csv('your_new_houses.csv')

# Apply same feature engineering (use the function from main.py)
from src.main import feature_engineering
new_data_engineered = feature_engineering(new_data)

# Make predictions
predictions = model.predict(new_data_engineered)
new_data['Predicted_Rent'] = predictions

# Save results
new_data.to_csv('predictions.csv', index=False)
```

**Option B: Using the sample file**
1. Edit `sample/sample_unseen_houses.csv` with your data
2. Run `python src/main.py`
3. Check `sample/predictions_output_tuned.csv` for results

### 3. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_main.py::TestDataLoading::test_data_file_exists -v
```

### 4. Check Results
All outputs are saved in the `outputs/` folder:
- **Visualizations**: PNG files (12+ plots)
- **Metrics**: Text files with all evaluation metrics
- **Model Comparison**: Side-by-side comparison of baseline vs tuned model

---

## 📊 Model Performance

### Baseline vs Tuned Model Comparison
| Metric | Baseline | Tuned | Improvement |
|--------|----------|-------|-------------|
| **R² Score** | 0.7530 | **0.7619** | ✅ +1.2% |
| **MAE** | ₹9,427 | **₹8,647** | ✅ **-8.3%** |
| **RMSE** | ₹20,588 | **₹20,214** | ✅ -1.8% |
| **MAPE** | 34.06% | **28.53%** | ✅ **-16.2%** |
| **Accuracy** | 90.48% | **91.55%** | ✅ +1.1% |

### Detailed Regression Metrics (Tuned Model)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.7619 | Model explains 76% of variance |
| **MAE** | ₹8,647 | Average error ~₹8.6K |
| **RMSE** | ₹20,214 | Larger errors penalized |
| **MAPE** | 28.53% | Average error ~28.5% |
| **Explained Variance** | 0.7621 | Similar to R² |
| **Max Error** | ₹195,953 | Worst prediction |
| **Median AE** | ₹3,259 | Half predictions within ₹3.2K |

### Classification Metrics (Tuned Model)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 91.55% | 91.55% correct predictions |
| **Precision (weighted)** | 92.93% | Good at avoiding false positives |
| **Recall (weighted)** | 91.55% | Good at finding true positives |
| **F1-Score (weighted)** | 92.12% | Good overall balance |

### Cross-Validation Results
| Metric | Value |
|--------|-------|
| **CV R² Scores** | [0.6807, 0.6711, 0.7111, 0.6866, 0.7364] |
| **Mean CV R²** | 0.6972 |
| **Std CV R²** | 0.0237 |

### Optimal Hyperparameters (Found via RandomizedSearchCV)
| Parameter | Optimal Value |
|-----------|---------------|
| `n_estimators` | 300 |
| `max_depth` | 20 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `max_features` | None |
| **Best CV R²** | 0.7902 |

---

## 📈 Results & Visualizations

### Generated Visualizations

| Visualization | Description |
|---------------|-------------|
| `rent_distribution.png` | Distribution of rent prices (histogram + KDE) |
| `rent_by_bhk.png` | Box plot showing rent by number of BHK |
| `rent_by_city.png` | Box plot showing rent by city |
| `rent_by_furnishing.png` | Box plot showing rent by furnishing status |
| `correlation_matrix.png` | Heatmap of feature correlations |
| `rent_vs_size.png` | Scatter plot of rent vs property size |
| `baseline_actual_vs_predicted.png` | Actual vs predicted for baseline model |
| `baseline_residual_plot.png` | Residual analysis for baseline |
| `baseline_feature_importance.png` | Top 20 features importance (baseline) |
| `tuned_actual_vs_predicted.png` | Actual vs predicted for tuned model |
| `tuned_residual_plot.png` | Residual analysis for tuned |
| `tuned_feature_importance.png` | Top 20 features importance (tuned) |
| `model_comparison_plot.png` | Side-by-side comparison |

### Sample Predictions
| BHK | Size | City | Predicted Rent |
|-----|------|------|----------------|
| 2 | 850 | Mumbai | ₹46,280 |
| 3 | 1500 | Bangalore | ₹58,139 |
| 1 | 550 | Pune | ₹7,456 |
| 2 | 1200 | Chennai | ₹44,413 |
| 3 | 1800 | Delhi | ₹32,711 |
| 1 | 450 | Hyderabad | ₹9,520 |
| 2 | 750 | Kolkata | ₹10,680 |
| 2 | 800 | Ahmedabad | ₹11,480 |

---

## 🧪 Testing

### Test Coverage
The project includes comprehensive unit tests covering:
- **Data Loading** - Verifies dataset existence and loading
- **Feature Selection** - Tests feature selection logic
- **Outlier Removal** - Tests both IQR and percentile methods
- **Feature Engineering** - Verifies new features are created
- **Pipeline** - Tests pipeline building and training
- **Model Loading** - Tests model save/load functionality
- **Predictions** - Tests prediction generation
- **Metrics** - Tests metric calculations
- **Outputs** - Verifies all output files are generated
- **Data Quality** - Tests for missing values, positive values, ranges

### Running Tests
```bash
# All tests with verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test class
pytest tests/test_main.py::TestDataQuality -v

# Run specific test
pytest tests/test_main.py::TestDataQuality::test_no_missing_values -v
```

### Expected Test Results
```
collected 20 items
✅ All 20 tests passed!
```

---

## 📂 Project Structure

```
house-rent-prediction/
│
├── data/
│   └── raw/
│       └── House_Rent_Dataset.csv          # Original dataset
├── models/
│   ├── random_forest_baseline.pkl          # Baseline model
│   └── random_forest_model_tuned.pkl       # Best tuned model
├── outputs/
│   ├── rent_distribution.png               # EDA Visualizations
│   ├── rent_by_bhk.png
│   ├── rent_by_city.png
│   ├── rent_by_furnishing.png
│   ├── correlation_matrix.png
│   ├── rent_vs_size.png
│   ├── baseline_model_actual_vs_predicted.png
│   ├── baseline_model_residual_plot.png
│   ├── baseline_model_residual_distribution.png
│   ├── baseline_model_feature_importance.png
│   ├── baseline_model_error_by_bhk.png
│   ├── baseline_model_metrics.txt
│   ├── tuned_model_actual_vs_predicted.png
│   ├── tuned_model_residual_plot.png
│   ├── tuned_model_residual_distribution.png
│   ├── tuned_model_feature_importance.png
│   ├── tuned_model_error_by_bhk.png
│   ├── tuned_model_metrics.txt
│   ├── model_comparison_plot.png
│   └── model_comparison.txt
├── sample/
│   ├── sample_unseen_houses.csv            # Sample input data
│   └── predictions_output_tuned.csv        # Sample predictions
├── src/
│   └── main.py                             # Main pipeline script
├── tests/
│   └── test_main.py                        # Unit tests
├── venv/                                   # Virtual environment
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── .gitignore                              # Git ignore file
└── LICENSE                                 # MIT License
```

---

## 🔧 Technical Choices

### Why Random Forest?
1. **Handles Non-linearity**: Captures complex relationships between features
2. **Feature Importance**: Provides interpretability (what features matter most)
3. **Robust to Outliers**: Less sensitive to outliers than linear models
4. **Handles Mixed Data**: Works with both numerical and categorical features
5. **No Scaling Required**: Tree-based models don't require feature scaling
6. **Excellent Performance**: Consistently top-performing for tabular data

### Why These Features?
1. **Floor_Number & Total_Floors**: Floor level affects rent (higher floors often premium)
2. **Is_Ground**: Ground floor properties have different pricing
3. **Size_per_BHK**: Better indicator than raw size (efficient use of space)
4. **Room_Bathroom_Ratio**: Indicates comfort level
5. **Log_Size**: Handles right-skewed distribution of property sizes
6. **Size_Squared**: Captures non-linear relationship between size and rent

### Preprocessing Pipeline
- **Categorical Features**: One-Hot Encoding
- **Numerical Features**: Standard Scaling (for algorithms that need it)
- **Outlier Removal**: Top 1% and bottom 1% percentile clipping
- **Missing Values**: None found, but pipeline handles if present

---


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `pytest tests/ -v`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Style
- Follow PEP 8 guidelines
- Write docstrings for all functions
- Add unit tests for new functionality
- Update README if needed

---


## ⚡ Quick Reference

### Common Commands
```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python src/main.py

# Run tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

# Make predictions (using saved model)
python predict.py  # Create this script
```

### Key Files
| File | Purpose |
|------|---------|
| `src/main.py` | Main pipeline script |
| `tests/test_main.py` | Unit tests |
| `requirements.txt` | Dependencies |
| `models/random_forest_model_tuned.pkl` | Best model |
| `outputs/tuned_model_metrics.txt` | All metrics |
| `sample/predictions_output_tuned.csv` | Predictions |

---
