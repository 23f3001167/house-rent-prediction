# src/main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    median_absolute_error
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "House_Rent_Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model_tuned.pkl"
OUTPUT_PATH = BASE_DIR / "outputs"
SAMPLE_PATH = BASE_DIR / "sample" / "sample_unseen_houses.csv"


def data_inspection(df):
    """Perform initial data inspection."""
    print("=" * 60)
    print("DATA INSPECTION REPORT")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Statistical Summary ---")
    print(df.describe(include='all'))
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print("\n--- Unique Values in Categorical Columns ---")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        if df[col].nunique() <= 10:
            print(f"  Values: {df[col].unique()}")
    
    return df

def exploratory_data_analysis(df, output_path):
    """Perform EDA and save visualizations."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rent'], bins=50, kde=True)
    plt.title('Distribution of Rent Prices', fontsize=14)
    plt.xlabel('Rent (in INR)')
    plt.ylabel('Frequency')
    plt.savefig(output_path / 'rent_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: rent_distribution.png")
    
    # 2. Rent by BHK
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='BHK', y='Rent')
    plt.title('Rent Distribution by BHK', fontsize=14)
    plt.xlabel('BHK')
    plt.ylabel('Rent (in INR)')
    plt.savefig(output_path / 'rent_by_bhk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: rent_by_bhk.png")
    
    # 3. Rent by City
    plt.figure(figsize=(12, 6))
    top_cities = df['City'].value_counts().head(10).index
    filtered_df = df[df['City'].isin(top_cities)]
    sns.boxplot(data=filtered_df, x='City', y='Rent')
    plt.title('Rent Distribution by City', fontsize=14)
    plt.xticks(rotation=45)
    plt.xlabel('City')
    plt.ylabel('Rent (in INR)')
    plt.tight_layout()
    plt.savefig(output_path / 'rent_by_city.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: rent_by_city.png")
    
    # 4. Rent by Furnishing Status
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Furnishing Status', y='Rent')
    plt.title('Rent Distribution by Furnishing Status', fontsize=14)
    plt.xlabel('Furnishing Status')
    plt.ylabel('Rent (in INR)')
    plt.savefig(output_path / 'rent_by_furnishing.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: rent_by_furnishing.png")
    
    # 5. Correlation matrix
    numerical_cols = ['BHK', 'Rent', 'Size', 'Bathroom']
    if all(col in df.columns for col in numerical_cols):
        plt.figure(figsize=(8, 6))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created: correlation_matrix.png")
    
    # 6. Rent vs Size
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Size'], df['Rent'], alpha=0.5)
    plt.title('Rent vs Size', fontsize=14)
    plt.xlabel('Size (sq ft)')
    plt.ylabel('Rent (in INR)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'rent_vs_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: rent_vs_size.png")

def select_features_target(df):
    """Select features and target for modeling."""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)
    
    features = ['BHK', 'Size', 'Floor', 'Area Type', 'Area Locality', 'City', 
                'Furnishing Status', 'Tenant Preferred', 'Bathroom', 'Point of Contact']
    target = 'Rent'
    
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    X = df[available_features]
    y = df[target]
    
    print(f"Features ({len(X.columns)}): {X.columns.tolist()}")
    print(f"Target: {target}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"Target mean: {y.mean():.2f}")
    print(f"Target median: {y.median():.2f}")
    
    return X, y

def remove_outliers(df, y, method='iqr'):
    """Remove outliers from the dataset."""
    print("\n" + "=" * 60)
    print("OUTLIER REMOVAL")
    print("=" * 60)
    
    original_shape = df.shape
    print(f"Original shape: {original_shape}")
    
    if method == 'iqr':
        # Remove outliers from Rent (target variable)
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove extreme outliers (beyond 3*IQR for more conservative)
        extreme_upper = Q3 + 3 * IQR
        
        # Keep only data within bounds
        mask = (y <= extreme_upper) & (y >= lower_bound)
        df_clean = df[mask]
        y_clean = y[mask]
        
        print(f"Removed {(~mask).sum()} outliers ({(~mask).sum()/len(y)*100:.1f}%)")
        print(f"Rent range after removal: {y_clean.min():.2f} - {y_clean.max():.2f}")
    
    elif method == 'percentile':
        # Remove top 1% and bottom 1%
        lower_percentile = y.quantile(0.01)
        upper_percentile = y.quantile(0.99)
        mask = (y >= lower_percentile) & (y <= upper_percentile)
        df_clean = df[mask]
        y_clean = y[mask]
        
        print(f"Removed {(~mask).sum()} outliers ({(~mask).sum()/len(y)*100:.1f}%)")
        print(f"Rent range after removal: {y_clean.min():.2f} - {y_clean.max():.2f}")
    
    return df_clean, y_clean

def feature_engineering(X):
    """Perform feature engineering."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    X_engineered = X.copy()
    
    # 1. Extract floor number from Floor column
    if 'Floor' in X_engineered.columns:
        def extract_floor_number(floor_str):
            if pd.isna(floor_str):
                return 0
            floor_str = str(floor_str).lower()
            if 'ground' in floor_str:
                return 0
            elif 'basement' in floor_str:
                return -1
            else:
                import re
                numbers = re.findall(r'\d+', floor_str)
                if numbers:
                    return int(numbers[0])
                return 0
        
        X_engineered['Floor_Number'] = X_engineered['Floor'].apply(extract_floor_number)
        print("✓ Created: Floor_Number (extracted from Floor)")
        
        def extract_total_floors(floor_str):
            if pd.isna(floor_str):
                return 0
            floor_str = str(floor_str).lower()
            import re
            numbers = re.findall(r'\d+', floor_str)
            if len(numbers) >= 2:
                return int(numbers[1])
            elif 'ground' in floor_str and len(numbers) >= 1:
                return int(numbers[0])
            return 0
        
        X_engineered['Total_Floors'] = X_engineered['Floor'].apply(extract_total_floors)
        print("✓ Created: Total_Floors (extracted from Floor)")
        
        # Is Ground Floor?
        X_engineered['Is_Ground'] = X_engineered['Floor'].str.lower().str.contains('ground', case=False).astype(int)
        print("✓ Created: Is_Ground")
        
        # Floor ratio (floor number / total floors)
        X_engineered['Floor_Ratio'] = X_engineered['Floor_Number'] / (X_engineered['Total_Floors'] + 1)
        print("✓ Created: Floor_Ratio")
    
    # 2. Area per room
    if 'Size' in X_engineered.columns and 'BHK' in X_engineered.columns:
        X_engineered['Size_per_BHK'] = X_engineered['Size'] / (X_engineered['BHK'] + 1)
        print("✓ Created: Size_per_BHK")
    
    # 3. Room to bathroom ratio
    if 'BHK' in X_engineered.columns and 'Bathroom' in X_engineered.columns:
        X_engineered['Room_Bathroom_Ratio'] = X_engineered['BHK'] / (X_engineered['Bathroom'] + 1)
        print("✓ Created: Room_Bathroom_Ratio")
    
    # 4. Log transform of Size
    if 'Size' in X_engineered.columns:
        X_engineered['Log_Size'] = np.log1p(X_engineered['Size'])
        print("✓ Created: Log_Size")
    
    # 5. Size squared (polynomial feature)
    if 'Size' in X_engineered.columns:
        X_engineered['Size_Squared'] = X_engineered['Size'] ** 2
        print("✓ Created: Size_Squared")
    
    print(f"New feature count: {len(X_engineered.columns)} (was {len(X.columns)})")
    return X_engineered

def build_pipeline():
    """Build the preprocessing and modeling pipeline."""
    print("\n" + "=" * 60)
    print("BUILDING ML PIPELINE")
    print("=" * 60)
    
    categorical_features = ['Area Type', 'Area Locality', 'City', 'Furnishing Status', 
                           'Tenant Preferred', 'Point of Contact']
    numerical_features = ['BHK', 'Size', 'Bathroom', 'Floor_Number', 'Total_Floors', 
                         'Size_per_BHK', 'Room_Bathroom_Ratio', 'Log_Size', 'Size_Squared',
                         'Is_Ground', 'Floor_Ratio']
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("✓ Pipeline built successfully")
    print(f"  Numerical features: {numerical_features}")
    print(f"  Categorical features: {categorical_features}")
    print(f"  Model: RandomForestRegressor (initial parameters)")
    
    return model

def hyperparameter_tuning(pipeline, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Define parameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [10, 15, 20, 25, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2', None]
    }
    
    print("Parameter grid:")
    for key, value in param_grid.items():
        print(f"  {key}: {value}")
    
    # Random search with 10 combinations
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,  # Try 20 random combinations
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nStarting hyperparameter tuning (this may take a few minutes)...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"✓ Best cross-validation R² score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def calculate_classification_metrics(y_true, y_pred, n_bins=5):
    """Calculate classification metrics by binning rent values."""
    # Create bins for classification
    y_binned = pd.cut(y_true, bins=n_bins, labels=False)
    y_pred_binned = pd.cut(y_pred, bins=n_bins, labels=False)
    
    metrics = {
        'accuracy': accuracy_score(y_binned, y_pred_binned),
        'precision_macro': precision_score(y_binned, y_pred_binned, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_binned, y_pred_binned, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_binned, y_pred_binned, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_binned, y_pred_binned, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_binned, y_pred_binned, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_binned, y_pred_binned, average='weighted', zero_division=0)
    }
    
    return metrics, y_binned, y_pred_binned

def evaluate_model(model, X_test, y_test, output_path, model_name="Model"):
    """Evaluate the model with comprehensive metrics."""
    print("\n" + "=" * 60)
    print(f"{model_name} EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    explained_var = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    
    # Print all metrics
    print("\n--- REGRESSION METRICS ---")
    print(f"MAE (Mean Absolute Error):                ₹{mae:,.2f}")
    print(f"RMSE (Root Mean Squared Error):           ₹{rmse:,.2f}")
    print(f"R² Score:                                 {r2:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error):    {mape:.2f}%")
    print(f"Explained Variance Score:                 {explained_var:.4f}")
    print(f"Max Error:                                ₹{max_err:,.2f}")
    print(f"Median Absolute Error:                    ₹{median_ae:,.2f}")
    
    # Calculate classification metrics (binned rent values)
    print("\n--- CLASSIFICATION METRICS (Binned Rent) ---")
    class_metrics, y_binned, y_pred_binned = calculate_classification_metrics(y_test, y_pred)
    print(f"Accuracy (macro):                         {class_metrics['accuracy']:.4f}")
    print(f"Precision (macro):                        {class_metrics['precision_macro']:.4f}")
    print(f"Precision (weighted):                     {class_metrics['precision_weighted']:.4f}")
    print(f"Recall (macro):                           {class_metrics['recall_macro']:.4f}")
    print(f"Recall (weighted):                        {class_metrics['recall_weighted']:.4f}")
    print(f"F1-Score (macro):                         {class_metrics['f1_macro']:.4f}")
    print(f"F1-Score (weighted):                      {class_metrics['f1_weighted']:.4f}")
    
    # Cross-validation scores
    print("\n--- CROSS-VALIDATION SCORES ---")
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
    print(f"CV R² Scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f}")
    print(f"Std CV R²: {cv_scores.std():.4f}")
    
    # Save all metrics to file
    with open(output_path / f'{model_name.lower().replace(" ", "_")}_metrics.txt', 'w') as f:
        f.write(f"{model_name} EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("REGRESSION METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"MAE (Mean Absolute Error):                Rs.{mae:,.2f}\n")
        f.write(f"RMSE (Root Mean Squared Error):           Rs.{rmse:,.2f}\n")
        f.write(f"R² Score:                                 {r2:.4f}\n")
        f.write(f"MAPE (Mean Absolute Percentage Error):    {mape:.2f}%\n")
        f.write(f"Explained Variance Score:                 {explained_var:.4f}\n")
        f.write(f"Max Error:                                Rs.{max_err:,.2f}\n")
        f.write(f"Median Absolute Error:                    Rs.{median_ae:,.2f}\n\n")
        
        f.write("CLASSIFICATION METRICS (Binned Rent)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy (macro):                         {class_metrics['accuracy']:.4f}\n")
        f.write(f"Precision (macro):                        {class_metrics['precision_macro']:.4f}\n")
        f.write(f"Precision (weighted):                     {class_metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (macro):                           {class_metrics['recall_macro']:.4f}\n")
        f.write(f"Recall (weighted):                        {class_metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (macro):                         {class_metrics['f1_macro']:.4f}\n")
        f.write(f"F1-Score (weighted):                      {class_metrics['f1_weighted']:.4f}\n\n")
        
        f.write("CROSS-VALIDATION SCORES\n")
        f.write("-" * 30 + "\n")
        f.write(f"CV R² Scores: {cv_scores}\n")
        f.write(f"Mean CV R²: {cv_scores.mean():.4f}\n")
        f.write(f"Std CV R²: {cv_scores.std():.4f}\n")
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Rent (INR)', fontsize=12)
    plt.ylabel('Predicted Rent (INR)', fontsize=12)
    plt.title(f'{model_name}: Actual vs Predicted Rent (R² = {r2:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / f'{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Created: {model_name.lower().replace(' ', '_')}_actual_vs_predicted.png")
    
    # 2. Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, s=10)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Rent (INR)', fontsize=12)
    plt.ylabel('Residuals (INR)', fontsize=12)
    plt.title(f'{model_name}: Residual Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / f'{model_name.lower().replace(" ", "_")}_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: {model_name.lower().replace(' ', '_')}_residual_plot.png")
    
    # 3. Residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel('Residuals (INR)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name}: Distribution of Residuals', fontsize=14)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.savefig(output_path / f'{model_name.lower().replace(" ", "_")}_residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: {model_name.lower().replace(' ', '_')}_residual_distribution.png")
    
    # 4. Feature importance (if Random Forest)
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = get_feature_names(model.named_steps['preprocessor'])
        importances = model.named_steps['regressor'].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{model_name}: Top 20 Feature Importances', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path / f'{model_name.lower().replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Created: {model_name.lower().replace(' ', '_')}_feature_importance.png")
        
        # Save feature importance to file
        feature_importance_df.to_csv(output_path / f'{model_name.lower().replace(" ", "_")}_feature_importance.csv', index=False)
    
    # 5. Error distribution by BHK
    if 'BHK' in X_test.columns:
        error_df = pd.DataFrame({
            'BHK': X_test['BHK'],
            'Error': residuals
        })
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=error_df, x='BHK', y='Error')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('BHK', fontsize=12)
        plt.ylabel('Prediction Error (INR)', fontsize=12)
        plt.title(f'{model_name}: Prediction Error by BHK', fontsize=14)
        plt.savefig(output_path / f'{model_name.lower().replace(" ", "_")}_error_by_bhk.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Created: {model_name.lower().replace(' ', '_')}_error_by_bhk.png")
    
    return y_pred

def get_feature_names(preprocessor):
    """Get feature names after preprocessing."""
    feature_names = []
    
    # Numerical features
    for transformer in preprocessor.transformers_:
        if transformer[0] == 'num':
            num_features = transformer[2]
            feature_names.extend(num_features)
        elif transformer[0] == 'cat':
            cat_transformer = transformer[1]
            cat_features = transformer[2]
            
            if hasattr(cat_transformer, 'named_steps'):
                encoder = cat_transformer.named_steps['onehot']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_names = encoder.get_feature_names_out(cat_features)
                    feature_names.extend(cat_names)
    
    return feature_names

def save_model(model, path):
    """Save the trained model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\n✓ Model saved to: {path}")
    
def load_model(path):
    """Load a saved model."""
    if not path.exists():
        print(f"✗ Model file not found: {path}")
        return None
    model = joblib.load(path)
    print(f"✓ Model loaded from: {path}")
    return model

def predict_unseen_houses(model, sample_path):
    """Predict rent for unseen houses."""
    print("\n" + "=" * 60)
    print("PREDICTING UNSEEN HOUSES")
    print("=" * 60)
    
    if not sample_path.exists():
        print(f"✗ Sample file not found: {sample_path}")
        return None
    
    sample_df = pd.read_csv(sample_path)
    print(f"✓ Loaded {len(sample_df)} samples from: {sample_path}")
    print("\nSample data preview:")
    print(sample_df.head())
    
    # Apply feature engineering
    sample_engineered = feature_engineering(sample_df)
    
    # Make predictions
    predictions = model.predict(sample_engineered)
    sample_df['Predicted_Rent'] = predictions
    
    # Add confidence intervals
    if hasattr(model.named_steps['regressor'], 'estimators_'):
        tree_predictions = np.array([
            tree.predict(model.named_steps['preprocessor'].transform(sample_engineered))
            for tree in model.named_steps['regressor'].estimators_
        ])
        sample_df['Prediction_Std'] = tree_predictions.std(axis=0)
        sample_df['Lower_Bound'] = sample_df['Predicted_Rent'] - 1.96 * sample_df['Prediction_Std']
        sample_df['Upper_Bound'] = sample_df['Predicted_Rent'] + 1.96 * sample_df['Prediction_Std']
    
    # Save predictions
    output_path = BASE_DIR / "sample" / "predictions_output_tuned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    
    print("\nPredictions (first 10):")
    print(sample_df[['BHK', 'Size', 'City', 'Predicted_Rent']].head(10))
    print(f"\n✓ Predictions saved to: {output_path}")
    
    return sample_df

def compare_models(baseline_model, tuned_model, X_test, y_test, output_path):
    """Compare baseline and tuned models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Baseline predictions
    baseline_pred = baseline_model.predict(X_test)
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_mape = mean_absolute_percentage_error(y_test, baseline_pred) * 100
    
    # Tuned predictions
    tuned_pred = tuned_model.predict(X_test)
    tuned_r2 = r2_score(y_test, tuned_pred)
    tuned_mae = mean_absolute_error(y_test, tuned_pred)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
    tuned_mape = mean_absolute_percentage_error(y_test, tuned_pred) * 100
    
    print("\n--- COMPARISON TABLE ---")
    print(f"{'Metric':<20} {'Baseline':<15} {'Tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'R² Score':<20} {baseline_r2:<15.4f} {tuned_r2:<15.4f} {(tuned_r2 - baseline_r2) / baseline_r2 * 100:>+10.1f}%")
    print(f"{'MAE':<20} Rs.{baseline_mae:<13,.0f} Rs.{tuned_mae:<13,.0f} {(tuned_mae - baseline_mae) / baseline_mae * 100:>+10.1f}%")
    print(f"{'RMSE':<20} Rs.{baseline_rmse:<13,.0f} Rs.{tuned_rmse:<13,.0f} {(tuned_rmse - baseline_rmse) / baseline_rmse * 100:>+10.1f}%")
    print(f"{'MAPE':<20} {baseline_mape:<14.2f}% {tuned_mape:<14.2f}% {(tuned_mape - baseline_mape) / baseline_mape * 100:>+10.1f}%")
    
    # Save comparison
    with open(output_path / 'model_comparison.txt', 'w') as f:
        f.write("MODEL COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metric':<20} {'Baseline':<15} {'Tuned':<15} {'Improvement':<15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'R² Score':<20} {baseline_r2:<15.4f} {tuned_r2:<15.4f} {(tuned_r2 - baseline_r2) / baseline_r2 * 100:>+10.1f}%\n")
        f.write(f"{'MAE':<20} Rs.{baseline_mae:<13,.0f} Rs.{tuned_mae:<13,.0f} {(tuned_mae - baseline_mae) / baseline_mae * 100:>+10.1f}%\n")
        f.write(f"{'RMSE':<20} Rs.{baseline_rmse:<13,.0f} Rs.{tuned_rmse:<13,.0f} {(tuned_rmse - baseline_rmse) / baseline_rmse * 100:>+10.1f}%\n")
        f.write(f"{'MAPE':<20} {baseline_mape:<14.2f}% {tuned_mape:<14.2f}% {(tuned_mape - baseline_mape) / baseline_mape * 100:>+10.1f}%\n")
    
    # Visual comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, baseline_pred, alpha=0.5, s=10)
    min_val = min(y_test.min(), baseline_pred.min())
    max_val = max(y_test.max(), baseline_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Rent (INR)', fontsize=10)
    plt.ylabel('Predicted Rent (INR)', fontsize=10)
    plt.title(f'Baseline Model\nR² = {baseline_r2:.3f}', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, tuned_pred, alpha=0.5, s=10)
    min_val = min(y_test.min(), tuned_pred.min())
    max_val = max(y_test.max(), tuned_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Rent (INR)', fontsize=10)
    plt.ylabel('Predicted Rent (INR)', fontsize=10)
    plt.title(f'Tuned Model\nR² = {tuned_r2:.3f}', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Created: model_comparison_plot.png")
    
    return baseline_pred, tuned_pred

def main():
    """Main execution function."""
    print("=" * 60)
    print("HOUSE RENT PREDICTION SYSTEM (WITH TUNING)")
    print("=" * 60)
    print(f"Project Directory: {BASE_DIR}")
    
    if not DATA_PATH.exists():
        print(f"\n✗ Data file not found: {DATA_PATH}")
        return
    
    # 1. Load data
    print(f"\n✓ Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Data inspection
    df = data_inspection(df)
    
    # 3. EDA
    exploratory_data_analysis(df, OUTPUT_PATH)
    
    # 4. Select features and target
    X, y = select_features_target(df)
    
    # 5. Remove outliers
    X, y = remove_outliers(X, y, method='percentile')
    
    # 6. Feature engineering
    X = feature_engineering(X)
    
    # 7. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n✓ Train/Test split: {len(X_train)} train, {len(X_test)} test samples")
    
    # 8. Build baseline pipeline
    baseline_pipeline = build_pipeline()
    print("\nTraining baseline model...")
    baseline_pipeline.fit(X_train, y_train)
    print("✓ Baseline model trained successfully")
    
    # 9. Evaluate baseline model
    baseline_pred = evaluate_model(baseline_pipeline, X_test, y_test, OUTPUT_PATH, "Baseline_Model")
    
    # 10. Save baseline model
    baseline_path = BASE_DIR / "models" / "random_forest_baseline.pkl"
    save_model(baseline_pipeline, baseline_path)
    
    # 11. Hyperparameter tuning
    tuned_pipeline = hyperparameter_tuning(baseline_pipeline, X_train, y_train)
    
    # 12. Evaluate tuned model
    tuned_pred = evaluate_model(tuned_pipeline, X_test, y_test, OUTPUT_PATH, "Tuned_Model")
    
    # 13. Save tuned model
    save_model(tuned_pipeline, MODEL_PATH)
    
    # 14. Compare models
    compare_models(baseline_pipeline, tuned_pipeline, X_test, y_test, OUTPUT_PATH)
    
    # 15. Predict unseen houses with tuned model
    predict_unseen_houses(tuned_pipeline, SAMPLE_PATH)
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated outputs:")
    print(f"  • Baseline Model: {baseline_path}")
    print(f"  • Tuned Model: {MODEL_PATH}")
    print(f"  • Evaluation plots: {OUTPUT_PATH}/")
    print(f"  • Evaluation metrics: {OUTPUT_PATH}/")
    print(f"  • Model comparison: {OUTPUT_PATH}/model_comparison.txt")
    print(f"  • Predictions: {BASE_DIR}/sample/predictions_output_tuned.csv")

if __name__ == "__main__":
    main()