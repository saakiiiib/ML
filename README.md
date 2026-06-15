# Machine Learning - Random Forest Framework

## Overview

This project implements a **Random Forest Regression framework** for analyzing the **CsSnI3 perovskite dataset**. The framework trains models to predict photovoltaic performance metrics (Voc, Jsc, FF, PCE) based on material properties.

## Features

- **Automated preprocessing**: Log transformation of doping and defect densities
- **One-hot encoding**: Material categorization
- **Zero-variance feature detection**: Automatic removal of invariant features
- **Hyperparameter optimization**: RandomizedSearchCV with KFold cross-validation
- **Comprehensive evaluation**: R², RMSE, MAE metrics
- **Feature importance analysis**: Using Random Forest built-in importance
- **SHAP interpretability**: TreeExplainer for model explanation
- **Visualization**: Feature importance plots, parity plots, SHAP summary plots

## Requirements

```bash
pip install pandas numpy scikit-learn shap matplotlib
```

## Dataset

Expected CSV file: `CsSnI3_dataset.csv`

### Required Columns

- **Features**:
  - `Thickness` (float)
  - `DopingDensity` (float)
  - `DefectDensity` (float)
  - `Material` (categorical)

- **Targets**:
  - `Voc` (open circuit voltage)
  - `Jsc` (short circuit current density)
  - `FF` (fill factor)
  - `PCE` (power conversion efficiency)

## Configuration

Edit the `CONFIG` dictionary in `rf_framework.py`:

```python
CONFIG = {
    "dataset_path": "CsSnI3_dataset.csv",
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "search_iterations": 30,
    "output_file": "results_random_forest.csv"
}
```

## Usage

```bash
python rf_framework.py
```

## Outputs

1. **results_random_forest.csv** - Summary table with all metrics
2. **{target}_feature_importance.png** - Feature importance visualizations
3. **{target}_parity_plot.png** - Actual vs Predicted plots
4. **{target}_shap_summary.png** - SHAP summary plots

## Workflow

1. Load and preprocess dataset
2. Apply log transformations
3. One-hot encode material types
4. Detect and remove zero-variance features
5. Train RandomForest with hyperparameter search (RandomizedSearchCV)
6. Evaluate on train/test sets
7. Generate feature importance plots
8. Generate parity plots
9. Compute and visualize SHAP values

## Output Metrics

For each target variable (Voc, Jsc, FF, PCE):

- **Train R²**: Training set coefficient of determination
- **Test R²**: Testing set coefficient of determination
- **Train RMSE**: Training set root mean squared error
- **Test RMSE**: Testing set root mean squared error
- **Train MAE**: Training set mean absolute error
- **Test MAE**: Testing set mean absolute error
- **Best Parameters**: Optimal hyperparameter configuration

## Notes

- Zero-variance features are automatically removed before training
- Cross-validation uses 5 folds with shuffling
- Hyperparameter search uses 30 random iterations
- All plots are saved as high-resolution PNG files (300 dpi)
