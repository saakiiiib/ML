# Machine Learning Framework for Photovoltaic Material Property Prediction

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![Status](https://img.shields.io/badge/status-active-green.svg)]

A modular machine learning framework for photovoltaic material property prediction using supervised regression algorithms. This repository provides reusable components and model implementations so researchers and engineers can train, evaluate, and explain regression models for photovoltaic material properties using tabular datasets.

The repository currently contains Random Forest and XGBoost implementations. It is designed to work with any tabular regression dataset that satisfies the required column format. Additional models can easily be added in the future.

---

## Table of contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Available Models](#available-models)
  - [Random Forest](#random-forest)
  - [XGBoost](#xgboost)
- [Dataset Requirements](#dataset-requirements)
- [Required Dataset Format](#required-dataset-format)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Adding New Models](#adding-new-models)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Modular machine learning framework
- Multiple regression models
- Shared dataset interface
- Automatic preprocessing
- Hyperparameter optimization
- Cross-validation
- Model evaluation
- Feature importance analysis
- SHAP explainability (where supported)
- Visualization utilities
- Extensible architecture

---

## Repository Structure

ML/
│
├── README.md
├── rf_framework.py
├── xgb_framework.py
├── datasets/
│   └── <your_dataset>.csv
└── outputs/

---

## Available Models

### Random Forest

Features
- Random Forest Regressor
- Automatic preprocessing (shared dataset interface)
- Feature encoding for categorical inputs
- Hyperparameter optimization
- Cross-validation
- Feature importance
- SHAP analysis (when SHAP is available and enabled)
- Performance visualization

Hyperparameter Optimization

RandomizedSearchCV is used to perform efficient hyperparameter search over configurable parameter distributions. The framework exposes the search configuration so users can adjust parameter ranges, number of iterations, and cross-validation strategy.

Outputs

- trained model (serialized)
- predictions for test/holdout sets
- evaluation metrics (e.g., RMSE, MAE, R2)
- feature importance plots
- SHAP summary/force plots (when generated)
- result files saved to the outputs/ directory

---

### XGBoost

Features
- XGBoost Regressor (scikit-learn compatible wrapper)
- Automatic preprocessing (shared dataset interface)
- Feature encoding for categorical inputs (or passthrough if already encoded)
- Hyperparameter optimization
- Cross-validation
- Feature importance
- SHAP analysis (SHAP supports XGBoost models)
- Performance visualization

Hyperparameter Optimization

RandomizedSearchCV (or a scikit-learn-compatible randomized search) is used to tune XGBoost hyperparameters. The search configuration is adjustable to control parameter ranges, iterations, and cross-validation folds.

Outputs

- trained model (serialized)
- predictions for test/holdout sets
- evaluation metrics (e.g., RMSE, MAE, R2)
- feature importance plots
- SHAP summary/force plots (when generated)
- result files saved to the outputs/ directory

---

## Dataset Requirements

- Any CSV dataset can be used with this framework.
- Place your dataset file inside the datasets/ directory.
- Both Random Forest and XGBoost expect the same dataset structure; models share the dataset interface so switching datasets does not require code changes to model pipelines.
- Update the dataset path in the model configuration or script argument to point to datasets/<your_dataset>.csv if necessary.

Show

datasets/
└── <your_dataset>.csv

---

## Required Dataset Format

Requirement | Description
--- | ---
CSV format | The input dataset must be a valid CSV file.
Numerical and/or categorical input features | Features may be numeric or categorical. The framework includes encoding and preprocessing utilities for categorical data.
One or more continuous regression targets | At least one continuous target column (float) must be present for supervised regression.
Header required | The CSV must include a header row with column names.
Missing values | Missing values should be handled before training or supported by the selected model/pipeline (imputation routines are available in preprocessing).

Note: Column selection and target assignment are configurable via the model configuration or script arguments; no particular column names are required by the framework.

---

## Installation

Install required packages (example):

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn shap
```

Optional (development / notebooks):

```bash
pip install jupyterlab
```

Or create a requirements.txt with the packages above and run:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Random Forest training and evaluation pipeline:

```bash
python rf_framework.py
```

Run the XGBoost training and evaluation pipeline:

```bash
python xgb_framework.py
```

By default the scripts read datasets/<your_dataset>.csv. You can override the dataset path via a configuration file or command-line argument if supported by the script.

---

## Outputs

Typical generated outputs (model-dependent):

outputs/
├── predictions.csv
├── metrics.csv
├── feature_importance.png
├── shap_summary.png
└── trained_model.pkl

The exact outputs depend on the selected model and configured analysis options (SHAP plots are generated where SHAP is available and enabled).

---

## Adding New Models

The framework is designed for expansion. To add a new model, implement a model wrapper that conforms to the shared dataset interface and plug it into the existing training/evaluation flow.

Planned / possible future models:
- Random Forest ✅
- XGBoost ✅
- LightGBM
- CatBoost
- Support Vector Regression (SVR)
- Multi-Layer Perceptron (MLP)
- TabNet

---

## Contributing

Contributions are welcome. Please follow these guidelines:

- Open an issue to discuss significant changes or new model additions.
- Fork the repository and create feature branches for pull requests.
- Keep changes focused and include tests or example runs where appropriate.
- Document new features in the README and any example notebooks or scripts.
- Report bugs with a clear description, reproducible steps, and sample configuration/data where possible.

We appreciate well-documented pull requests that include rationale and testing information.

---

## License
