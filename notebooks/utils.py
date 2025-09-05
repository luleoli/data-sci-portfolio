
# Linear Models

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Non-linear Models

from sklearn.neighbors import KNeighborsRegressor

# Tree-based Models

from sklearn.tree import DecisionTreeRegressor

# Ensemble Methods

from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

# Data Preparation

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Evaluation Metrics

from sklearn.metrics import mean_squared_error

# Utility imports

import numpy as np
import pandas as pd

import time

import warnings
from tqdm import tqdm

models_list = {
    "LinearRegressionOLS": LinearRegression(),
    "LinearRegressionSGD": SGDRegressor(max_iter=1000, random_state=42, learning_rate='adaptive', eta0=0.1),
    "LinearRegressionRidge": Ridge(alpha=0.1),
    "LinearRegressionLasso": Lasso(alpha=0.1),
    "LinearRegressionElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "KNN": KNeighborsRegressor(n_neighbors=5, algorithm='kd_tree'),
    "DecisionTree": DecisionTreeRegressor(max_depth=20,random_state=42),
    "SVM": SVR(max_iter=1000, kernel='rbf', C=1.0, epsilon=0.1),
    "RandomForest": RandomForestRegressor(n_estimators = 100, random_state=42, verbose=0),
    "XGBoost": XGBRegressor(n_estimators = 100, random_state=42, verbose=0),
    "LightGBM": LGBMRegressor(n_estimators = 100, random_state=42, verbose=0),
    "CatBoost": CatBoostRegressor(n_estimators = 100, random_state=42, verbose=0)
}


def run_models(df, dataset, target, models=models_list):
    """ Runs the specified models on the provided data and evaluates their performance."""

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Sample the dataframe if it has more than 500000 rows
    if len(df) > 500000:
        df = df.sample(n=500000, random_state=42).reset_index(drop=True)

    # Fit and transform all non-float and non-int columns using LabelEncoder
    for col in df.select_dtypes(exclude=['float', 'int']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    X = df.drop(columns=[target])  # Assuming 'target' is the name of the target variable
    y = df[target]  # Assuming 'target' is the name of the target

    # Split the data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"{name} MSE: {mse:.4f}, Train Size: {len(X_train)}, Training time: {training_time:.2f} seconds")

        test_mse = mse
        results[name] = {
            "Test MSE": test_mse,
            "Size": len(X_train),
            "Time": training_time
        }

    return results