from sklearn.preprocessing import LabelEncoder

# =============================================================================
# fe_v0_0_0
# description: Initial version with basic label encoding for categorical features.
# -----------------------------------------------------------------------------
# This module contains feature engineering processes for the churn prediction
# project. It is intended to centralize all transformations and preprocessing
# steps applied to the dataset, including encoding categorical variables and
# other feature manipulations. The functions herein are designed to prepare
# data for machine learning models by ensuring consistent and reproducible
# feature engineering workflows.
# =============================================================================

def preprocess_data(df):

    object_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders
