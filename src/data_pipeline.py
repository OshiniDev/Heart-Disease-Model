"""
Data Pipeline Module for Heart Disease Prediction
Handles loading, preprocessing, Z-score standardization, train/test splitting,
and reshaping for Deep Learning sequence models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the comprehensive heart disease dataset."""
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = 'target'):
    """
    Clean dataset and prepare features/target.
    - Handles zero blood pressure and cholesterol anomalies if present
    - Separates features X and target y
    """
    df_clean = df.copy()

    # In clinical datasets, resting BP of 0 is physiologically invalid;
    # replace 0 with the median of valid values (if any)
    if (df_clean['resting bp s'] == 0).any():
        median_bp = df_clean.loc[df_clean['resting bp s'] > 0, 'resting bp s'].median()
        df_clean.loc[df_clean['resting bp s'] == 0, 'resting bp s'] = median_bp

    # Handle cholesterol = 0 (common missing indicator in Hungarian/Switzerland subsets)
    if (df_clean['cholesterol'] == 0).any():
        median_chol = df_clean.loc[df_clean['cholesterol'] > 0, 'cholesterol'].median()
        df_clean.loc[df_clean['cholesterol'] == 0, 'cholesterol'] = median_chol

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col].astype(int)

    return X, y, df_clean


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """
    Splits data into 80% train and 20% test using stratified holdout,
    and applies Z-score standardization: z = (x - mean) / std.
    StandardScaler is fitted strictly on X_train to avoid data leakage.
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    feature_names = X.columns.tolist()

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'scaler': scaler,
        'feature_names': feature_names
    }


def prepare_dl_inputs(X_train: np.ndarray, X_test: np.ndarray):
    """
    Reshapes 2D tabular features into 3D tensors (batch_size, timesteps=11, features=1)
    for sequence/convolutional DL models (1D-CNN, LSTM, CNN-LSTM).
    """
    # Reshape from (N, 11) to (N, 11, 1)
    X_train_seq = np.expand_dims(X_train, axis=-1)
    X_test_seq = np.expand_dims(X_test, axis=-1)

    return X_train_seq, X_test_seq
