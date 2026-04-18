"""
Automated Preprocessing Script for Credit Card Fraud Detection
Author: Muhammad Ivan
Converted from Eksperimen_Muhammad-Ivan.ipynb
Same preprocessing steps as notebook but in function-based structure
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='../namadataset_raw/creditcard.csv'):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def handle_missing_values(df):
    print("\n--- Handling Missing Values ---")
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values: {missing_count}")
    if missing_count > 0:
        print("Dropping rows with missing values...")
        df = df.dropna()
    print(f"Rows after handling missing values: {len(df)}")
    return df


def handle_duplicates(df):
    print("\n--- Handling Duplicates ---")
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")
    if dup_count > 0:
        print("Dropping duplicate rows...")
        df = df.drop_duplicates()
    print(f"Rows after handling duplicates: {len(df)}")
    return df


def scale_features(df):
    print("\n--- Scaling Features (Time and Amount) ---")
    scaler = StandardScaler()
    df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    print(f"Columns after scaling: {df.columns.tolist()}")
    return df


def split_data(df, test_size=0.2, random_state=42):
    print("\n--- Train-Test Split ---")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples (Normal: {(y_train == 0).sum()}, Fraud: {(y_train == 1).sum()})")
    print(f"Testing set: {X_test.shape[0]} samples (Normal: {(y_test == 0).sum()}, Fraud: {(y_test == 1).sum()})")
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir='namadataset_preprocessing'):
    print("\n--- Saving Preprocessed Data ---")
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['Class'] = y_train.values
    train_path = os.path.join(output_dir, 'creditcard_train_preprocessed.csv')
    train_df.to_csv(train_path, index=False)
    print(f"Saved training data: {train_path} ({len(train_df)} rows)")

    test_df = X_test.copy()
    test_df['Class'] = y_test.values
    test_path = os.path.join(output_dir, 'creditcard_test_preprocessed.csv')
    test_df.to_csv(test_path, index=False)
    print(f"Saved testing data: {test_path} ({len(test_df)} rows)")

    full_df = pd.concat([X_train, X_test], axis=0)
    full_df['Class'] = pd.concat([y_train, y_test], axis=0)
    full_path = os.path.join(output_dir, 'creditcard_preprocessed.csv')
    full_df.to_csv(full_path, index=False)
    print(f"Saved full preprocessed data: {full_path} ({len(full_df)} rows)")

    return train_path, test_path, full_path


def automate_preprocessing(input_path='../namadataset_raw/creditcard.csv',
                           output_dir='namadataset_preprocessing',
                           test_size=0.2,
                           random_state=42):
    print("=" * 60)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("Credit Card Fraud Detection - Muhammad Ivan")
    print("=" * 60)

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    train_path, test_path, full_path = save_preprocessed_data(
        X_train, X_test, y_train, y_test, output_dir
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("Data is ready for model training!")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = automate_preprocessing()