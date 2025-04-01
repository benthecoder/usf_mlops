"""
Data preprocessing script for cardiovascular disease dataset.
This script reads the original cardio.csv dataset, performs preprocessing,
and splits the data into train, validation, and test sets.

It includes:
- Data loading
- Data cleaning and handling outliers
- Feature engineering
- Train/validation/test split
- Saving processed datasets
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


def load_data(filepath, sep=";"):
    """
    Load the cardiovascular dataset.

    Parameters:
    -----------
    filepath : str
        Path to the dataset
    sep : str, default=';'
        Separator used in the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath, sep=sep)
    print(f"Dataset shape: {data.shape}")
    return data


def clean_data(data):
    """
    Clean the dataset by removing outliers and handling missing values.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("Cleaning data...")

    # Make a copy to avoid modifying the original dataframe
    data_cleaned = data.copy()

    # Convert age from days to years
    data_cleaned["age"] = data_cleaned["age"] / 365.25

    # Remove height outliers - heights below 120 cm or above 220 cm
    data_cleaned = data_cleaned[
        (data_cleaned["height"] >= 120) & (data_cleaned["height"] <= 220)
    ]

    # Remove weight outliers - weights below 30 kg or above 200 kg
    data_cleaned = data_cleaned[
        (data_cleaned["weight"] >= 30) & (data_cleaned["weight"] <= 200)
    ]

    # Remove blood pressure outliers
    # Systolic blood pressure (ap_hi) should be higher than diastolic (ap_lo)
    data_cleaned = data_cleaned[data_cleaned["ap_hi"] > data_cleaned["ap_lo"]]

    # Remove extreme blood pressure values
    data_cleaned = data_cleaned[
        (data_cleaned["ap_hi"] >= 80) & (data_cleaned["ap_hi"] <= 220)
    ]
    data_cleaned = data_cleaned[
        (data_cleaned["ap_lo"] >= 40) & (data_cleaned["ap_lo"] <= 120)
    ]

    # Report the number of rows removed
    rows_removed = len(data) - len(data_cleaned)
    print(
        f"Removed {rows_removed} rows ({rows_removed / len(data) * 100:.2f}% of data)"
    )
    print(f"Cleaned dataset shape: {data_cleaned.shape}")

    return data_cleaned


def feature_engineering(data):
    """
    Perform feature engineering on the dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    print("Performing feature engineering...")

    # Make a copy to avoid modifying the original dataframe
    data_featured = data.copy()

    # Calculate BMI
    data_featured["bmi"] = data_featured["weight"] / (
        (data_featured["height"] / 100) ** 2
    )

    # Calculate pulse pressure
    data_featured["pulse_pressure"] = data_featured["ap_hi"] - data_featured["ap_lo"]

    # Convert categorical variables to numeric if needed
    # Already done in this dataset

    return data_featured


def select_features(X_train, y_train, X_val, X_test, k=6):
    """
    Select the most relevant features using chi-squared test.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    X_test : pd.DataFrame
        Test features
    k : int, default=6
        Number of features to select

    Returns:
    --------
    tuple
        Tuple containing the selected features for train, validation, and test sets
    """
    print(f"Selecting top {k} features...")

    # Initialize the chi-squared feature selector
    selector = SelectKBest(chi2, k=k)

    # Fit and transform the training data
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices]
    print(f"Selected features: {', '.join(selected_features)}")

    # Create DataFrames with only the selected features
    X_train_selected = X_train.iloc[:, selected_indices]
    X_val_selected = X_val.iloc[:, selected_indices]
    X_test_selected = X_test.iloc[:, selected_indices]

    return X_train_selected, X_val_selected, X_test_selected, selected_features


def split_data(data, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    val_size : float, default=0.25
        Proportion of the training dataset to include in the validation split
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    tuple
        Tuple containing train, validation, and test sets
    """
    print("Splitting data into train, validation, and test sets...")

    # Separate features and target
    X = data.drop("cardio", axis=1)
    y = data["cardio"]

    # First split: training + validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_datasets(train_data, val_data, test_data, output_dir="data/cardio/"):
    """
    Save the preprocessed datasets to CSV files.

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training dataset
    val_data : pd.DataFrame
        Validation dataset
    test_data : pd.DataFrame
        Test dataset
    output_dir : str, default='data/cardio/'
        Directory to save the datasets
    """
    print(f"Saving datasets to {output_dir}...")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    train_data.to_csv(f"{output_dir}train_data.csv", index=False)
    val_data.to_csv(f"{output_dir}val_data.csv", index=False)
    test_data.to_csv(f"{output_dir}test_data.csv", index=False)

    print("Datasets saved successfully.")


def save_selected_datasets(
    train_data_selected,
    val_data_selected,
    test_data_selected,
    output_dir="data/cardio/",
):
    """
    Save the feature-selected datasets to CSV files.

    Parameters:
    -----------
    train_data_selected : pd.DataFrame
        Training dataset with selected features
    val_data_selected : pd.DataFrame
        Validation dataset with selected features
    test_data_selected : pd.DataFrame
        Test dataset with selected features
    output_dir : str, default='data/cardio/'
        Directory to save the datasets
    """
    print(f"Saving feature-selected datasets to {output_dir}...")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    train_data_selected.to_csv(f"{output_dir}train_data_selected.csv", index=False)
    val_data_selected.to_csv(f"{output_dir}val_data_selected.csv", index=False)
    test_data_selected.to_csv(f"{output_dir}test_data_selected.csv", index=False)

    print("Feature-selected datasets saved successfully.")


def main():
    """Main function to execute the preprocessing pipeline."""
    # Parameters (can be moved to a params.yaml file for DVC)
    data_path = "data/cardio/cardio.csv"
    output_dir = "data/cardio/"
    test_size = 0.2
    val_size = 0.25
    random_state = 42
    n_features = 6

    # Load data
    data = load_data(data_path)

    # Clean data
    data_cleaned = clean_data(data)

    # Feature engineering
    data_featured = feature_engineering(data_cleaned)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        data_featured, test_size, val_size, random_state
    )

    # Create and save full datasets
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    save_datasets(train_data, val_data, test_data, output_dir)

    # Select features
    X_train_selected, X_val_selected, X_test_selected, selected_features = (
        select_features(X_train, y_train, X_val, X_test, k=n_features)
    )

    # Create and save selected feature datasets
    train_data_selected = pd.concat([X_train_selected, y_train], axis=1)
    val_data_selected = pd.concat([X_val_selected, y_val], axis=1)
    test_data_selected = pd.concat([X_test_selected, y_test], axis=1)

    save_selected_datasets(
        train_data_selected, val_data_selected, test_data_selected, output_dir
    )

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
