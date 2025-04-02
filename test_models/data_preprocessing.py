import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data_path, sequence_length=10):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Handle missing values
    data.fillna(0, inplace=True)  # Replace missing values with 0

    # Identify categorical columns
    categorical_columns = ['RegionName', 'RegionType', 'StateName']

    # One-hot encode categorical columns
    if any(col in data.columns for col in categorical_columns):
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Extract relevant columns (assuming the total monthly payment is the target)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target (Total Monthly Payment)

    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    # Create input-output sequences
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y
