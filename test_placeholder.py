

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generating dummy dataset
def generate_dummy_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'income': np.random.randint(20000, 100000, 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'loan_amount': np.random.randint(5000, 50000, 1000),
        'loan_status': np.random.choice([0, 1], 1000)  # 0: Default, 1: Paid
    })
    return data

# Splitting data into train and test sets
def split_data(data):
    X = data[['income', 'credit_score', 'loan_amount']]
    y = data['loan_status']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Testing model
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Test cases
def test_data_generation():
    data = generate_dummy_data()
    assert not data.empty
    assert 'income' in data.columns
    assert 'credit_score' in data.columns
    assert 'loan_amount' in data.columns
    assert 'loan_status' in data.columns

def test_data_splitting():
    data = generate_dummy_data()
    X_train, X_test, y_train, y_test = split_data(data)
    assert len(X_train) > len(X_test)
    assert len(y_train) > len(y_test)

def test_model_training():
    data = generate_dummy_data()
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    assert model is not None

def test_model_accuracy():
    data = generate_dummy_data()
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    accuracy = test_model(model, X_test, y_test)
    assert accuracy > 0.5  # Ensuring reasonable accuracy

print("Loan prediction test file loaded.")
