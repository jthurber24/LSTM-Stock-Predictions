#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Function to load and preprocess data for LSTM
def load_and_preprocess_for_lstm(file_path, sequence_length):
    df = pd.read_csv(file_path)

    # Check if 'Adj Close' column exists, otherwise use 'Close'
    if 'Adj Close' in df.columns:
        price_column = 'Adj Close'
    elif 'Close' in df.columns:
        price_column = 'Close'
    else:
        raise KeyError(f"No suitable price column found in the file: {file_path}")

    data = df[price_column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Creating sequences for LSTM
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler


# Define the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, sequence_length):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size * sequence_length, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_layer_size * sequence_length)
        predictions = self.linear(lstm_out)
        return predictions

# Train and evaluate the LSTM model
def train_and_evaluate_lstm(X_train, X_test, y_train, y_test, model, scaler, epochs=25, learning_rate=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), batch_size=32, shuffle=True)
    
    # Training the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, labels in train_data:
            optimizer.zero_grad()
            y_pred = model(seq)
            y_pred = y_pred.squeeze()
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data)
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss}')

    # Evaluating the model
    def evaluate_model(data_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for seq, labels in data_loader:
                y_pred = model(seq).squeeze()
                loss = loss_function(y_pred, labels)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()), batch_size=32, shuffle=False)

    train_loss = evaluate_model(train_loader)
    test_loss = evaluate_model(test_loader)
    print(f'Final Train Loss: {train_loss}, Test Loss: {test_loss}')

def create_binary_outcomes(prices):
    return np.array([1 if prices[i] > prices[i-1] else 0 for i in range(1, len(prices))])

# Directory paths for 'stocks' and 'etfs'
stocks_dir = 'C:/Stocks/stocks'
etfs_dir = 'C:/Stocks/etfs'

# Collecting all CSV file paths from 'stocks' and 'etfs' directories
stock_files = [os.path.join(stocks_dir, f) for f in os.listdir(stocks_dir) if f.endswith('.csv')]
etf_files = [os.path.join(etfs_dir, f) for f in os.listdir(etfs_dir) if f.endswith('.csv')]

# Combined list of all file paths
all_file_paths = stock_files + etf_files

# Set sequence length
sequence_length = 60

# Define the LSTM model
input_size = 1  # Number of features
hidden_layer_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_layer_size, output_size, sequence_length)

for file_path in all_file_paths[:5]:
    # Load and preprocess the data for LSTM
    (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_for_lstm(file_path, sequence_length)
    print(f'\nTraining and evaluating LSTM model for {file_path}')
    train_and_evaluate_lstm(X_train, X_test, y_train, y_test, model, scaler)

    # Evaluate and get predictions
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float()
        test_predictions = model(X_test_tensor).squeeze()
    test_predictions = scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1)).flatten()
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Plotting Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(test_predictions, label='Predicted Prices', color='red')
    plt.title(f'Stock Price Prediction - Actual vs Predicted for {os.path.basename(file_path)}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Convert to binary outcomes and create a confusion matrix
    actual_binary = create_binary_outcomes(actual_prices)
    predicted_binary = create_binary_outcomes(test_predictions)
    conf_matrix = confusion_matrix(actual_binary[1:], predicted_binary[:-1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {os.path.basename(file_path)}')
    plt.xlabel('Predicted')
    plt.ylabel('True ')
    plt.show()
    print(classification_report(actual_binary[1:], predicted_binary[:-1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




