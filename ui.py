import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import gradio as gr
import os

# Define the LSTMClassifier class as before
class LSTMClassifier(nn.Module):
    def __init__(self, sequence_length):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 64).to(x.device)
        c_0 = torch.zeros(1, x.size(0), 64).to(x.device)
        out, _ = self.lstm1(x, (h_0, c_0))
        out = self.dropout1(out)
        h_1 = torch.zeros(1, x.size(0), 32).to(x.device)
        c_1 = torch.zeros(1, x.size(0), 32).to(x.device)
        out, _ = self.lstm2(out, (h_1, c_1))
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out.squeeze()

# Initialize the model variable
model = None
sequence_length = 100  # Ensure this matches the model
MODEL_FILE = 'moonquake_model.pth'

# Data preprocessing functions remain the same
def read_csv_file(file_obj):
    # Read the CSV and handle missing values represented by -1 or -1.0
    data = pd.read_csv(file_obj)
    data.replace(-1, np.nan, inplace=True)
    data.replace(-1.0, np.nan, inplace=True)
    return data

def clean_data(data):
    # Handle missing values
    data['velocity(m/s)'].fillna(method='ffill', inplace=True)  # Forward fill
    data['velocity(m/s)'].fillna(method='bfill', inplace=True)  # Backward fill
    # If there are still NaNs, fill them with zero
    data['velocity(m/s)'].fillna(0, inplace=True)
    return data

def normalize_data(data):
    # Normalize the velocity data
    mean = data['velocity(m/s)'].mean()
    std = data['velocity(m/s)'].std()
    data['velocity(m/s)'] = (data['velocity(m/s)'] - mean) / std
    return data

def create_sequences(data, sequence_length):
    data = data.squeeze()  # Ensure data is 1D
    num_sequences = len(data) - sequence_length + 1
    if num_sequences <= 0:
        return np.array([])
    # Create sequences using advanced indexing
    indices = np.arange(num_sequences)[:, None] + np.arange(sequence_length)
    sequences = data[indices]
    return sequences

def make_predictions(file_obj, model_file=None):
    global model
    # Check if the model is loaded
    if model is None:
        # Try to load from the default model file
        if os.path.exists(MODEL_FILE):
            # Load the model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMClassifier(sequence_length).to(device)
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            model.eval()
        else:
            # Check if the user uploaded a model file
            if model_file is None:
                return "No model uploaded.", None, None
            else:
                # Load the model from the uploaded file
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = LSTMClassifier(sequence_length).to(device)
                model.load_state_dict(torch.load(model_file.name, map_location=device))
                model.eval()

    # Read and preprocess data
    data = read_csv_file(file_obj)
    data = clean_data(data)
    data = normalize_data(data)
    times = data['time_rel(sec)'].values
    velocity = data['velocity(m/s)'].values
    velocity = velocity.reshape(-1, 1)

    # Create sequences
    sequences = create_sequences(velocity, sequence_length)
    if sequences.size == 0:
        return "Insufficient data length for the given sequence length.", None, None

    # Convert to tensor
    X = torch.tensor(sequences, dtype=torch.float32).to(device)

    # Make predictions
    predictions = []
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
    predictions = np.array(predictions)

    # Prepare plots
    # Plot velocity
    fig1 = plt.figure(figsize=(15, 5))
    plt.plot(times, velocity, label='Velocity')
    plt.title('Seismic Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Velocity')
    plt.legend()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    plt.close(fig1)

    # Plot predictions
    fig2 = plt.figure(figsize=(15, 5))
    plt.plot(times[sequence_length-1:], predictions, label='Event Probability', color='red')
    plt.title('Predicted Event Probabilities')
    plt.xlabel('Time (s)')
    plt.ylabel('Event Probability')
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    plt.close(fig2)

    # Detect events (adjust threshold as needed)
    threshold = 0.5
    events = np.where(predictions > threshold)[0]
    event_times = times[sequence_length - 1 + events]
    event_times_str = "\n".join([f"{t} seconds" for t in event_times])

    if len(event_times) == 0:
        event_times_str = "No events detected."

    return event_times_str, buf1, buf2

# Define Gradio interface
inputs = [
    gr.inputs.File(label="Upload Seismic Data CSV File"),
    gr.inputs.File(label="Upload Model File (.pth)", optional=True)
]
outputs = [
    gr.outputs.Textbox(label="Detected Events"),
    gr.outputs.Image(type="auto", label="Seismic Data Plot"),
    gr.outputs.Image(type="auto", label="Predicted Event Probabilities")
]

title = "Moonquake Detection using LSTM Model"
description = """
Upload a seismic data CSV file to detect moonquakes using the trained LSTM model.
The CSV file should contain at least two columns: 'time_rel(sec)' and 'velocity(m/s)'.

If the default model file 'moonquake_model.pth' is not available, please upload a model file (.pth) using the 'Upload Model File' option.
"""

iface = gr.Interface(
    fn=make_predictions,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    theme="default"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
