# Install necessary libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import logging
import gradio as gr
import io
from PIL import Image
from torch.amp import autocast

# Configuration
MODEL_FILE = 'moonquake_model.pth'  # Path to your saved model
SEQUENCE_LENGTH = 100  # Same as used during training

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

@torch.no_grad()
def make_predictions(test_features, model):
    model.eval()
    predictions = []
    batch_size = 4096  # Increased batch size for GPU
    num_batches = len(test_features) // batch_size + (1 if len(test_features) % batch_size != 0 else 0)

    for i in tqdm(range(0, len(test_features), batch_size), total=num_batches, desc="Predicting"):
        X_batch = test_features[i:i+batch_size].to(device)
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
        predictions.extend(probs.cpu().numpy())

    return np.array(predictions)

def process_data(file):
    data = pd.read_csv(file)
    data.replace([-1, -1.0], np.nan, inplace=True)
    data['velocity(m/s)'] = data['velocity(m/s)'].ffill().bfill().fillna(0)

    mean = data['velocity(m/s)'].mean()
    std = data['velocity(m/s)'].std()
    data['velocity(m/s)'] = (data['velocity(m/s)'] - mean) / std

    velocity = data['velocity(m/s)'].values
    derivative = np.diff(velocity, prepend=velocity[0])
    rolling_mean = pd.Series(velocity).rolling(window=10, min_periods=1).mean().values
    rolling_std = pd.Series(velocity).rolling(window=10, min_periods=1).std().fillna(0).values

    features = np.stack((velocity, derivative, rolling_mean, rolling_std), axis=1)

    num_sequences = len(features) - SEQUENCE_LENGTH + 1
    X = np.lib.stride_tricks.sliding_window_view(features, (SEQUENCE_LENGTH, features.shape[1]))
    X = X.reshape(num_sequences, SEQUENCE_LENGTH, features.shape[1])

    return data, torch.tensor(X, dtype=torch.float32)

def plot_results(test_data, predictions):
    plt.figure(figsize=(15, 5))
    plt.plot(test_data['time_rel(sec)'], test_data['velocity(m/s)'], label='Normalized Velocity')
    plt.plot(test_data['time_rel(sec)'][SEQUENCE_LENGTH-1:], predictions, label='Event Probability', color='red')

    threshold = 0.5
    events = np.where(predictions > threshold)[0]
    event_times = test_data['time_rel(sec)'].iloc[events + SEQUENCE_LENGTH - 1]
    for event_time in event_times:
        plt.axvline(x=event_time, color='green', linestyle='--', label='Detected Event')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.title('Test Data and Predictions with Detected Events')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Velocity / Event Probability')
    plt.grid(True)

    # Add percentage of detected events
    event_percentage = (len(events) / len(predictions)) * 100
    plt.text(0.02, 0.98, f'Events Detected: {event_percentage:.2f}%',
             transform=plt.gca().transAxes, verticalalignment='top')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)

def create_event_table(test_data, predictions, threshold=0.5):
    events = np.where(predictions > threshold)[0]
    event_times = test_data['time_rel(sec)'].iloc[events + SEQUENCE_LENGTH - 1]
    event_probabilities = predictions[events]

    event_df = pd.DataFrame({
        'Event Time (s)': event_times,
        'Event Probability': event_probabilities
    })
    event_df = event_df.sort_values('Event Probability', ascending=False).reset_index(drop=True)
    event_df.index += 1  # Start index from 1
    return event_df

def gradio_interface(file):
    try:
        progress = gr.Progress()

        progress(0, desc="Loading model")
        model = LSTMClassifier(input_size=4).to(device)

        if not os.path.exists(MODEL_FILE):
            logger.error(f"Model file not found: {MODEL_FILE}")
            return "Error: Model file not found.", None

        state_dict = torch.load(MODEL_FILE, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        progress(0.2, desc="Processing data")
        test_data, test_features = process_data(file.name)

        progress(0.4, desc="Making predictions")
        predictions = make_predictions(test_features.to(device), model)

        progress(0.6, desc="Creating event table")
        event_table = create_event_table(test_data, predictions)

        progress(0.8, desc="Plotting results")
        plot_image = plot_results(test_data, predictions)

        progress(1.0, desc="Complete")
        return event_table, plot_image

    except Exception as e:
        logger.exception("An error occurred during processing")
        return pd.DataFrame({"Error": [str(e)]}), None

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(label="Upload CSV File"),
    outputs=[
        gr.Dataframe(label="Detected Events"),
        gr.Image(label="Event Detection Plot")
    ],
    title="LSTM Event Detection",
    description="Upload a CSV file containing velocity data to detect events."
)

iface.launch(share=True, debug=True)