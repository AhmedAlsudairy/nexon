import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
from matplotlib import cm
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from sklearn.model_selection import train_test_split

# Import statements updated for PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For progress bars
from tqdm import tqdm

# Import logging
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TRAINING_DATA_DIR = './data/lunar/training/data/S12_GradeA/'
TEST_DATA_DIR = './data/lunar/test/data/S12_GradeA/'
CATALOG_FILE = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
MODEL_FILE = 'moonquake_model.pth'  # Changed extension for PyTorch model

def read_catalog(catalog_file):
    return pd.read_csv(catalog_file)

def read_csv_file(csv_file):
    # Read the CSV and handle missing values represented by -1 or -1.0
    data = pd.read_csv(csv_file)
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

def plot_seismic_data(times, data, arrival_time, title, x_label='Time (s)', y_label='Velocity (m/s)'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(times, data)
    ax.axvline(x=arrival_time, c='red', label='Arrival')
    ax.set_xlim([min(times), max(times)])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontweight='bold')
    ax.legend()
    plt.show()

def plot_spectrogram(times, data, arrival_time, sampling_rate):
    f, t, sxx = signal.spectrogram(data, sampling_rate)

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(times, data)
    ax1.axvline(x=arrival_time, color='red', label='Detection')
    ax1.set_xlim([min(times), max(times)])
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(loc='upper left')

    ax2 = plt.subplot(2, 1, 2)
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet)
    ax2.set_xlim([min(times), max(times)])
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax2.axvline(x=arrival_time, c='red')
    cbar = plt.colorbar(vals, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
    plt.show()

def run_sta_lta(data, sampling_rate, sta_len, lta_len):
    return classic_sta_lta(data, int(sta_len * sampling_rate), int(lta_len * sampling_rate))

def plot_sta_lta(times, cft):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(times, cft)
    ax.set_xlim([min(times), max(times)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic function')
    plt.show()

def plot_triggers(times, data, on_off):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    for triggers in on_off:
        ax.axvline(x=times[triggers[0]], color='red', label='Trig. On')
        ax.axvline(x=times[triggers[1]], color='purple', label='Trig. Off')
    ax.plot(times, data)
    ax.set_xlim([min(times), max(times)])
    ax.legend()
    plt.show()

# Modify SeismicDataset for lazy loading
class SeismicDataset(Dataset):
    def __init__(self, data, labels, sequence_length):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length - 1]
        return x, y

def create_sequences(data, labels, sequence_length):
    logger.info("Creating sequences with sequence_length = %d", sequence_length)
    data = data.squeeze()  # Ensure data is 1D
    num_sequences = len(data) - sequence_length + 1
    logger.info("Number of sequences to be created: %d", num_sequences)
    if num_sequences <= 0:
        logger.warning("No sequences created. Data length (%d) is less than sequence length (%d).", len(data), sequence_length)
        return np.array([]), np.array([])
    # Create sequences using advanced indexing
    indices = np.arange(num_sequences)[:, None] + np.arange(sequence_length)
    sequences = data[indices]
    sequence_labels = labels[sequence_length - 1:]
    logger.info("Sequences shape: %s", sequences.shape)
    logger.info("Sequence labels shape: %s", sequence_labels.shape)
    return sequences, sequence_labels

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

def train_model(model, device, train_loader, val_loader, epochs, checkpoint_dir='checkpoints'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        train_loss = epoch_loss / total
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_loss = val_loss / total
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the model checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }
        checkpoint_filename = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_filename)
        print(f"Model checkpoint saved to {checkpoint_filename}")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }
    return model, history

def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Read catalog
    cat = read_catalog(CATALOG_FILE)
    print("Catalog loaded. First few entries:")
    print(cat.head())

    # Select a detection for visualization
    row = cat.iloc[6]
    arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
    arrival_time_rel = row['time_rel(sec)']
    test_filename = row.filename

    # Read and plot CSV data
    csv_file = f'{TRAINING_DATA_DIR}{test_filename}.csv'
    if os.path.exists(csv_file):
        data_cat = read_csv_file(csv_file)
        data_cat = clean_data(data_cat)  # Data cleansing
        data_cat = normalize_data(data_cat)  # Data normalization
        csv_times = np.array(data_cat['time_rel(sec)'].tolist())
        csv_data = np.array(data_cat['velocity(m/s)'].tolist())
        plot_seismic_data(csv_times, csv_data, arrival_time_rel, test_filename)
    else:
        print(f"CSV file not found: {csv_file}")

    # Read and plot miniseed data
    mseed_file = f'{TRAINING_DATA_DIR}{test_filename}.mseed'
    if os.path.exists(mseed_file):
        st = read(mseed_file)
        tr = st.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data
        starttime = tr.stats.starttime.datetime
        arrival = (arrival_time - starttime).total_seconds()
        plot_seismic_data(tr_times, tr_data, arrival, test_filename)

        # Filter the trace and plot spectrogram
        st_filt = st.copy()
        st_filt.filter('bandpass', freqmin=0.5, freqmax=1.0)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        plot_spectrogram(tr_times_filt, tr_data_filt, arrival, tr_filt.stats.sampling_rate)

        # Run STA/LTA
        sta_len, lta_len = 120, 600
        cft = run_sta_lta(tr_data, tr.stats.sampling_rate, sta_len, lta_len)
        plot_sta_lta(tr_times, cft)

        # Trigger detection
        thr_on, thr_off = 4, 1.5
        on_off = trigger_onset(cft, thr_on, thr_off)
        plot_triggers(tr_times, tr_data, on_off)

        # Create detection catalog
        detection_times = []
        fnames = []
        for triggers in on_off:
            on_time = starttime + timedelta(seconds=tr_times[triggers[0]])
            on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
            detection_times.append(on_time_str)
            fnames.append(test_filename)

        detect_df = pd.DataFrame({
            'filename': fnames,
            'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
            'time_rel(sec)': [tr_times[triggers[0]] for triggers in on_off]
        })
        print("\nDetection catalog:")
        print(detect_df)
    else:
        print(f"MSEED file not found: {mseed_file}")

    # Process all training files for ML model
    all_data = []
    all_labels = []
    sequence_length = 100  # Adjust as needed

    print("Processing training data...")
    for index, row in tqdm(cat.iterrows(), total=cat.shape[0], desc="Reading Catalog"):
        csv_file = f'{TRAINING_DATA_DIR}{row.filename}.csv'
        if not os.path.exists(csv_file):
            continue  # Skip files that don't exist
        data_cat = read_csv_file(csv_file)
        data_cat = clean_data(data_cat)  # Data cleansing
        data_cat = normalize_data(data_cat)  # Data normalization

        csv_data = np.array(data_cat['velocity(m/s)'].tolist())
        arrival_time_rel = row['time_rel(sec)']

        # Create label array (1 for event, 0 for non-event)
        labels = np.zeros(len(csv_data))
        event_index = np.argmin(np.abs(data_cat['time_rel(sec)'] - arrival_time_rel))
        labels[event_index:event_index+10] = 1  # Mark 10 samples as event

        all_data.extend(csv_data)
        all_labels.extend(labels)

    # Ensure that all_data and all_labels are numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    # Reshape data to have one feature
    all_data = all_data.reshape(-1, 1)

    # Log the size of the data
    logger.info("Total data points: %d", len(all_data))
    logger.info("Total labels: %d", len(all_labels))

    # Convert to PyTorch tensors
    all_data_tensor = torch.tensor(all_data, dtype=torch.float32)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

    # Create datasets and loaders using lazy loading
    sequence_length = 100  # Adjust as needed
    dataset = SeismicDataset(all_data_tensor, all_labels_tensor, sequence_length)
    logger.info("Total sequences in dataset: %d", len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # Initialize model
    model = LSTMClassifier(sequence_length).to(device)

    # Train model
    print("Training model...")
    model, history = train_model(model, device, train_loader, val_loader, epochs=10, checkpoint_dir='checkpoints')

    # Save the model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Load the model and use it on test data
    loaded_model = LSTMClassifier(sequence_length).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_FILE))
    loaded_model.eval()

    # Process test file
    test_file = 'your_test_file_name'  # Replace with your test file name
    test_csv_file = f'{TEST_DATA_DIR}{test_file}.csv'

    if os.path.exists(test_csv_file):
        test_data = read_csv_file(test_csv_file)
        test_data = clean_data(test_data)  # Data cleansing
        test_data = normalize_data(test_data)  # Data normalization
        test_velocity = np.array(test_data['velocity(m/s)'].tolist())

        # Reshape test data
        test_velocity = test_velocity.reshape(-1, 1)

        # Create sequences for test data
        logger.info("Creating sequences for test data")
        X_test, _ = create_sequences(test_velocity, np.zeros(len(test_velocity)), sequence_length)
        if X_test.size == 0:
            print("No test sequences created. Check the sequence length and test data size.")
            return
        # Convert to tensor
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Make predictions
        print("Making predictions on test data...")
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), 32), desc="Predicting"):
                X_batch = X_test[i:i+32]
                outputs = loaded_model(X_batch)
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions)

        # Plot test data and predictions
        plt.figure(figsize=(15, 5))
        plt.plot(test_data['time_rel(sec)'], test_velocity, label='Velocity')
        plt.plot(test_data['time_rel(sec)'][sequence_length-1:], predictions, label='Event Probability', color='red')
        plt.title('Test Data and Predictions')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Velocity / Event Probability')
        plt.legend()
        plt.show()

        # Detect events (you may need to adjust the threshold)
        threshold = 0.5
        events = np.where(predictions > threshold)[0]

        print("Detected events at:")
        for event in events:
            print(f"Time: {test_data['time_rel(sec)'].iloc[event + sequence_length - 1]} seconds")
    else:
        print(f"Test file not found: {test_csv_file}")

if __name__ == "__main__":
    main()
