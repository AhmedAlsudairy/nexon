import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
from matplotlib import cm
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import find_peaks

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

# For multiprocessing
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Configuration
TRAINING_DATA_DIR = 'data/lunar/training/data/S12_GradeA/'
TEST_DATA_DIR = 'data/lunar/test/data/S12_GradeA/'
CATALOG_FILE = 'data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
MODEL_FILE = 'moonquake_model.pth'  # Changed extension for PyTorch model

def read_catalog(catalog_file):
    logger.info(f"Reading catalog from {catalog_file}")
    return pd.read_csv(catalog_file)

def read_csv_file(csv_file):
    logger.info(f"Reading CSV file {csv_file}")
    # Read the CSV and handle missing values represented by -1 or -1.0
    data = pd.read_csv(csv_file)
    data.replace(-1, np.nan, inplace=True)
    data.replace(-1.0, np.nan, inplace=True)
    return data

def clean_data(data):
    logger.debug("Cleaning data")
    # Handle missing values without chained assignments
    data = data.copy()
    # Use ffill() and bfill() methods directly
    data['velocity(m/s)'] = data['velocity(m/s)'].ffill()
    data['velocity(m/s)'] = data['velocity(m/s)'].bfill()
    # If there are still NaNs, fill them with zero
    data['velocity(m/s)'] = data['velocity(m/s)'].fillna(0)
    return data

def normalize_data(data):
    logger.debug("Normalizing data")
    # Normalize the velocity data
    mean = data['velocity(m/s)'].mean()
    std = data['velocity(m/s)'].std()
    if std == 0:
        std = 1  # Prevent division by zero
    data['velocity(m/s)'] = (data['velocity(m/s)'] - mean) / std
    return data

def compute_derivative(data):
    derivative = np.diff(data, prepend=data[0])
    return derivative

def compute_rolling_stats(data, window_size):
    df = pd.Series(data)
    rolling_mean = df.rolling(window=window_size, min_periods=1).mean().values
    rolling_std = df.rolling(window=window_size, min_periods=1).std().values
    # Replace NaN with 0
    rolling_std = np.nan_to_num(rolling_std)
    return rolling_mean, rolling_std

def detect_peaks(data, height=None, threshold=None, distance=None, prominence=None):
    peaks, _ = find_peaks(data, height=height, threshold=threshold, distance=distance, prominence=prominence)
    return peaks

# Modify SeismicDataset for multiple features
class SeismicDataset(Dataset):
    def __init__(self, data, labels, sequence_length):
        self.data = data  # Shape: (num_samples, num_features)
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]  # Shape: (sequence_length, num_features)
        y = self.labels[idx + self.sequence_length - 1]
        return x, y

class LSTMClassifier(nn.Module):
    def __init__(self, input_size):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)
        # No Sigmoid activation here

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def train_model(model, device, train_loader, val_loader, epochs, pos_weight, checkpoint_dir='checkpoints'):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
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
        all_true_labels = []
        all_predictions = []

        loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_true_labels.extend(y_batch.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        train_loss = epoch_loss / total
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Calculate and print classification report
        logger.info("Training Metrics:")
        logger.info(classification_report(all_true_labels, all_predictions, digits=6))

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_true_labels = []
        val_predictions = []
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}")
            for X_batch, y_batch in loop:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                val_true_labels.extend(y_batch.cpu().numpy())
                val_predictions.extend(preds.cpu().numpy())
                loop.set_postfix(loss=loss.item())

        val_loss = val_loss / total
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Calculate and print validation classification report
        logger.info("Validation Metrics:")
        logger.info(classification_report(val_true_labels, val_predictions, digits=6))

        logger.info(f"Epoch {epoch+1}/{epochs}, "
                    f"Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.6f}, "
                    f"Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.6f}")

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
        logger.info(f"Model checkpoint saved to {checkpoint_filename}")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }
    return model, history

def process_file(row):
    csv_file = f'{TRAINING_DATA_DIR}{row.filename}.csv'
    if not os.path.exists(csv_file):
        logger.warning(f"File not found: {csv_file}")
        return None  # Skip files that don't exist
    data_cat = read_csv_file(csv_file)
    data_cat = clean_data(data_cat)  # Data cleansing
    data_cat = normalize_data(data_cat)  # Data normalization

    csv_data = np.array(data_cat['velocity(m/s)'].tolist())
    derivative = compute_derivative(csv_data)
    rolling_mean, rolling_std = compute_rolling_stats(csv_data, window_size=10)

    # Stack the features together
    try:
        features = np.stack((csv_data, derivative, rolling_mean, rolling_std), axis=1)  # Shape: (num_samples, num_features)
    except ValueError as e:
        print(f"Error stacking features for file {csv_file}: {e}")
        return None

    print(f"Features shape for file {csv_file}: {features.shape}")

    # Detect peaks in the velocity data
    peaks = detect_peaks(csv_data, prominence=1)  # Adjust 'prominence' as needed

    # Create label array (1 for event, 0 for non-event)
    labels = np.zeros(len(csv_data))
    # Label known events from the catalog
    arrival_time_rel = row['time_rel(sec)']
    event_indices = np.where(np.abs(data_cat['time_rel(sec)'] - arrival_time_rel) <= 10)[0]
    labels[event_indices] = 1  # Mark samples within 10 seconds of event as positive

    # Label peaks as events
    labels[peaks] = 1

    return features, labels

def calculate_class_distribution(labels):
    labels = np.array(labels)
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    total = len(labels)
    positive_ratio = num_positive / total if total > 0 else 0
    print(f"Total samples: {total}")
    print(f"Positive samples: {num_positive}")
    print(f"Negative samples: {num_negative}")
    print(f"Positive class ratio: {positive_ratio:.6f}")

def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Read catalog
    cat = read_catalog(CATALOG_FILE)
    print("Catalog loaded. First few entries:")
    print(cat.head())

    # Process all training files for ML model
    sequence_length = 100  # Adjust as needed

    # Check if preprocessed data exists
    if os.path.exists('preprocessed_data.npz'):
        print("Loading preprocessed data...")
        data = np.load('preprocessed_data.npz', allow_pickle=True)
        all_data = data['all_data']
        all_labels = data['all_labels']
    else:
        print("Processing training data...")
        results = Parallel(n_jobs=cpu_count())(
            delayed(process_file)(row) for index, row in tqdm(cat.iterrows(), total=cat.shape[0], desc="Reading Catalog")
        )

        # Now, collect the data
        all_data = []
        all_labels = []
        for result in results:
            if result is None:
                continue
            features, labels = result
            all_data.append(features)
            all_labels.append(labels)
        if len(all_data) == 0:
            print("No data was loaded. Please check your data files.")
            return
        # Concatenate all data
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        print(f"Shape of all_data after concatenation: {all_data.shape}")
        print(f"Shape of all_labels after concatenation: {all_labels.shape}")
        # Save preprocessed data
        np.savez('preprocessed_data.npz', all_data=all_data, all_labels=all_labels)

    # Ensure that all_data and all_labels are numpy arrays
    all_data = np.array(all_data)  # Shape: (num_samples, num_features)
    all_labels = np.array(all_labels)

    # Log the size of the data
    logger.info("Total data points: %d", len(all_data))
    logger.info("Total labels: %d", len(all_labels))

    # Check if all_data is empty
    if len(all_data.shape) == 1 or all_data.shape[0] == 0:
        print("Error: all_data is empty or has incorrect dimensions.")
        return

    # Calculate and print class distribution
    calculate_class_distribution(all_labels)

    # Convert to PyTorch tensors
    all_data_tensor = torch.tensor(all_data, dtype=torch.float32)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    print(f"all_data_tensor shape: {all_data_tensor.shape}")
    print(f"all_labels_tensor shape: {all_labels_tensor.shape}")

    # Under-sample the negative class
    print("Under-sampling the negative class to balance the dataset...")
    # Indices of positive and negative samples
    positive_indices = np.where(all_labels == 1)[0]
    negative_indices = np.where(all_labels == 0)[0]

    # Number of positive samples
    num_positive = len(positive_indices)

    # Under-sample negative indices
    np.random.seed(42)
    negative_sample_size = num_positive * 5 if num_positive * 5 <= len(negative_indices) else len(negative_indices)
    negative_indices_under = np.random.choice(negative_indices, size=negative_sample_size, replace=False)

    # Combine positive indices with under-sampled negative indices
    selected_indices = np.concatenate([positive_indices, negative_indices_under])

    # Shuffle selected indices
    np.random.shuffle(selected_indices)

    # Subset the data and labels
    all_data_tensor = all_data_tensor[selected_indices]
    all_labels_tensor = all_labels_tensor[selected_indices]

    # Update dataset size
    logger.info("After under-sampling:")
    calculate_class_distribution(all_labels_tensor)

    # Create datasets and loaders using stratified sampling
    dataset = SeismicDataset(all_data_tensor, all_labels_tensor, sequence_length)
    logger.info("Total sequences in dataset: %d", len(dataset))

    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare labels for stratification
    labels_for_split = np.array([all_labels_tensor[i + sequence_length - 1].item() for i in range(len(dataset))])
    logger.info("Prepared labels for stratification.")

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    indices = np.arange(len(dataset))
    for train_indices, val_indices in splitter.split(indices, labels_for_split):
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Adjust batch size and number of workers
    batch_size = 64  # Adjust as needed
    num_workers = 0   # Set to 0 for Windows or adjust as needed

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Calculate class weights for loss function
    num_positive = (all_labels_tensor == 1).sum().item()
    num_negative = (all_labels_tensor == 0).sum().item()
    if num_positive == 0:
        print("Error: No positive samples in the dataset.")
        return
    pos_weight = torch.tensor([num_negative / num_positive]).to(device)
    print(f"Positive class weight: {pos_weight.item():.6f}")

    # Initialize model with correct input size
    input_size = all_data_tensor.shape[1]  # Number of features
    print(f"Input size for the model: {input_size}")
    model = LSTMClassifier(input_size=input_size).to(device)

    # Train model
    print("Training model...")
    model, history = train_model(model, device, train_loader, val_loader, epochs=10, pos_weight=pos_weight, checkpoint_dir='checkpoints')

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
    loaded_model = LSTMClassifier(input_size=input_size).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    loaded_model.eval()

    # Process test file
    test_file = 'your_test_file_name'  # Replace with your test file name (without .csv extension)
    test_csv_file = f'{TEST_DATA_DIR}{test_file}.csv'

    if os.path.exists(test_csv_file):
        test_data = read_csv_file(test_csv_file)
        test_data = clean_data(test_data)  # Data cleansing
        test_data = normalize_data(test_data)  # Data normalization
        test_velocity = np.array(test_data['velocity(m/s)'].tolist())
        test_derivative = compute_derivative(test_velocity)
        test_rolling_mean, test_rolling_std = compute_rolling_stats(test_velocity, window_size=10)

        # Stack features
        test_features = np.stack((test_velocity, test_derivative, test_rolling_mean, test_rolling_std), axis=1)

        # Create sequences for test data
        logger.info("Creating sequences for test data")
        num_sequences = len(test_features) - sequence_length + 1
        if num_sequences <= 0:
            print("No test sequences created. Check the sequence length and test data size.")
            return
        X_test = np.array([test_features[i:i+sequence_length] for i in range(num_sequences)])
        # Convert to tensor
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Make predictions
        print("Making predictions on test data...")
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
                X_batch = X_test[i:i+batch_size]
                outputs = loaded_model(X_batch)
                probs = torch.sigmoid(outputs)  # Apply Sigmoid
                predictions.extend(probs.cpu().numpy())

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
            event_time = test_data['time_rel(sec)'].iloc[event + sequence_length - 1]
            print(f"Time: {event_time} seconds")
    else:
        print(f"Test file not found: {test_csv_file}")

if __name__ == "__main__":
    main()
