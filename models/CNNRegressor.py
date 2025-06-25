import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby2, sosfiltfilt
import random

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import XQRS

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def calculate_ecg_sdnn(dataset_name, data_dir='data'):
    """
    Calculate SDNN from ECG data (polar.csv) for a given dataset using XQRS.
    
    Args:
        dataset_name: Name of dataset (e.g., 'HRV102M_Finger_Baseline')
        data_dir: Base directory containing the data folders
    
    Returns:
        float: SDNN value in milliseconds, or None if file not found
    """
    # Construct the path to the polar.csv file
    possible_paths = [
        os.path.join(data_dir, 'HR_Lab_Data_Day_1_Mom', dataset_name, 'polar.csv'),
        os.path.join(data_dir, 'HR_Lab_Data_Day_1_Teen', dataset_name, 'polar.csv'),
        os.path.join(data_dir, 'HR_Lab_Data_Day_2_Mom', dataset_name, 'polar.csv'),
        os.path.join(data_dir, 'HR_Lab_Data_Day_2_Teen', dataset_name, 'polar.csv'),
    ]
    
    ecg_file = None
    for path in possible_paths:
        if os.path.exists(path):
            ecg_file = path
            break
    
    if ecg_file is None:
        print(f"Warning: Could not find polar.csv for {dataset_name}")
        return None
    
    try:
        # Load ECG data
        ecg_df = pd.read_csv(ecg_file)
        
        # Convert timestamp to relative time in milliseconds
        timestamps = ecg_df['timestamp_ms'].values
        ecg_signal = ecg_df['value'].values
        
        # Convert to relative time starting from 0
        relative_time = timestamps - timestamps[0]
        
        # Use XQRS algorithm for ECG peak detection
        xqrs = XQRS()
        peak_times, peak_indices = xqrs.find_peaks(relative_time, ecg_signal, 
                                                  search_window=5, refinement=True)
        
        if len(peak_times) < 2:
            print(f"Warning: Found only {len(peak_times)} peaks in ECG for {dataset_name}")
            return None
        
        # Calculate RR intervals
        rr_intervals = np.diff(peak_times)
        
        # Remove any negative or extremely small intervals
        rr_intervals = rr_intervals[rr_intervals > 0]
        
        if len(rr_intervals) == 0:
            print(f"Warning: No valid RR intervals found for {dataset_name}")
            return None
        
        # Calculate SDNN
        sdnn = np.std(rr_intervals, ddof=1)
        
        print(f"Calculated ECG SDNN for {dataset_name}: {sdnn:.3f} ms ({len(peak_times)} peaks)")
        return sdnn
        
    except Exception as e:
        print(f"Error calculating ECG SDNN for {dataset_name}: {e}")
        return None

def get_all_datasets():
    """
    Load all datasets from the new data structure and calculate dynamic SDNN values.
    
    Returns:
        tuple: (all_data, all_sdnn_values, dataset_names)
    """
    base_data_dir = "data"
    
    # Find all bangle.csv files in the data structure
    pattern = os.path.join(base_data_dir, "HR_Lab_Data_Day_*", "HRV*", "bangle.csv")
    data_paths = glob.glob(pattern)
    
    all_data = []
    all_sdnn_values = []
    dataset_names = []
    
    print(f"Found {len(data_paths)} potential datasets")
    
    for path in data_paths:
        try:
            # Extract dataset name from path
            path_parts = os.path.normpath(path).split(os.sep)
            dataset_name = path_parts[-2]  # e.g., 'HRV102M_Finger_Baseline'
            
            # Load PPG data
            df = pd.read_csv(path)
            df.iloc[:, 0] -= df.iloc[0, 0]  # Normalize time to start from 0
            
            # Calculate SDNN from ECG data
            sdnn = calculate_ecg_sdnn(dataset_name, base_data_dir)
            
            if sdnn is not None:
                all_data.append(df)
                all_sdnn_values.append(sdnn)
                dataset_names.append(dataset_name)
                print(f"Successfully loaded: {dataset_name}")
            else:
                print(f"Skipped {dataset_name}: Could not calculate SDNN")
                
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print(f"\nSuccessfully loaded {len(all_data)} datasets with valid SDNN values")
    return all_data, all_sdnn_values, dataset_names

# Load all datasets with dynamic SDNN calculation
data_list, sdnn_values, dataset_names = get_all_datasets()

if len(data_list) == 0:
    print("Error: No datasets loaded successfully. Please check data paths.")
    exit(1)

row_counts = [len(df) for df in data_list]

# Using maximum number of rows for padding approach
max_rows = max(row_counts)
print(f"Maximum number of rows: {max_rows}")
print(f"Row count statistics: min={min(row_counts)}, max={max(row_counts)}, mean={np.mean(row_counts):.1f}")

# Initializing Matrix X with padding
X = np.zeros((len(data_list), max_rows))

# Padding All Samples to max length
for i, raw_data in enumerate(data_list):
    ppg_values = raw_data.iloc[:, 1].values  # Extract PPG values (second column)
    sample_length = len(ppg_values)
    
    # Place PPG values at the beginning, rest remains zero (padding)
    X[i, :sample_length] = ppg_values
    
    if i < 5:  # Show first 5 for debugging
        print(f"  Dataset {i}: {sample_length} samples -> padded to {max_rows}")

print(f"Initial matrix X shape: {X.shape}")

# Filtering Matrix X
for i, raw_data in enumerate(data_list):
    time = raw_data.iloc[:, 0].values
    
    unique, counts = np.unique(np.diff(time), return_counts=True)
    most_frequent = unique[np.argmax(counts)]
    fs = 1/(most_frequent*1e-3)
    
    sos = cheby2(N=4, rs=40, Wn=[0.528, 8.0], btype='bandpass', fs=fs, output='sos')
    X[i, :] = sosfiltfilt(sos, X[i, :])
    
    X[i, :] = (X[i, :] - np.mean(X[i, :])) / np.std(X[i, :])

print("Filtering and normalization complete.")

# Reshape Matrix X for passing into CNN: Adding Depth (2D -> 3D)
X = X.reshape(X.shape[0], 1, X.shape[1])
print(f"Final matrix X shape: {X.shape}")

# Use dynamically calculated SDNN values as target labels
y = np.array(sdnn_values)

# Remove NaN values and extreme outliers that cause training instability
valid_mask = ~np.isnan(y) & (y < 1000)  # Remove NaN and values > 1000ms (extreme outliers)
X = X[valid_mask]
y = y[valid_mask]
dataset_names = [name for i, name in enumerate(dataset_names) if valid_mask[i]]

print(f"Loaded {len(y)} valid SDNN target values (removed {sum(~valid_mask)} outliers/NaN):")
print(f"SDNN range: {y.min():.3f} - {y.max():.3f} ms")
print(f"SDNN mean ± std: {y.mean():.3f} ± {y.std():.3f} ms")

for i, (name, sdnn) in enumerate(zip(dataset_names[:10], y[:10])):  # Show first 10 only
    print(f"  {i+1:2d}. {name}: {sdnn:.3f} ms")
if len(y) > 10:
    print(f"  ... and {len(y)-10} more datasets")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, train_size=0.70, random_state=42, shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, train_size=0.50, random_state=42, shuffle=True
)

train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                         torch.from_numpy(y_train).float())
val_ds = TensorDataset(torch.from_numpy(X_val).float(),
                       torch.from_numpy(y_val).float())
test_ds = TensorDataset(torch.from_numpy(X_test).float(),
                        torch.from_numpy(y_test).float())

# Change batch_size when we have more datasets
train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

class CNNRegressor(nn.Module):
    def __init__(self, input_length):
        super(CNNRegressor, self).__init__()

        # 1st Set Of: Convolutional Layer -> Batch Normalization -> ReLU -> Pooling
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(num_features=32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 2nd Set Of: Convolution Layer -> Batch Normalization -> ReLU -> Pooling
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
        self.batchnorm2 = nn.BatchNorm1d(num_features=64)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 3rd Set Of (Potential Overfitting Problem; Model Won't Generalize): Convolution Layer -> Batch Normalization -> ReLU -> Pooling
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='same')
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Flatten Outputs:
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # Apply Regularization Technique: Dropout:
        self.dropout = nn.Dropout(p=0.30)

        # Calculate the flattened size dynamically based on input length
        # After 3 pooling layers (each divides by 2): input_length // (2^3) = input_length // 8
        flattened_size = 128 * (input_length // 8)
        
        # Fully Connected Layers:
        self.fc1 = nn.Linear(in_features=flattened_size, out_features=64)
        self.act4 = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=128, out_features=64)
        # self.act5 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.pool1(self.act1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.act2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.act3(self.batchnorm3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act4(x)
        # x = self.fc2(x)
        # x = self.act5(x)
        x = self.fc3(x)
        return x.squeeze(-1)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Fix: Pass the actual input length to the model constructor
# This calculates the correct fully connected layer size based on our data dimensions
model = CNNRegressor(input_length=max_rows).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

'L2 Regularization With Weight Decay:'
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

'Learning Rate Scheduler:'
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val_loss = float('inf')
patience, trials = 10, 0

for epoch in range(1, 101):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        train_preds = model(xb)
        loss = criterion(train_preds, yb)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    train_loss = np.mean(train_losses)
    print(f"Epoch {epoch} Train MSE: {train_loss:.4f}")

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch} Val MSE: {val_loss:.4f}")
    # scheduler.step(val_loss)
    for param_group in optimizer.param_groups:
        print(f"Current LR: {param_group['lr']:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        trials = 0
        print(f"New best validation loss: {val_loss:.4f}")
    else:
        trials += 1
        if trials >= patience:
            print('Early Stopping.')
            break

# Load best model if it was saved, otherwise use current model
if os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt'))
    print("Loaded best model from training")
else:
    print("No best model saved, using current model")
model.eval()

# Test on final test set and generate predictions
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(yb.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"\n{'='*50}")
print("FINAL PREDICTIONS AND RESULTS")
print(f"{'='*50}")

# **PREDICTIONS ARE GENERATED HERE** - Model outputs SDNN predictions for each PPG sample
print(f"Generated {len(y_pred)} predictions from CNN model")
print(f"True SDNN values range: {y_true.min():.3f} - {y_true.max():.3f} ms")
print(f"Predicted SDNN values range: {y_pred.min():.3f} - {y_pred.max():.3f} ms")

# Check for any remaining NaN values in predictions
if np.any(np.isnan(y_pred)):
    print(f"Warning: {np.sum(np.isnan(y_pred))} NaN predictions detected")
    valid_preds = ~np.isnan(y_pred)
    y_true = y_true[valid_preds]
    y_pred = y_pred[valid_preds]
    
    if len(y_true) == 0:
        print("Error: All predictions are NaN. Model architecture needs debugging.")
        print("The model is producing invalid outputs due to dimension mismatch or gradient issues.")
        exit(1)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Test samples: {len(y_true)}")
print(f"Test MSE: {mse:.2f} ms²")
print(f"Test RMSE: {rmse:.2f} ms")
print(f"Test MAE: {mae:.2f} ms")
print(f"R²: {r2:.3f}")

print(f"\nPrediction statistics:")
print(f"True SDNN range: {y_true.min():.2f} - {y_true.max():.2f} ms")
print(f"Predicted SDNN range: {y_pred.min():.2f} - {y_pred.max():.2f} ms")

plt.figure(figsize=(10, 8))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.xlabel('True SDNN (ms)')
plt.ylabel('Predicted SDNN (ms)')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
plt.legend()
plt.title(f'CNN Regressor: True vs Predicted SDNN\nR² = {r2:.3f}, RMSE = {rmse:.2f} ms')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()