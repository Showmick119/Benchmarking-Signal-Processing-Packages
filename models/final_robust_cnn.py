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
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import XQRS

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def calculate_ecg_sdnn(dataset_name, data_dir='data'):
    """Calculate SDNN from ECG data using XQRS"""
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
        return None
    
    try:
        ecg_df = pd.read_csv(ecg_file)
        timestamps = ecg_df['timestamp_ms'].values
        ecg_signal = ecg_df['value'].values
        relative_time = timestamps - timestamps[0]
        
        xqrs = XQRS()
        peak_times, peak_indices = xqrs.find_peaks(relative_time, ecg_signal, 
                                                  search_window=5, refinement=True)
        
        if len(peak_times) < 2:
            return None
        
        rr_intervals = np.diff(peak_times)
        rr_intervals = rr_intervals[rr_intervals > 0]
        
        if len(rr_intervals) == 0:
            return None
        
        sdnn = np.std(rr_intervals, ddof=1)
        return sdnn
        
    except Exception:
        return None

def safe_filter_and_normalize(signal, fs, min_length=100):
    """Safely filter and normalize a signal, handling edge cases"""
    signal = np.array(signal, dtype=np.float64)
    
    # Check for invalid input
    if len(signal) < min_length:
        return np.zeros_like(signal)
    
    # Remove any existing NaN/inf values
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If signal is all zeros or constant, return normalized version
    if np.std(signal) < 1e-10:
        return np.zeros_like(signal)
    
    try:
        # Apply bandpass filter
        sos = cheby2(N=4, rs=40, Wn=[0.528, 8.0], btype='bandpass', fs=fs, output='sos')
        filtered_signal = sosfiltfilt(sos, signal)
        
        # Check for NaN/inf after filtering
        filtered_signal = np.nan_to_num(filtered_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize: zero mean, unit variance
        mean_val = np.mean(filtered_signal)
        std_val = np.std(filtered_signal)
        
        if std_val < 1e-10:  # Constant signal
            return np.zeros_like(filtered_signal)
        else:
            normalized = (filtered_signal - mean_val) / std_val
            return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
    except Exception:
        # If filtering fails, just normalize the original signal
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val < 1e-10:
            return np.zeros_like(signal)
        else:
            normalized = (signal - mean_val) / std_val
            return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

def get_all_datasets():
    """Load ALL datasets with safe preprocessing"""
    base_data_dir = "data"
    pattern = os.path.join(base_data_dir, "HR_Lab_Data_Day_*", "HRV*", "bangle.csv")
    data_paths = glob.glob(pattern)
    
    all_data = []
    all_sdnn_values = []
    dataset_names = []
    
    print(f"Loading ALL {len(data_paths)} datasets...")
    
    for i, path in enumerate(data_paths):
        try:
            path_parts = os.path.normpath(path).split(os.sep)
            dataset_name = path_parts[-2]
            
            df = pd.read_csv(path)
            df.iloc[:, 0] -= df.iloc[0, 0]  # Normalize time to start from 0
            
            # Only include datasets with sufficient data
            if len(df) < 1000:
                continue
            
            sdnn = calculate_ecg_sdnn(dataset_name, base_data_dir)
            
            if sdnn is not None and 10 < sdnn < 500:  # Reasonable SDNN range
                all_data.append(df)
                all_sdnn_values.append(sdnn)
                dataset_names.append(dataset_name)
                
                if i % 50 == 0:
                    print(f"  Loaded {len(all_data)} datasets so far...")
                
        except Exception:
            continue
    
    print(f"Successfully loaded {len(all_data)} datasets with valid SDNN values")
    return all_data, all_sdnn_values, dataset_names

# Robust CNN Model
class FinalRobustCNN(nn.Module):
    def __init__(self, input_length):
        super(FinalRobustCNN, self).__init__()
        
        # Aggressive pooling to handle large inputs
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(8)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(8)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        
        # Global average pooling for size independence
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Small fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x.squeeze(-1)

if __name__ == "__main__":
    print("Loading ALL datasets with robust preprocessing...")
    data_list, sdnn_values, dataset_names = get_all_datasets()
    
    if len(data_list) < 20:
        print(f"Need at least 20 datasets, got {len(data_list)}")
        exit(1)
    
    # Use reasonable fixed length
    fixed_length = 3000
    print(f"Using truncation to {fixed_length} samples")
    
    # Process data with robust preprocessing
    X = np.zeros((len(data_list), fixed_length))
    valid_datasets = []
    
    print("Processing PPG data with robust filtering...")
    for i, raw_data in enumerate(data_list):
        try:
            ppg_values = raw_data.iloc[:, 1].values
            time_values = raw_data.iloc[:, 0].values
            
            # Calculate sampling frequency
            if len(time_values) > 1:
                time_diffs = np.diff(time_values)
                # Remove outliers in time differences
                valid_diffs = time_diffs[(time_diffs > 0) & (time_diffs < 100)]
                if len(valid_diffs) > 0:
                    avg_dt = np.median(valid_diffs)
                    fs = 1 / (avg_dt * 1e-3)  # Convert to Hz
                else:
                    fs = 50  # Default fallback
            else:
                fs = 50
            
            # Ensure reasonable sampling frequency
            fs = max(10, min(fs, 200))
            
            # Truncate to fixed length
            sample_length = min(len(ppg_values), fixed_length)
            signal_segment = ppg_values[:sample_length]
            
            # Apply safe filtering and normalization
            processed_signal = safe_filter_and_normalize(signal_segment, fs)
            
            # Place in matrix
            X[i, :sample_length] = processed_signal[:sample_length]
            
            # Check for any remaining invalid values
            if not (np.isfinite(X[i, :]).all()):
                X[i, :] = 0  # Zero out problematic signals
                
            valid_datasets.append(i)
            
            if i % 100 == 0:
                print(f"  Processed {i}/{len(data_list)} datasets...")
                
        except Exception as e:
            print(f"  Error processing dataset {i}: {e}")
            X[i, :] = 0  # Zero out on error
    
    # Filter out datasets with all zeros
    non_zero_mask = np.any(X != 0, axis=1)
    X = X[non_zero_mask]
    sdnn_values = np.array(sdnn_values)[non_zero_mask]
    dataset_names = [dataset_names[i] for i in range(len(dataset_names)) if non_zero_mask[i]]
    
    print(f"Kept {len(X)} datasets after filtering")
    
    # Final data check
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(sdnn_values)
    
    # Reshape for CNN
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    print(f"Final data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"SDNN range: {y.min():.3f} - {y.max():.3f} ms")
    print(f"Data statistics: X min={X.min():.3f}, max={X.max():.3f}, mean={X.mean():.3f}")
    
    # Train/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"\nDataset splits:")
    print(f"Training:   {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples") 
    print(f"Test:       {len(X_test)} samples")
    
    # Normalize targets
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    # Create data loaders
    batch_size = 16
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_norm).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val_norm).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test_norm).float())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = FinalRobustCNN(fixed_length).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining final robust model...")
    best_val_loss = float('inf')
    
    for epoch in range(30):
        # Training
        model.train()
        train_loss = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'final_robust_model.pt')
    
    # Load best model and test
    if os.path.exists('final_robust_model.pt'):
        model.load_state_dict(torch.load('final_robust_model.pt'))
    
    print("\n" + "="*80)
    print("FINAL ROBUST CNN - PREDICTIONS GENERATED HERE - NO NaN VALUES!")
    print("="*80)
    
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(yb.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Denormalize
    y_true = y_true * y_std + y_mean
    y_pred = y_pred * y_std + y_mean
    
    print(f"Generated {len(y_pred)} CLEAN predictions!")
    print(f"True SDNN range: {y_true.min():.3f} - {y_true.max():.3f} ms")
    print(f"Predicted SDNN range: {y_pred.min():.3f} - {y_pred.max():.3f} ms")
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print("FINAL ROBUST CNN RESULTS - ALL NaN ISSUES RESOLVED!")
    print(f"{'='*60}")
    print(f"Test samples:     {len(y_pred)}")
    print(f"MSE:              {mse:.2f} ms²")
    print(f"RMSE:             {rmse:.2f} ms")
    print(f"MAE:              {mae:.2f} ms")
    print(f"R² Score:         {r2:.3f}")
    print(f"Prediction range: {y_pred.max() - y_pred.min():.3f} ms")
    
    # Show sample predictions
    print(f"\nSample Predictions:")
    print("True SDNN | Predicted | Error")
    print("-" * 35)
    for i in range(min(15, len(y_true))):
        error = abs(y_true[i] - y_pred[i])
        print(f"{y_true[i]:9.3f} | {y_pred[i]:9.3f} | {error:5.3f}")
    
    print(f"\n🎉 SUCCESS: Model trained on {len(data_list)} datasets without NaN issues!")
    print(f"📊 Pipeline: Raw PPG → CNN → SDNN predictions vs ECG ground truth") 