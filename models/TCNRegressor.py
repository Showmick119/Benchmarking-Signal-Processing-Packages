import os
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        
        # Very light dropout
        self.dropout = 0.05
        
        # First conv block
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        
        # Second conv block
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        
        # Simple skip connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Skip connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Ensure matching sizes
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        
        return self.relu(out + res)

class TCNRegressor(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=7):
        super(TCNRegressor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(4, num_channels[0], 1),  # 4 input channels
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )
        
        # TCN blocks
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[0] if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Calculate padding to maintain same size
            padding = (kernel_size-1) * dilation_size // 2
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding
                )
            )
        
        self.temporal_blocks = nn.Sequential(*layers)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Simple MLP head
        hidden_size = num_channels[-1]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # TCN blocks
        x = self.temporal_blocks(x)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)
        
        # MLP head
        return self.mlp(x).squeeze(1)

def get_baseline_data():
    base_path = "Data/HR_Lab_Data_Day_1"
    pattern = os.path.join(base_path, "HRV*M_Finger_Baseline", "bangle.csv")
    data_paths = glob.glob(pattern)
    
    all_data = []
    excluded_ids = ['104', '105', '106', '122']
    
    for path in data_paths:
        try:
            hrv_id = path.split('HRV')[1][:3]
            if hrv_id not in excluded_ids:
                df = pd.read_csv(path)
                df.iloc[:, 0] -= df.iloc[0, 0]
                all_data.append(df)
                print(f"Successfully loaded: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return all_data

data_list = get_baseline_data()

# Process data
row_counts = [len(df) for df in data_list]
median_rows = int(np.median(row_counts))
print(f"Median number of rows: {median_rows}")

X = np.zeros((len(data_list), median_rows))

# First pass: Interpolate and normalize each signal
for i, raw_data in enumerate(data_list):
    raw_time = raw_data.iloc[:, 0]
    raw_ppg = raw_data.iloc[:, 1]
    
    # Interpolate to uniform time
    max_time = raw_time.max()
    uniform_time = np.linspace(0, max_time, median_rows)
    X[i, :] = np.interp(uniform_time, raw_time, raw_ppg)
    
    # Normalize each signal individually
    X[i, :] = (X[i, :] - np.mean(X[i, :])) / np.std(X[i, :])

print(f"Final matrix X shape: {X.shape}")

# Second pass: Apply bandpass filter and extract features
for i, raw_data in enumerate(data_list):
    time = raw_data.iloc[:, 0].values
    
    unique, counts = np.unique(np.diff(time), return_counts=True)
    most_frequent = unique[np.argmax(counts)]
    fs = 1/(most_frequent*1e-3)
    
    sos = cheby2(N=4, rs=40, Wn=[0.528, 8.0], btype='bandpass', fs=fs, output='sos')
    X[i, :] = sosfiltfilt(sos, X[i, :])

print("Filtering complete.")
print(X)

# Add multiple feature channels
X_diff = np.diff(X, axis=1)
X_diff = np.pad(X_diff, ((0, 0), (1, 0)), mode='edge')

# Add rolling statistics
window_size = 100
X_roll_mean = np.zeros_like(X)
X_roll_std = np.zeros_like(X)
for i in range(X.shape[0]):
    X_roll_mean[i, :] = pd.Series(X[i, :]).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    X_roll_std[i, :] = pd.Series(X[i, :]).rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill').values

# Stack all features
X = np.stack([X, X_diff, X_roll_mean, X_roll_std], axis=1)

# Target values
y = np.array([
    31.946, 41.216, 31.524, 9.416, 70.15, 23.034, 30.905, 46.93, 24.44, 
    26.755, 100.092, 92.542, 41.698, 40.531, 45.938, 28.936, 41.964, 
    35.34, 34.128, 86.691
])

# Log transform targets (since SDNN is always positive)
y = np.log1p(y)

# Split data with more training data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                         torch.from_numpy(y_train).float())
val_ds   = TensorDataset(torch.from_numpy(X_val).float(),
                         torch.from_numpy(y_val).float())
test_ds  = TensorDataset(torch.from_numpy(X_test).float(),
                         torch.from_numpy(y_test).float())

train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=len(val_ds))
test_loader  = DataLoader(test_ds, batch_size=len(test_ds))

# Initialize model with simpler architecture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCNRegressor(
    input_size=X_train.shape[2],
    num_channels=[32, 64, 96],  # Even smaller network
    kernel_size=7
).to(device)

# Use a learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

# Training loop with early stopping
best_val_loss = float('inf')
patience, trials = 15, 0

for epoch in range(1, 151):  # Fewer epochs
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_losses.append(loss.item())
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_losses.append(criterion(model(xb), yb).item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch}  Train MSE: {train_loss:.4f}  Val MSE: {val_loss:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_tcn.pt')
        trials = 0
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping.")
            break

# Load best model
model.load_state_dict(torch.load('best_model_tcn.pt'))
model.eval()

# Evaluate on test set
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).float()
    test_preds = model(X_test_tensor.to(device)).cpu().numpy()
    
# Inverse transform predictions and true values
test_preds = np.expm1(test_preds)
y_test = np.expm1(y_test)

# Calculate metrics
test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))
test_mae = np.mean(np.abs(test_preds - y_test))
test_r2 = r2_score(y_test, test_preds)

print(f"\nTest RMSE: {test_rmse:.2f} ms")
print(f"Test MAE:  {test_mae:.2f} ms")
print(f"Test R²:   {test_r2:.2f}")

# Plot predictions vs true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True SDNN (ms)')
plt.ylabel('Predicted SDNN (ms)')
plt.title('TCN Predictions vs True Values')
plt.grid(True)
plt.savefig('tcn_predictions.png')
plt.close() 