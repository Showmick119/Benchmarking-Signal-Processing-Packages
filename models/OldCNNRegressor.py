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

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

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

row_counts = [len(df) for df in data_list]

# Using median number of rows, as it minimizes interpolation error the most
median_rows = int(np.median(row_counts))
print(f"Median number of rows: {median_rows}")

# Initializing Matrix X
X = np.zeros((len(data_list), median_rows))

# Interpolating All Samples
for i, raw_data in enumerate(data_list):
    raw_time = raw_data.iloc[:, 0]
    raw_ppg = raw_data.iloc[:, 1]
    
    max_time = raw_time.max()
    uniform_time = np.linspace(0, max_time, median_rows)
    X[i, :] = np.interp(uniform_time, raw_time, raw_ppg)

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

y = np.array([
    31.946, 
    41.216, 
    31.524, 
    9.416, 
    70.15, 
    23.034, 
    30.905, 
    46.93, 
    24.44, 
    26.755, 
    100.092, 
    92.542, 
    41.698, 
    40.531, 
    45.938,
    28.936,
    41.964,
    35.34,
    34.128,
    86.691
])

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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        sample = torch.zeros(1, 1, X_train.shape[2])
        out = self.pool2(self.conv2(self.pool1(self.conv1(sample))))
        flat_size = out.numel()
        self.fc1  = nn.Linear(flat_size, 64)
        self.fc2  = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CNNRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

best_val_loss = float('inf')
patience, trials = 10, 0

for epoch in range(1, 101):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_losses.append(criterion(model(xb), yb).item())
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch}  Val MSE: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        trials = 0
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping.")
            break

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(yb.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Test MSE: {mse:.2f} ms")
print(f"Test RMSE: {rmse:.2f} ms")
print(f"Test MAE: {mae:.2f} ms")
print(f"R²: {r2:.2f} ms")

plt.figure()
plt.scatter(y_true, y_pred)
plt.xlabel('True SDNN (ms)')
plt.ylabel('Predicted SDNN (ms)')
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()])
plt.grid()
plt.tight_layout()
plt.show()