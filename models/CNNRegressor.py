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

        # Fully Connected Layers:
        self.fc1 = nn.Linear(in_features=128*685, out_features=64)
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

model = CNNRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

'L2 Regularization With Weight Decay:'
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

'Learning Rate Scheduler:'
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

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
    else:
        trials += 1
        if trials >= patience:
            print('Early Stopping.')
            break

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(yb.cpu().numpy())

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
         [y_pred.min(), y_pred.max()])
plt.grid()
plt.tight_layout()
plt.show()