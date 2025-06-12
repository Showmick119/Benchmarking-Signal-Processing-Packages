import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import cheby2, sosfiltfilt
import random
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

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

# Load and preprocess data
data_list = get_baseline_data()

row_counts = [len(df) for df in data_list]
median_rows = int(np.median(row_counts))
print(f"Median number of rows: {median_rows}")

X = np.zeros((len(data_list), median_rows))

# Interpolate data to uniform length
for i, raw_data in enumerate(data_list):
    raw_time = raw_data.iloc[:, 0]
    raw_ppg = raw_data.iloc[:, 1]
    
    max_time = raw_time.max()
    uniform_time = np.linspace(0, max_time, median_rows)
    X[i, :] = np.interp(uniform_time, raw_time, raw_ppg)

print(f"Final matrix X shape: {X.shape}")

# Apply filtering and normalization
for i, raw_data in enumerate(data_list):
    time = raw_data.iloc[:, 0].values
    
    unique, counts = np.unique(np.diff(time), return_counts=True)
    most_frequent = unique[np.argmax(counts)]
    fs = 1/(most_frequent*1e-3)
    
    sos = cheby2(N=4, rs=40, Wn=[0.528, 8.0], btype='bandpass', fs=fs, output='sos')
    X[i, :] = sosfiltfilt(sos, X[i, :])
    
    X[i, :] = (X[i, :] - np.mean(X[i, :])) / np.std(X[i, :])

print("Filtering and normalization complete.")

# Target values (SDNN)
y = np.array([
    31.946, 41.216, 31.524, 9.416, 70.15, 23.034, 30.905, 46.93, 
    24.44, 26.755, 100.092, 92.542, 41.698, 40.531, 45.938,
    28.936, 41.964, 35.34, 34.128, 86.691
])

# Split data into train, validation, and test sets (70-15-15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 1.0, 5.0]
}

# Initialize base model
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

print("Best parameters found:")
print(random_search.best_params_)

# Get best model
best_model = random_search.best_estimator_

# Make predictions on validation set
val_predictions = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"\nValidation RMSE: {val_rmse:.2f} ms")

# Make predictions on test set
test_predictions = best_model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f"\nTest Set Metrics:")
print(f"RMSE: {rmse:.2f} ms")
print(f"MAE:  {mae:.2f} ms")
print(f"R²:   {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True SDNN (ms)')
plt.ylabel('Predicted SDNN (ms)')
plt.title('XGBoost: True vs Predicted SDNN')
plt.tight_layout()
plt.show()

# Feature importance plot
plt.figure(figsize=(12, 6))
xgb.plot_importance(best_model, max_num_features=20)
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.show()

# Save the best model
best_model.save_model('best_xgboost_model.json')