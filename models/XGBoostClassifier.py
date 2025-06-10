import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby2, sosfiltfilt
import random
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class XGBoostPeakClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        
    def load_data(self, data_path='peak_training_data.npz'):
        """Load the training data prepared by prepare_peak_data.py"""
        data = np.load(data_path)
        self.X = data['X']
        self.y = data['y']
        self.feature_names = data['feature_names']

        n_base_features = 7
        self.X_base = self.X[:, :n_base_features]
        self.X_waveform = self.X[:, n_base_features:]
        
        self.X_base_scaled = self.scaler.fit_transform(self.X_base)
        
        self.X_processed = np.hstack([self.X_base_scaled, self.X_waveform])
        
        print(f"Loaded data: {self.X_processed.shape[0]} samples, {self.X_processed.shape[1]} features")
        print(f"Class distribution: {np.sum(self.y == 1)} true peaks, {np.sum(self.y == 0)} false peaks")

    def train(self, random_state=42):
        """Train the XGBoost classifier with hyperparameter tuning"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, self.y, test_size=0.2, random_state=random_state, stratify=self.y
        )
        
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]
        }
        
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Training model with hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        y_pred = self.model.predict(X_test)
        
        print("\nBest parameters:", self.best_params)
        print("\nTest set metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_pred):.3f}")
        print(f"Recall: {recall_score(y_test, y_pred):.3f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        self.plot_feature_importance()

    def plot_feature_importance(self):
        """Plot feature importance scores"""
        importance = self.model.feature_importances_
        base_features = self.feature_names[:7]

        base_importance = importance[:7]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=base_importance, y=base_features)
        plt.title('Feature Importance (Base Features)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        """Predict peak labels for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_base = X[:, :7]
        X_waveform = X[:, 7:]
        X_base_scaled = self.scaler.transform(X_base)

        X_processed = np.hstack([X_base_scaled, X_waveform])
        
        return self.model.predict(X_processed)

    @staticmethod
    def calculate_sdnn(peak_times):
        """Calculate SDNN from peak times"""
        rr_intervals = np.diff(peak_times)

        sdnn = np.std(rr_intervals)
        
        return sdnn

    def compare_sdnn(self, original_peak_times, corrected_peak_times):
        """Compare SDNN between original and corrected peaks"""
        original_sdnn = self.calculate_sdnn(original_peak_times)
        corrected_sdnn = self.calculate_sdnn(corrected_peak_times)
        
        print("\nSDNN Comparison:")
        print(f"Original SDNN: {original_sdnn:.3f} ms")
        print(f"Corrected SDNN: {corrected_sdnn:.3f} ms")
        print(f"Absolute difference: {abs(original_sdnn - corrected_sdnn):.3f} ms")
        print(f"Relative difference: {100 * abs(original_sdnn - corrected_sdnn) / original_sdnn:.1f}%")
        
        return original_sdnn, corrected_sdnn

if __name__ == "__main__":
    classifier = XGBoostPeakClassifier()
    classifier.load_data()
    classifier.train()