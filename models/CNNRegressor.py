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

        # Applying Attention
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Flatten Outputs:
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # Apply Regularization Technique: Dropout:
        self.dropout = nn.Dropout(p=0.30)

        # Fully Connected Layers:
        self.fc1 = nn.Linear(in_features=128*685, out_features=64)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.pool1(self.act1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.act2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.act3(self.batchnorm3(self.conv3(x))))

        x = x.permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x