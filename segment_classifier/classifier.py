"""
CNN Model for PPG Segment Quality Classification
"""

import torch
import torch.nn as nn


class SegmentCNNClassifier(nn.Module):
    """
    CNN-based classifier for PPG segment quality assessment.
    Binary classification: 0=bad quality, 1=good quality
    """
    
    def __init__(self, input_channels=1, hidden_channels=None, kernel_sizes=None, dropout=0.2, pool_size=2):
        super(SegmentCNNClassifier, self).__init__()
        # Default architecture
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        
        # 1st CNN Block: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels[0], 
                              kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_channels[0])
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        
        # 2nd CNN Block: Conv -> BatchNorm -> ReLU -> MaxPool  
        self.conv2 = nn.Conv1d(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                              kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_channels[1])
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        
        # 3rd CNN Block: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv3 = nn.Conv1d(in_channels=hidden_channels[1], out_channels=hidden_channels[2],
                              kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=hidden_channels[2])
        self.act3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        
        # Global Average Pooling (handles variable lengths)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Flatten
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=hidden_channels[2], out_features=128)
        self.batchnorm_fc1 = nn.BatchNorm1d(num_features=128)
        self.act4 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.batchnorm_fc2 = nn.BatchNorm1d(num_features=64)
        self.act5 = nn.ReLU(inplace=True)
        
        # Changed final layer to output single value
        self.fc3 = nn.Linear(in_features=64, out_features=1)

        # Added sigmoid activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths=None):
        # Ensure input has correct shape: (batch_size, channels, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros(x.size(0), device=x.device)
        
        # CNN feature extraction
        x = self.pool1(self.act1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.act2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.act3(self.batchnorm3(self.conv3(x))))
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = self.flatten(x)
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.batchnorm_fc1(x)
        x = self.act4(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm_fc2(x)
        x = self.act5(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        
        # Final check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros(x.size(0), device=x.device)
        
        return x.squeeze(-1)  # Return (batch_size,) tensor


class EarlyStopping:
    """Simple early stopping utility."""
    
    def __init__(self, patience=7, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False 