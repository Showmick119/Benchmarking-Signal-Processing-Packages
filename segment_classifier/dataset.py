# dataset.py - Dataset utilities for PPG Segment Quality Classification
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter


def augment_ppg_signal(signal, noise_std=0.01, scale_range=(0.95, 1.05)):
    """
    Apply simple augmentation to PPG signal.
    
    Args:
        signal (np.array): Original PPG signal
        noise_std (float): Standard deviation for Gaussian noise
        scale_range (tuple): Range for scaling factor
    
    Returns:
        np.array: Augmented signal
    """
    augmented = signal.copy()
    
    # Add small amount of Gaussian noise
    noise = np.random.normal(0, noise_std * np.std(signal), len(signal))
    augmented += noise
    
    # Apply slight scaling
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    augmented *= scale_factor
    
    return augmented.astype(np.float32)


class SegmentDataset(Dataset):
    """
    Dataset for PPG segment quality classification.
    
    Loads .npz files containing PPG signals and quality labels.
    Each file should contain:
    - 'ppg_signal': 1D numpy array of PPG signal
    - 'label': Binary label (0=bad quality, 1=good quality)
    
    Includes robust error handling to skip corrupted/NaN files.
    Supports oversampling minority class for balanced training.
    """
    
    def __init__(self, file_paths, balance_classes=False):
        """
        Initialize the dataset.
        
        Args:
            file_paths (list): List of paths to .npz files
            balance_classes (bool): Whether to oversample minority class to 50-50 balance
        """
        self.file_paths = file_paths
        self.balance_classes = balance_classes
        
        # Pre-load and validate all data to filter out corrupted files
        self.valid_data = []
        self._load_and_validate_data()
        
        # Apply class balancing if requested
        if self.balance_classes:
            self._balance_dataset()
    
    def _load_and_validate_data(self):
        """Load and validate all data, filtering out corrupted files."""
        print(f"Loading {len(self.file_paths)} segment files...")
        
        skipped_files = []
        nan_files = []
        inf_files = []
        conversion_errors = []
        
        for file_path in self.file_paths:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    ppg_signal = data['ppg_signal']
                    label = data['label']

                    try:
                        ppg_signal = self._process_ppg_signal(ppg_signal)
                    except Exception as e:
                        conversion_errors.append((file_path, f"Signal error: {str(e)}"))
                        continue
                    
                    try:
                        label = self._process_label(label)
                    except Exception as e:
                        conversion_errors.append((file_path, f"Label error: {str(e)}"))
                        continue
                    
                    # Validate signal quality
                    if not self._is_valid_signal(ppg_signal):
                        if np.isnan(ppg_signal).any():
                            nan_files.append(file_path)
                        elif np.isinf(ppg_signal).any():
                            inf_files.append(file_path)
                        else:
                            skipped_files.append((file_path, "Invalid signal quality"))
                        continue
                    
                    # Store valid data
                    self.valid_data.append({
                        'signal': ppg_signal,
                        'label': label,
                        'file_path': file_path,
                        'is_augmented': False  # Track if this is an augmented sample
                    })
                    
            except Exception as e:
                skipped_files.append((file_path, str(e)))
                continue
        
        total_files = len(self.file_paths)
        valid_files = len(self.valid_data)
        
        print(f"Successfully loaded {valid_files} valid segment files")
        
        if skipped_files or nan_files or inf_files or conversion_errors:
            print(f"Filtered out {total_files - valid_files} problematic files:")
            if nan_files:
                print(f"  - {len(nan_files)} files with NaN values")
            if inf_files:
                print(f"  - {len(inf_files)} files with Inf values")
            if conversion_errors:
                print(f"  - {len(conversion_errors)} files with conversion errors")
            if skipped_files:
                print(f"  - {len(skipped_files)} files with other issues")
        
        if self.valid_data:
            labels = [item['label'] for item in self.valid_data]
            class_dist = Counter(labels)
            print(f"Original class distribution: {dict(class_dist)}")
    
    def _balance_dataset(self):
        """Balance dataset by oversampling minority class with augmentation."""
        if not self.valid_data:
            return
        
        # Separate classes
        class_0_samples = [item for item in self.valid_data if item['label'] == 0]
        class_1_samples = [item for item in self.valid_data if item['label'] == 1]
        
        print(f"Before balancing: Class 0: {len(class_0_samples)}, Class 1: {len(class_1_samples)}")
        
        # Find minority and majority class
        if len(class_0_samples) < len(class_1_samples):
            minority_samples = class_0_samples
            majority_samples = class_1_samples
            minority_label = 0
        else:
            minority_samples = class_1_samples
            majority_samples = class_0_samples
            minority_label = 1
        
        # Calculate how many augmented samples we need
        target_size = len(majority_samples)
        current_minority_size = len(minority_samples)
        samples_needed = target_size - current_minority_size
        
        print(f"Need to generate {samples_needed} augmented samples for class {minority_label}")
        
        # Store augmented samples
        augmented_samples = []
        np.random.seed(42)
        
        for i in range(samples_needed):
            # Randomly select a minority sample to augment
            base_sample = np.random.choice(minority_samples)
            
            # Create augmented version
            augmented_signal = augment_ppg_signal(
                base_sample['signal'],
                noise_std=0.01,  # Small noise - 1% of signal std
                scale_range=(0.98, 1.02)  # Slight scaling variation
            )
            
            # Create new sample
            augmented_sample = {
                'signal': augmented_signal,
                'label': base_sample['label'],
                'file_path': f"{base_sample['file_path']}_aug_{i}",
                'is_augmented': True
            }
            
            augmented_samples.append(augmented_sample)
        
        # Combine original and augmented data
        self.valid_data = class_0_samples + class_1_samples + augmented_samples
        
        # Report final distribution
        final_labels = [item['label'] for item in self.valid_data]
        final_dist = Counter(final_labels)
        augmented_count = sum(1 for item in self.valid_data if item['is_augmented'])
        
        print(f"After balancing: {dict(final_dist)}")
        print(f"Added {augmented_count} augmented samples")
        print(f"Final dataset size: {len(self.valid_data)}")
    
    def _process_ppg_signal(self, ppg_signal):
        """Process and validate PPG signal with robust error handling."""
        if isinstance(ppg_signal, np.ndarray) and ppg_signal.dtype == object:
            try:
                ppg_signal = np.array([float(x) for x in ppg_signal.flat], dtype=np.float32)
            except (ValueError, TypeError):
                try:
                    ppg_signal = ppg_signal.astype(np.float32)
                except:
                    ppg_signal = np.array(ppg_signal.tolist(), dtype=np.float32)
        else:
            ppg_signal = ppg_signal.astype(np.float32)
        
        if ppg_signal.ndim > 1:
            ppg_signal = ppg_signal.flatten()
        
        return ppg_signal
    
    def _process_label(self, label):
        """Process and validate label with robust error handling."""
        if isinstance(label, np.ndarray):
            if label.shape == ():
                label = int(label.item())
            elif len(label) == 1:
                label = int(label[0])
            else:
                raise ValueError(f"Invalid label shape: {label.shape}")
        elif isinstance(label, (int, float, np.integer, np.floating)):
            label = int(label)
        else:
            label = int(label)
        
        if label not in [0, 1]:
            raise ValueError(f"Label must be 0 or 1, got {label}")
        
        return label
    
    def _is_valid_signal(self, signal):
        """Check if signal is valid (no NaN/Inf, reasonable length, etc.)."""
        if np.isnan(signal).any() or np.isinf(signal).any():
            return False

        if len(signal) < 10:  # too short to be meaningful
            return False

        if np.std(signal) == 0:  # constant signal
            return False

        signal_range = np.max(signal) - np.min(signal)
        if signal_range > 10000 or abs(np.mean(signal)) > 5000:
            return False

        return True

    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if idx >= len(self.valid_data):
            return None
        
        data_item = self.valid_data[idx]
        signal = data_item['signal'].copy()  # create a copy to avoid modifying original
        label = data_item['label']
        
        try:
            signal_tensor = torch.from_numpy(signal).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            if torch.isnan(signal_tensor).any() or torch.isinf(signal_tensor).any():
                print(f"NaN/Inf detected in tensor conversion for {data_item['file_path']}")
                return None
            
            return {
                'signal': signal_tensor,
                'label': label_tensor,
                'lengths': torch.tensor(len(signal), dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Tensor conversion failed for {data_item['file_path']}: {e}")
            return None


def create_k_fold_datasets(data_path, k_folds=5, test_ratio=0.15, random_seed=42):
    """
    Create stratified K-fold datasets with robust error handling.
    
    Args:
        data_path (str): Path to directory containing .npz files
        k_folds (int): Number of folds for cross-validation
        test_ratio (float): Ratio of data to hold out for final testing
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (test_dataset, fold_datasets)
    """
    files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
    file_paths = [os.path.join(data_path, f) for f in files]
    
    print(f"Found {len(files)} .npz files in {data_path}")

    labels = []
    valid_files = []
    
    for file_path in file_paths:
        try:
            data = np.load(file_path, allow_pickle=True)
            label = data['label']
            
            if isinstance(label, np.ndarray):
                if label.shape == ():
                    label = int(label.item())
                elif len(label) == 1:
                    label = int(label[0])
                else:
                    continue  # skip invalid labels
            elif isinstance(label, (int, float, np.integer, np.floating)):
                label = int(label)
            else:
                continue  # skip invalid labels
            
            if label in [0, 1]:  # valid binary label
                labels.append(label)
                valid_files.append(file_path)
                
        except Exception:
            continue  # skip problematic files
    
    print(f"Successfully processed {len(valid_files)} valid files")
    if len(valid_files) != len(file_paths):
        print(f"Skipped {len(file_paths) - len(valid_files)} files during initial filtering")

    class_counts = Counter(labels)
    print(f"Overall class distribution: {dict(class_counts)}")
    
    np.random.seed(random_seed)
    
    # Stratified train-test split
    train_files, test_files, train_labels, test_labels = train_test_split(
        valid_files, labels, 
        test_size=test_ratio, 
        stratify=labels, 
        random_state=random_seed
    )
    
    print(f"Test set: {len(test_files)} files")
    print(f"Train+Val set: {len(train_files)} files")
    
    # creating TEST dataset WITHOUT balancing (original distribution to account for real world testing)
    test_dataset = SegmentDataset(test_files, balance_classes=False)
    
    # Create K-fold splits on training data
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    fold_datasets = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_files, train_labels)):
        fold_train_files = [train_files[i] for i in train_idx]
        fold_val_files = [train_files[i] for i in val_idx]
        
        # create TRAINING dataset WITH balancing (50-50)
        train_dataset = SegmentDataset(fold_train_files, balance_classes=True)
        
        # create VALIDATION dataset WITHOUT balancing (original distribution)
        val_dataset = SegmentDataset(fold_val_files, balance_classes=False)
        
        print(f"Fold {fold_idx + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        fold_datasets.append((train_dataset, val_dataset))
    
    return test_dataset, fold_datasets


def collate_fn(batch):
    """
    Custom collate function that handles None samples and variable-length sequences.
    
    Args:
        batch (list): List of samples from dataset
        
    Returns:
        dict: Batched data with padded sequences
    """
    # Filter out None samples
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        return None
    
    # Extract components
    signals = [item['signal'] for item in valid_batch]
    labels = torch.stack([item['label'] for item in valid_batch])
    lengths = torch.stack([item['lengths'] for item in valid_batch])
    
    # Pad sequences to same length
    max_length = max(len(signal) for signal in signals)
    padded_signals = torch.zeros(len(signals), max_length)
    
    for i, signal in enumerate(signals):
        padded_signals[i, :len(signal)] = signal
    
    return {
        'signal': padded_signals,
        'label': labels,
        'lengths': lengths
    }


def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset (already balanced at 50-50)
        val_dataset: Validation dataset (original distribution)
        batch_size: Batch size
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create training loader - simple shuffle since data is already balanced
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        pin_memory=True
    )
    
    # Create validation loader (no shuffling for consistent validation)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader