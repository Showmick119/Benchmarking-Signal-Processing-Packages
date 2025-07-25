# Loads .npz file to create train/test/val set
import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence


def create_dataset(data_dir, train_ratio = 0.7, val_ratio = 0.15, test_ratio= 0.15, random_seed = 0):
    """Create train/val/test datasets from directory"""
    full_dataset = PPGSignalDataset(data_dir)
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    return train_dataset, val_dataset, test_dataset


def collate_fn(batch):
    """
    batch: list of items from __getitem__(), each is a dict with keys: 'ppg', 'ecg', 'label', 'file_name'
    """
    batch_ppg = []
    batch_ecg = []
    batch_labels = []
    segment_lengths = []
    batch_file_names = []

    valid = [
        item for item in batch
        if (item is not None)
        and len(item['ppg']) > 0
        and len(item['ecg']) > 0
    ]

    if len(valid) == 0:
        return None
    
    for item in valid:
        ppg_segs = item['ppg']
        segment_lengths.append([seg.shape[0] for seg in ppg_segs]) # Original lengths

        # Each 'ppg' is a list of variable-length 1D tensors (segments)
        padded_ppg = pad_sequence(item['ppg'], batch_first = True)  # shape: (num_segments, max_length)
        padded_ecg = pad_sequence(item['ecg'], batch_first = True)
            
        batch_ppg.append(padded_ppg)
        batch_ecg.append(padded_ecg)

        batch_labels.append(item['label'])
        batch_file_names.append(item['file_name'])

    return {
        'ppg': batch_ppg,                   # list of (num_segments, max_len)
        'ecg': batch_ecg,
        'segment_lengths': segment_lengths, # list of lists (length of original/unmasked seq length)
        'label': torch.stack(batch_labels),
        'file_name': batch_file_names
    }


class PPGSignalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
            Args:
                data_dir (string): Directory with .npz files (each containing one sample)
                transform (optional): Optional transform to be applied to segments
        """
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

        # Pre-load file metadata
        self.sample_info = []
        for file_name in self.file_list:
            file_path = os.path.join(data_dir, file_name)
            with np.load(file_path, allow_pickle=True) as sample:
                self.sample_info.append({
                    'file_name': file_name,
                    'n_segments': len(sample['segments']),
                    'label': sample['label']
                })

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        try:
            with np.load(file_path, allow_pickle=True) as sample:
                segments = sample['segments']
                gold_label = sample['label']
                
                # Collect segments as lists (preserves variable-length)
                ppg_segments = [torch.tensor(seg['ppg_segment'], dtype=torch.float32) for seg in segments]
                ecg_segments = [torch.tensor(seg['ecg_segment'], dtype=torch.float32) for seg in segments]

                # Apply transforms if specified
                if self.transform:
                    ppg_segments = [self.transform(seg) for seg in ppg_segments]
                    ecg_segments = [self.transform(seg) for seg in ecg_segments]
                

                gold_label = torch.tensor(gold_label, dtype = torch.float32)

                # Return as dictionary
                return {
                    'ppg': ppg_segments,
                    'ecg': ecg_segments,
                    'label': gold_label,
                    'file_name': file_name
                }
        
        except Exception as e:
            #print (f"[WARN] Skipping {file_name} due to error: {e}")
            return None