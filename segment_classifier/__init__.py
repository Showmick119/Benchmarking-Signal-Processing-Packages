# segment_classifier package
"""
PPG Segment Quality Classifier Package

This package provides functionality for training and running inference
with a CNN-based classifier to assess PPG signal quality.
"""

from .classifier import SegmentCNNClassifier, EarlyStopping
from .dataset import SegmentDataset, create_k_fold_datasets, create_data_loaders, collate_fn
from .inference import SegmentClassifierInference, load_model

__all__ = [
    'SegmentCNNClassifier', 'EarlyStopping',
    'SegmentDataset', 'create_k_fold_datasets', 'create_data_loaders', 'collate_fn',
    'SegmentClassifierInference', 'load_model'
] 