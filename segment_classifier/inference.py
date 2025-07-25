# inference.py - Inference script for PPG Segment Quality Classifier
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import random

from .classifier import SegmentCNNClassifier


class SegmentClassifierInference:
    """
    Inference wrapper for the trained PPG segment quality classifier.
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize the inference model.
        
        Args:
            model_path (str): Path to the saved .pt model file
            device (str, optional): Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']

        self.model = SegmentCNNClassifier(
            input_channels=self.config['model']['input_channels'],
            hidden_channels=self.config['model']['hidden_channels'],
            kernel_sizes=self.config['model']['kernel_sizes'],
            dropout=self.config['model']['dropout'],
            pool_size=self.config['model']['pool_size']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Model performance on test set:")
        if 'test_metrics' in checkpoint:
            test_metrics = checkpoint['test_metrics']
            print(f"  Test F1: {test_metrics['f1']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Test AUC: {test_metrics['auc']:.4f}")
    
    @torch.no_grad()
    def predict_single_segment(self, ppg_signal):
        """
        Predict quality for a single PPG segment.
        
        Args:
            ppg_signal (np.ndarray): PPG signal array
            
        Returns:
            dict: Prediction results with probability and class
        """
        if isinstance(ppg_signal, np.ndarray):
            if ppg_signal.dtype == object:
                ppg_signal = np.array(ppg_signal, dtype=np.float32)
            signal_tensor = torch.tensor(ppg_signal, dtype=torch.float32)
        else:
            ppg_signal = np.array(ppg_signal, dtype=np.float32)
            signal_tensor = torch.tensor(ppg_signal, dtype=torch.float32)

        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        probabilities = self.model(signal_tensor)
        probability = torch.sigmoid(probabilities).item()
        predicted_class = int(probability >= 0.5)
        
        return {
            'predicted_class': predicted_class,
            'probability_bad': 1 - probability,
            'probability_good': probability,
            'is_good_quality': bool(predicted_class)
        }


def plot_ppg_segment(ppg_signal, true_label, prediction, confidence, ax=None):
    """
    Plot a PPG segment with prediction information.
    
    Args:
        ppg_signal (np.ndarray): PPG signal to plot
        true_label (int): True label (0=bad, 1=good)
        prediction (int): Predicted label
        confidence (float): Model's confidence in good quality
        ax (matplotlib.axes, optional): Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(ppg_signal, 'b-', label='PPG Signal', alpha=0.7)

    title = f"True: {'Good' if true_label else 'Bad'} | "
    title += f"Predicted: {'Good' if prediction else 'Bad'} | "
    title += f"Confidence in Good: {confidence:.2f}"

    color = 'green' if prediction == true_label else 'red'
    ax.set_title(title, color=color, fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    
    return ax


def load_model(model_path="segment_checkpoints/best_segment_classifier.pt"):
    """
    Convenience function to load the trained model.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        SegmentClassifierInference: Loaded inference model
    """
    return SegmentClassifierInference(model_path)


def demo_inference(num_samples=4):
    """
    Enhanced demonstration of model inference with visualizations.
    
    Args:
        num_samples (int): Number of random samples to test
    """
    try:
        model = load_model()

        data_path = "data/Labeled_segments"
        all_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

        sample_files = random.sample(all_files, min(num_samples, len(all_files)))

        rows = (num_samples + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        print(f"\nTesting {len(sample_files)} random segments...")
        
        for idx, file_name in enumerate(sample_files):
            file_path = os.path.join(data_path, file_name)
            data = np.load(file_path, allow_pickle=True)
            
            ppg_signal = data['ppg_signal']
            true_label = int(data['label'].item())

            result = model.predict_single_segment(ppg_signal)

            row = idx // 2
            col = idx % 2
            plot_ppg_segment(
                ppg_signal, 
                true_label, 
                result['predicted_class'],
                result['probability_good'],
                ax=axes[row, col]
            )

            print(f"\nSegment {idx + 1} ({file_name}):")
            print(f"  True label: {'Good' if true_label else 'Bad'}")
            print(f"  Predicted: {'Good' if result['is_good_quality'] else 'Bad'}")
            print(f"  Confidence in Good: {result['probability_good']:.3f}")
            print(f"  Correct: {'✓' if result['predicted_class'] == true_label else '✗'}")

        plt.tight_layout()
        plt.show()
        
        print(f"\nInference demo completed!")
        
    except FileNotFoundError:
        print("Model file not found. Please train the model first using:")
        print("python -m segment_classifier.train")
    except Exception as e:
        print(f"Error during inference demo: {str(e)}")


if __name__ == "__main__":
    demo_inference() 