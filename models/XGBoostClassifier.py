import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import XQRS

# np.random.seed(2)
# random.seed(2)

class XGBoostPeakClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        
        # Feature names for the 15 calculated features
        self.feature_names = [
            'amplitude', 'prominence', 'width_half_prom', 'pulse_area',
            'rise_time', 'decay_time', 'max_upslope', 'max_inflection',
            'ibi_prev', 'ibi_next', 'ibi_ratio', 'local_variance', 'snr',
            'freq_energy', 'wavelet_coef'
        ]
        
    @staticmethod
    def calculate_ecg_sdnn(dataset_name, data_dir=None):
        """
        Calculate SDNN from ECG data (polar.csv) for a given dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., 'HRV102M_Finger_Baseline')
            data_dir: Base directory containing the data folders (auto-detected if None)
        
        Returns:
            float: SDNN value in milliseconds, or None if file not found
        """
        # Smart path detection if data_dir not specified
        if data_dir is None:
            current_dir = os.getcwd()
            if 'test' in os.path.basename(current_dir):
                # Running from test/ directory (local)
                data_dir = '../data'
            else:
                # Running from project root (Colab)
                data_dir = 'data'
        
        # Construct the path to the polar.csv file
        # Need to figure out which data folder it's in (Day_1_Mom, Day_1_Teen, etc.)
        possible_paths = [
            os.path.join(data_dir, 'HR_Lab_Data_Day_1_Mom', dataset_name, 'polar.csv'),
            os.path.join(data_dir, 'HR_Lab_Data_Day_1_Teen', dataset_name, 'polar.csv'),
            os.path.join(data_dir, 'HR_Lab_Data_Day_2_Mom', dataset_name, 'polar.csv'),
            os.path.join(data_dir, 'HR_Lab_Data_Day_2_Teen', dataset_name, 'polar.csv'),
        ]
        
        ecg_file = None
        for path in possible_paths:
            if os.path.exists(path):
                ecg_file = path
                break
        
        if ecg_file is None:
            print(f"Warning: Could not find polar.csv for {dataset_name}")
            return None
        
        try:
            # Load ECG data
            ecg_df = pd.read_csv(ecg_file)
            
            # Convert timestamp to relative time in milliseconds
            timestamps = ecg_df['timestamp_ms'].values
            ecg_signal = ecg_df['value'].values
            
            # Convert to relative time starting from 0
            relative_time = timestamps - timestamps[0]
            
            # Use XQRS algorithm for ECG peak detection
            xqrs = XQRS()
            peak_times, peak_indices = xqrs.find_peaks(relative_time, ecg_signal, 
                                                      search_window=5, refinement=True)
            
            if len(peak_times) < 2:
                print(f"Warning: Found only {len(peak_times)} peaks in ECG for {dataset_name}")
                return None
            
            # Calculate RR intervals
            rr_intervals = np.diff(peak_times)
            
            # Remove any negative or extremely small intervals
            rr_intervals = rr_intervals[rr_intervals > 0]
            
            if len(rr_intervals) == 0:
                print(f"Warning: No valid RR intervals found for {dataset_name}")
                return None
            
            # Calculate SDNN
            sdnn = np.std(rr_intervals, ddof=1)
            
            print(f"Calculated ECG SDNN for {dataset_name}: {sdnn:.3f} ms ({len(peak_times)} peaks)")
            return sdnn
            
        except Exception as e:
            print(f"Error calculating ECG SDNN for {dataset_name}: {e}")
            return None
        
    def load_data(self, data_dir='model-data'):
        """
        Load the training data from CSV files in model-data/participant_datasets directory

        Args:
            data_dir: Base directory containing the data folders
        
        Returns:
            np.array: Matrix X for model training and testing
            np.array: Vectory y for model training and testing
        """

        # Look for CSV files in the participant_datasets subdirectory
        participant_dir = os.path.join(data_dir, 'participant_datasets')
        
        # Fallback to root model-data for backward compatibility
        if os.path.exists(participant_dir):
            csv_files = glob.glob(os.path.join(participant_dir, '*_features.csv'))
            print(f"Loading data from: {participant_dir}")
        else:
            csv_files = glob.glob(os.path.join(data_dir, '*_features.csv'))
            print(f"Loading data from: {data_dir} (fallback)")
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}. Make sure you've run prepare_data.py first.")
        
        print(f"Found {len(csv_files)} CSV files:")
        
        all_data = []
        dataset_info = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dataset_name = os.path.basename(csv_file).replace('_features.csv', '')
            
            print(f"  - {dataset_name}: {len(df)} peaks")
            
            # Extract features (columns 2-16, excluding metadata and label)
            features = df[self.feature_names].values
            labels = df['label'].values
            peak_times = df['peak_time'].values
            
            all_data.append({
                'features': features,
                'labels': labels,
                'peak_times': peak_times,
                'dataset_name': dataset_name
            })
            
            dataset_info.append({
                'name': dataset_name,
                'total_peaks': len(df),
                'true_peaks': np.sum(labels == 1),
                'false_peaks': np.sum(labels == 0)
            })
        
        # Combine all datasets
        self.X = np.vstack([data['features'] for data in all_data])
        self.y = np.concatenate([data['labels'] for data in all_data])
        self.dataset_info = dataset_info
        self.all_data = all_data
        
        print(f"\nCombined dataset:")
        print(f"Total samples: {self.X.shape[0]}")
        print(f"Features: {self.X.shape[1]}")
        print(f"True peaks: {np.sum(self.y == 1)}")
        print(f"False peaks: {np.sum(self.y == 0)}")
        print(f"Class balance: {np.sum(self.y == 1) / len(self.y) * 100:.1f}% true peaks")
        
        # Display dataset info
        print(f"\nDataset breakdown:")
        for info in dataset_info:
            print(f"  {info['name']}: {info['total_peaks']} peaks ({info['true_peaks']} true, {info['false_peaks']} false)")
        
        return self.X, self.y

    def train(self, random_state=2):
        """
        Train the XGBoost classifier with a 70/30 train/test split

        Args:
            random_state: The seed for RNG

        Returns:
            XGB: The trained classifier model
        """
        
        # Scale the features with z-score normalization
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Split into 70% train, 30% test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.3, random_state=random_state, stratify=self.y
        )
        
        print(f"Data splits:")
        print(f"  Training: {len(X_train)} samples ({len(X_train)/len(self.X)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(self.X)*100:.1f}%)")
        
        # Store splits for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]
        }
        
        # Create base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=random_state,
            eval_metric='logloss',
            enable_categorical=False
        )
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nTraining the XGBoost Classifier with hyperparameter tuning...")
        print("This may take a couple minutes...")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        
        # COMMENTED OUT: Show best parameters (now handled in dedicated notebook cell)
        # print("\nBest parameters:", self.best_params)
        
        print("\nTest set metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_test_pred):.3f}")
        print(f"Recall: {recall_score(y_test, y_test_pred):.3f}")
        print(f"F1 Score: {f1_score(y_test, y_test_pred):.3f}")
        
        # COMMENTED OUT: Plot confusion matrix (now handled in dedicated notebook cell)
        # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # 
        # # Test confusion matrix
        # cm_test = confusion_matrix(y_test, y_test_pred)
        # sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax)
        # ax.set_title('Test Set Confusion Matrix')
        # ax.set_ylabel('True Label')
        # ax.set_xlabel('Predicted Label')
        # 
        # plt.tight_layout()
        # plt.show()

        # COMMENTED OUT: Feature importance plot (now handled in dedicated notebook cell)
        # self.plot_feature_importance()
        
        return self.model

    def plot_feature_importance(self):
        """Plot feature importance scores"""
        if self.model is None:
            print("Model not trained yet!")
            return
            
        importance = self.model.feature_importances_
        
        # Create a DataFrame for easier plotting
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_df, x='importance', y='feature', palette='viridis')
        plt.title('XGBoost Feature Importance for Peak Classification')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        # Print top features
        print("\nTop 10 most important features:")
        for i, (feature, imp) in enumerate(feature_df.tail(10)[['feature', 'importance']].values[::-1], 1):
            print(f"{i:2d}. {feature:15s}: {imp:.4f}")

    def predict(self, X):
        """Predict peak labels for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale the features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    @staticmethod
    def calculate_sdnn(peak_times):
        """Calculate SDNN from initially detected peak times (in milliseconds)"""
        if len(peak_times) < 2:
            return 0.0
            
        # Calculate RR intervals (time differences between consecutive peaks)
        rr_intervals = np.diff(peak_times)
        
        # Remove any negative or extremely small intervals (artifacts)
        rr_intervals = rr_intervals[rr_intervals > 0]
        
        if len(rr_intervals) == 0:
            return 0.0
        
        # Calculate SDNN (standard deviation of RR intervals)
        sdnn = np.std(rr_intervals, ddof=1)
        
        return sdnn
    
    @staticmethod
    def calculate_sdnn_corrected(peak_times, labels):
        """
        Calculate SDNN from peak times, but only for consecutive true peaks (label=1)
        that don't have false peaks (label=0) between them.
        
        Args:
            peak_times: Array of peak times
            labels: Array of labels (1=true peak, 0=false peak)
        """
        if len(peak_times) < 2 or len(peak_times) != len(labels):
            return 0.0
        
        # Find consecutive groups of true peaks (label=1)
        true_peak_indices = np.where(labels == 1)[0]
        
        if len(true_peak_indices) < 2:
            return 0.0
        
        # Find consecutive runs of true peaks
        rr_intervals = []
        
        for i in range(len(true_peak_indices) - 1):
            current_idx = true_peak_indices[i]
            next_idx = true_peak_indices[i + 1]
            
            # Check if they are consecutive (no false peaks between them)
            if next_idx == current_idx + 1:
                # Calculate RR interval between consecutive true peaks
                rr_interval = peak_times[next_idx] - peak_times[current_idx]
                if rr_interval > 0:  # Sanity check
                    rr_intervals.append(rr_interval)
        
        if len(rr_intervals) == 0:
            return 0.0
        
        # Calculate SDNN (standard deviation of RR intervals)
        sdnn = np.std(rr_intervals, ddof=1)
        
        return sdnn

    def apply_peak_correction(self, dataset_name):
        """Apply trained model to correct peaks for a specific dataset"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Find the dataset in our loaded data
        dataset = None
        for data in self.all_data:
            if data['dataset_name'] == dataset_name:
                dataset = data
                break
        
        if dataset is None:
            raise ValueError(f"Dataset {dataset_name} not found!")
        
        # Get original features and peak times
        features = dataset['features']
        original_labels = dataset['labels']
        peak_times = dataset['peak_times']
        
        # Make predictions
        corrected_labels = self.predict(features)
        
        # Filter peaks based on corrected labels
        true_peak_mask = corrected_labels == 1
        corrected_peak_times = peak_times[true_peak_mask]
        original_true_peak_times = peak_times[original_labels == 1]
        
        # Calculate SDNN values
        original_sdnn = self.calculate_sdnn(original_true_peak_times)
        corrected_sdnn = self.calculate_sdnn_corrected(peak_times, corrected_labels)
        
        # Calculate ground truth SDNN from ECG data
        ground_truth_sdnn = self.calculate_ecg_sdnn(dataset_name)
        
        return {
            'dataset_name': dataset_name,
            'original_labels': original_labels,
            'corrected_labels': corrected_labels,
            'peak_times': peak_times,
            'original_true_peak_times': original_true_peak_times,
            'corrected_peak_times': corrected_peak_times,
            'original_sdnn': original_sdnn,
            'corrected_sdnn': corrected_sdnn,
            'ground_truth_sdnn': ground_truth_sdnn
        }

    def compare_sdnn_results(self, results_list):
        """Compare SDNN results across all datasets"""
        print("\n" + "="*80)
        print("SDNN COMPARISON RESULTS")
        print("="*80)
        
        comparison_data = []
        
        for results in results_list:
            dataset_name = results['dataset_name']
            original_sdnn = results['original_sdnn']
            corrected_sdnn = results['corrected_sdnn']
            ground_truth_sdnn = results['ground_truth_sdnn']
            
            print(f"\nDataset: {dataset_name}")
            print(f"  Original SDNN (with false peaks): {original_sdnn:.3f} ms")
            print(f"  Corrected SDNN (model filtered):  {corrected_sdnn:.3f} ms")
            
            if ground_truth_sdnn is not None:
                print(f"  Ground Truth SDNN (ECG):          {ground_truth_sdnn:.3f} ms")
                
                original_error = abs(original_sdnn - ground_truth_sdnn)
                corrected_error = abs(corrected_sdnn - ground_truth_sdnn)
                improvement = original_error - corrected_error
                
                print(f"  Original error:     {original_error:.3f} ms ({100*original_error/ground_truth_sdnn:.1f}%)")
                print(f"  Corrected error:    {corrected_error:.3f} ms ({100*corrected_error/ground_truth_sdnn:.1f}%)")
                print(f"  Improvement:        {improvement:.3f} ms ({'↑' if improvement > 0 else '↓'}{abs(improvement):.3f})")
                
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Original_SDNN': original_sdnn,
                    'Corrected_SDNN': corrected_sdnn,
                    'Ground_Truth_SDNN': ground_truth_sdnn,
                    'Original_Error': original_error,
                    'Corrected_Error': corrected_error,
                    'Improvement': improvement
                })
            else:
                print(f"  Ground Truth SDNN: Not available")
                
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Original_SDNN': original_sdnn,
                    'Corrected_SDNN': corrected_sdnn,
                    'Ground_Truth_SDNN': None,
                    'Original_Error': None,
                    'Corrected_Error': None,
                    'Improvement': None
                })
        
        # Create summary
        df = pd.DataFrame(comparison_data)
        valid_comparisons = df.dropna()
        
        if len(valid_comparisons) > 0:
            print(f"\n" + "="*40)
            print("SUMMARY")
            print("="*40)
            print(f"Average original error:   {valid_comparisons['Original_Error'].mean():.3f} ms")
            print(f"Average corrected error:  {valid_comparisons['Corrected_Error'].mean():.3f} ms")
            print(f"Average improvement:      {valid_comparisons['Improvement'].mean():.3f} ms")
            print(f"Datasets improved:        {sum(valid_comparisons['Improvement'] > 0)}/{len(valid_comparisons)}")
        
        return comparison_data

    def run_full_analysis(self):
        """Run complete analysis on all datasets"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        results_list = []
        
        for data in self.all_data:
            dataset_name = data['dataset_name']
            print(f"\nAnalyzing {dataset_name}...")
            
            results = self.apply_peak_correction(dataset_name)
            results_list.append(results)
            
            print(f"  Original peaks: {sum(results['original_labels'] == 1)}")
            print(f"  Corrected peaks: {sum(results['corrected_labels'] == 1)}")
            print(f"  Peaks removed: {sum(results['original_labels'] == 1) - sum(results['corrected_labels'] == 1)}")
        
        # Compare SDNN results
        comparison_data = self.compare_sdnn_results(results_list)
        
        return results_list, comparison_data


if __name__ == "__main__":
    # Initialize the classifier
    classifier = XGBoostPeakClassifier()
    
    # Load data from CSV files
    print("Loading data from model-data/ folder...")
    classifier.load_data()
    
    # Train the model with 70-30 split
    print("\nTraining XGBoost classifier...")
    classifier.train()
    
    # Run full analysis on all datasets
    print("\nRunning full analysis...")
    results_list, comparison_data = classifier.run_full_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("The model has been trained and evaluated on peak classification.")
    print("SDNN comparisons show how well the model improves HRV accuracy.")
    print("Use classifier.apply_peak_correction(dataset_name) for individual dataset analysis.")