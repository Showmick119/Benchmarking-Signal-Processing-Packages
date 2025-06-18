import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
random.seed(42)

class XGBoostPeakClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        
        # Ground truth SDNN values from ECG data (extensible as more datasets are processed)
        self.ground_truth_sdnn = {
            'HRV102M_Finger_Baseline': 31.946,
            'HRV103M_Finger_Baseline': 41.216,
            'HRV108M_Finger_Baseline': 31.524,
            'HRV113M_Finger_Baseline': 9.416,
            'HRV114M_Finger_Baseline': 70.15,
            'HRV115M_Finger_Baseline': 23.034,
            'HRV117M_Finger_Baseline': 30.905,
            'HRV118M_Finger_Baseline': 46.93,
            'HRV119M_Finger_Baseline': 24.44,
            'HRV121M_Finger_Baseline': 26.755,
            'HRV123M_Finger_Baseline': 100.092,
            'HRV124M_Finger_Baseline': 92.542,
            'HRV125M_Finger_Baseline': 41.698,
            'HRV126M_Finger_Baseline': 40.531,
            'HRV127M_Finger_Baseline': 45.938,
            'HRV128M_Finger_Baseline': 28.936,
            'HRV129M_Finger_Baseline': 41.964,
            'HRV130M_Finger_Baseline': 35.34,
            'HRV132M_Finger_Baseline': 34.128,
            'HRV134M_Finger_Baseline': 86.691
        }
        
        # Feature names for the 15 calculated features
        self.feature_names = [
            'amplitude', 'prominence', 'width_half_prom', 'pulse_area',
            'rise_time', 'decay_time', 'max_upslope', 'max_inflection',
            'ibi_prev', 'ibi_next', 'ibi_ratio', 'local_variance', 'snr',
            'freq_energy', 'wavelet_coef'
        ]
        
    def add_ground_truth_sdnn(self, dataset_name, sdnn_value):
        """Add or update ground truth SDNN value for a dataset"""
        self.ground_truth_sdnn[dataset_name] = sdnn_value
        print(f"Updated ground truth SDNN for {dataset_name}: {sdnn_value:.3f} ms")
    
    def get_ground_truth_sdnn_dict(self):
        """Get the current ground truth SDNN dictionary"""
        return self.ground_truth_sdnn.copy()
        
    def load_data(self, data_dir='model-data'):
        """Load the training data from CSV files in model-data directory"""
        csv_files = glob.glob(os.path.join(data_dir, '*_features.csv'))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
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

    def train(self, random_state=42):
        """Train the XGBoost classifier with 70-15-15 train/validation/test split"""
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # First split: 70% train, 30% temp (which will be split into 15% val, 15% test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, self.y, test_size=0.3, random_state=random_state, stratify=self.y
        )
        
        # Second split: Split the 30% temp into 15% validation and 15% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )
        
        print(f"Data splits:")
        print(f"  Training: {len(X_train)} samples ({len(X_train)/len(self.X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(self.X)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(self.X)*100:.1f}%)")
        
        # Store splits for later use
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        # Define parameter grid for hyperparameter tuning (simplified for faster testing)
        param_grid = {
            'max_depth': [4, 6],
            'learning_rate': [0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1],
            'gamma': [0],
            'subsample': [0.9],
            'colsample_bytree': [0.9],
            'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]
        }
        
        # Create base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=random_state,
            eval_metric='logloss'
        )
        
        # Perform grid search with cross-validation on training data
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nTraining model with hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        
        print("\nBest parameters:", self.best_params)
        
        print("\nValidation set metrics:")
        print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.3f}")
        print(f"Precision: {precision_score(y_val, y_val_pred):.3f}")
        print(f"Recall: {recall_score(y_val, y_val_pred):.3f}")
        print(f"F1 Score: {f1_score(y_val, y_val_pred):.3f}")
        
        print("\nTest set metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_test_pred):.3f}")
        print(f"Recall: {recall_score(y_test, y_test_pred):.3f}")
        print(f"F1 Score: {f1_score(y_test, y_test_pred):.3f}")
        
        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validation confusion matrix
        cm_val = confusion_matrix(y_val, y_val_pred)
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Validation Set Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Test confusion matrix
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Test Set Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()

        self.plot_feature_importance()
        
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
    
    def predict_proba(self, X):
        """Predict peak probabilities for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale the features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)

    @staticmethod
    def calculate_sdnn(peak_times):
        """Calculate SDNN from peak times (in milliseconds)"""
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
        probabilities = self.predict_proba(features)[:, 1]  # Probability of being a true peak
        
        # Filter peaks based on corrected labels
        true_peak_mask = corrected_labels == 1
        corrected_peak_times = peak_times[true_peak_mask]
        original_true_peak_times = peak_times[original_labels == 1]
        
        # Calculate SDNN values
        original_sdnn = self.calculate_sdnn(original_true_peak_times)
        corrected_sdnn = self.calculate_sdnn(corrected_peak_times)
        
        # Get ground truth SDNN if available
        ground_truth_sdnn = self.ground_truth_sdnn.get(dataset_name, None)
        
        return {
            'dataset_name': dataset_name,
            'original_labels': original_labels,
            'corrected_labels': corrected_labels,
            'probabilities': probabilities,
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
    
    # Train the model with 70-15-15 split
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