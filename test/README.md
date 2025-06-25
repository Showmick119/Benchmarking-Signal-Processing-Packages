# HRV Peak Classification Notebooks

This directory contains two Google Colab notebooks for training and testing XGBoost models for heart rate variability (HRV) peak classification.

## 🚀 **Current Status: Local Testing Mode**
Both notebooks are configured for **local development and stress-testing** with you. Once verified, simply uncomment the Colab sections for deployment.

## Notebooks

### 1. `train_hrv_model.ipynb` - Training Notebook
**Purpose**: Train an XGBoost classifier using corrected training data from `model-data/participant_datasets/`

**What it does**:
- Loads corrected training data (manually labeled peaks from Day 1 Mom datasets)
- Trains XGBoost model with hyperparameter tuning (70-30 split)
- Saves trained model and scaler as `.pkl` files **to `test/` directory**
- Displays training metrics, confusion matrix, and feature importance
- Runs SDNN analysis on training data

**Output**: `test/trained_xgboost_model.pkl` and `test/feature_scaler.pkl`

### 2. `test_hrv_model.ipynb` - Testing Notebook  
**Purpose**: Test the trained model on completely unseen datasets

**What it does**:
- Loads pre-trained model and scaler **from `test/` directory**
- Discovers all available datasets from 4 data subdirectories (**470 datasets found!**)
- Processes selected dataset on-the-fly (extracts features from raw `bangle.csv` data)
- Runs model inference to classify peaks as true/false
- Visualizes initial vs corrected peaks
- Calculates and compares SDNN values (initial vs corrected vs ground truth ECG)

**Key Features**:
- Interactive dataset selection from all available data
- Real-time feature extraction (no pre-saved CSV files needed)
- Two visualization plots: before and after model correction
- Comprehensive SDNN comparison with ground truth ECG data
- Error analysis and improvement metrics

## 🧪 **Stress Testing Results**

### ✅ **Training Notebook Tests**
- **Data Loading**: ✓ 16,328 samples from 97 datasets loaded successfully
- **Feature Structure**: ✓ 15 features per sample verified
- **Class Balance**: ✓ 89.9% true peaks, 10.1% false peaks
- **Model Saving**: ✓ Saves to `test/` directory for easy access
- **Environment**: ✓ Local testing mode works perfectly

### ✅ **Testing Notebook Tests**  
- **Dataset Discovery**: ✓ 470 datasets found across all 4 subdirectories
- **Data Loading**: ✓ `bangle.csv` (PPG) and `polar.csv` (ECG) loaded correctly
- **Feature Extraction**: ✓ 139 peaks detected from test dataset
- **Peak Rate**: ✓ 66.4 peaks/min (physiologically reasonable)
- **File Structure**: ✓ Model loads from `test/` directory

## Usage Instructions

### Prerequisites
Both notebooks automatically handle dependencies in **local testing mode**

### Running the Training Notebook (Local Testing)
1. Open `train_hrv_model.ipynb` in Jupyter/VS Code
2. Run all cells sequentially
3. Wait for training to complete (may take several minutes)
4. Model files saved to `test/` directory automatically

### Running the Testing Notebook (Local Testing)
1. **First**: Make sure you've run the training notebook
2. Open `test_hrv_model.ipynb` in Jupyter/VS Code  
3. Run cells sequentially until dataset selection
4. **Select a dataset**: Enter the number (1-470) 
5. Continue running cells to see results

### **🚀 Deploying to Google Colab**
When ready for Colab deployment:
1. **Uncomment** the git clone and installation sections
2. **Comment out** the local testing sections
3. Upload to Colab and run

### Expected Results
The testing notebook will show:
- **Initial peak detection plot**: All detected peaks in blue
- **Model-corrected plot**: True peaks (blue), false peaks marked for removal (red)  
- **SDNN comparison chart**: Initial PPG SDNN vs Corrected PPG SDNN vs Ground Truth ECG SDNN
- **Performance metrics**: Error reduction, improvement statistics

## Dataset Coverage
- **Training data**: Uses corrected datasets from `model-data/participant_datasets/` (Day 1 Mom only)
- **Testing data**: Can test on **ANY of 470 datasets** from all 4 subdirectories:
  - `HR_Lab_Data_Day_1_Mom/` (including uncorrected ones)
  - `HR_Lab_Data_Day_1_Teen/`
  - `HR_Lab_Data_Day_2_Mom/`
  - `HR_Lab_Data_Day_2_Teen/`

## Technical Details
- **Data Files**: Uses `bangle.csv` for PPG data, `polar.csv` for ECG ground truth
- **Feature extraction**: 15 morphological features calculated on-the-fly
- **Model type**: XGBoost classifier with hyperparameter optimization
- **SDNN calculation**: Uses specialized `calculate_sdnn_corrected()` function for model predictions
- **Ground truth**: ECG-based SDNN using XQRS algorithm for comparison
- **Visualization**: Professional plots with clear before/after comparisons

## Notes
- **Current Mode**: Local testing and stress-testing with development team
- **Deployment Ready**: Simply uncomment Colab sections when ready
- Testing notebook processes raw data and does NOT use pre-saved CSV files
- Model is trained only on Day 1 Mom corrected data but can generalize to other datasets
- Each run of the testing notebook can analyze a different dataset
- Ground truth ECG SDNN may not be available for all datasets 