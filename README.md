# Signal Processing & Machine Learning for HRV Prediction
- An evaluation of the performance and recency of various Python signal processing packages when applied to ECG and PPG data.
- Using machine learning models like XGBoost, and neural networks for predicting HRV metrics like SDNN (standard deviation of NN-intervals) from PPG data

---

### Tested and evaluated the following packages in `\packages-exploration`:
- Neurokit2
- HeartPy
- SciPy
- NumPy
- PyWavelets
- Biosppy

---

### Extracted, Analyzed and Plotted Practice Data in `\practice-data-analysis`:
- Extracted both PPG and ECG data using pandas
- Plotted both PPG and ECG data using matplotlib
- Calculated standard deviation of RR Intervals in both polarHRM.csv
- Applied IIR filters like the butterworth filter for the preprocessing of the signal data
  - Used 3rd order butterworth filter
  - Used bandpass filter type
  - Applied cutoff frequences with the Nyquist rate
- Applied peak detection algorithm from the scipy.signal module
- Calculated HRV metrics like SDNN, SDRR, RMSSD, pNN50, HR Max, HR Min, etc for both the PPG and ECG data

---

### Exploring various models for predicting HRV metrics (SDNN) from Raw PPG data:
- Using CNNs, LSTMs, Transformers and tree ensemble models like XGBoost in `\models`

---

### Purpose:
- Treating this repository as a progress report for my internship.
- Constantly updating the repository with new tasks, code and lab data.

### XGBoost Classifier Pipeline:
![image](https://github.com/user-attachments/assets/f21b2dc2-fe4c-48bc-b29f-de32ea404f91)

---

### Run the Models:
- Train the Model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Showmick119/Benchmarking-Signal-Processing-Packages/blob/main/test/train_hrv_model.ipynb)
- Run Inference on the Model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Showmick119/Benchmarking-Signal-Processing-Packages/blob/main/test/test_hrv_model.ipynb)

---
