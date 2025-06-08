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

### Exploring various models for predicting HRV metrics like SDNN from PPG data:
- Using CNNs, TCNs, XGBoost in `\models`
---
