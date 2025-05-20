import pandas as pd
import numpy as np
from scipy.signal import butter, cheby1, cheby2, ellip, firwin, lfilter, medfilt
from scipy.signal.windows import hamming, hann

'''Outlined Tasks:
- Plot the ECG and PPG data
- Calculate the Standard Deviation of the RRIs (the time interval in milliseconds, between successsive R-wave peaks) (in the polarHRM.csv)
- Try some of the processing techniques, like IIR and FIR filters, bandpass filtering, etc
- Find peaks within the signal
'''

"Loading All Datasets:"
bangle_data = pd.read_csv("Practice-HRV/bangle.csv")
print(bangle_data.head())

polar_data = pd.read_csv("Practice-HRV/polar.csv")
print(polar_data.head())

"""Note: Issue with polarHRM.csv, as it had 4 columns/features for some of data samples (rows), which was causing errors when 
loading the data. Why is this the case? As we should identify the issue, as it could lead to problems when working with larger
datasets, as this one is only a 2 minute reading."""

polarHRM_data = pd.read_csv("Practice-HRV/polarHRM.csv")
print(polarHRM_data.head())