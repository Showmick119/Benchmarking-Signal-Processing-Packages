import pandas as pd
import numpy as np
from scipy.signal import butter, cheby1, cheby2, ellip, firwin, lfilter, medfilt
from scipy.signal.windows import hamming, hann

'''Tasks:
- Plot the ECG and PPG data
- Calculate the Standard Deviation of the RRIs (the time interval in milliseconds, between successsive R-wave peaks) (in the polarHRM.csv)
- Try some of the processing techniques, like IIR and FIR filters, bandpass filtering, etc
- Find peaks within the signal
'''

data = pd.read_csv("Practice-HRV/bangle.csv")