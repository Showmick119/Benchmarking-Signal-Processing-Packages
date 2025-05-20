import pandas as pd
import numpy as np
from scipy.signal import butter, cheby1, cheby2, ellip, firwin, lfilter, medfilt
from scipy.signal.windows import hamming, hann

data = pd.read_csv("Practice-HRV/bangle.csv")