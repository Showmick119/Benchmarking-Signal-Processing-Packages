import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby2, sosfiltfilt, sosfreqz
from scipy.signal import butter, freqz, lfilter, filtfilt
from PyEMD import EEMD, EMD
import wfdb
import wfdb.processing
import os


class LoadDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_data(self):
        bangle_data_path = os.path.join(self.data_path, 'bangle.csv')
        ppg_data = pd.read_csv(f'{self.data_path}/bangle.csv')
        ecg_data = pd.read_csv(f'{self.data_path}/polar.csv')

        # Append 0's to the 4th column for polarHRM.csv
        rows = []
        with open(f'{self.data_path}/polarHRM.csv', 'r') as f:
            next(f) # Skip the header
            for line in f:
                parts = line.strip().split(',')

                # skip empty lines
                if len(parts) < 3:
                    continue

                # Pad with 0 if there's no second RRi
                # if len(parts) == 3:
                #     parts.append('0')
                
                parts = parts[:4]
                while len(parts) < 4:
                    parts.append('0')

                rows.append(parts)
        
        self.polar_HRM = pd.DataFrame(rows, columns=['timestamp_ms', 'value', 'rris1', 'rris2']) 
        self.polar_HRM['timestamp_ms'] = pd.to_numeric(self.polar_HRM['timestamp_ms'])
        self.polar_HRM['value'] = pd.to_numeric(self.polar_HRM['value'])
        self.polar_HRM['rris1'] = pd.to_numeric(self.polar_HRM['rris1'])
        self.polar_HRM['rris2'] = pd.to_numeric(self.polar_HRM['rris2'])

        # Align timestamp for each signal to start from 0
        self.time_ppg = (ppg_data['timestamp_ms'] - ppg_data['timestamp_ms'][0]).to_numpy()
        self.ppg = ppg_data['value'].to_numpy()
        self.time_ecg = (ecg_data['timestamp_ms'] - ecg_data['timestamp_ms'][0]).to_numpy()
        self.ecg = ecg_data['value'].to_numpy()

        assert (np.diff(self.time_ppg) == np.diff(ppg_data['timestamp_ms'])).all(), "Cannot alignn PPG timestamp to 0. Precision Loss."
        assert (np.diff(self.time_ecg) == np.diff(ecg_data['timestamp_ms'])).all(), "Cannot alignn ECG timestamp to 0. Precision Loss."

        #   Align timestamp for polar_HRM signal
        offset = self.polar_HRM['timestamp_ms'][0] - ppg_data['timestamp_ms'][0]
        self.time_polarHRM = self.polar_HRM['timestamp_ms'] - self.polar_HRM['timestamp_ms'][0] + offset
        assert (np.diff(self.time_polarHRM) == np.diff(self.polar_HRM['timestamp_ms'])).all(), "Cannot align polar_HRM."

        return self.time_ecg, self.ecg, self.time_ppg, self.ppg, self.time_polarHRM, self.polar_HRM


    def plot_raw_data(self, start_time=0, end_time=None):
        """
            Args:
                start_time: lower limit (time in ms)
                end_time: upper limit (time in ms)

            Returns:
                Plots for raw data
        """

        fig, axs = plt.subplots(3, 1, figsize=(12, 9))

        # --- Subplot 1: ECG ---
        axs[0].plot(self.time_ecg, self.ecg, label='ECG', color='tab:blue')
        axs[0].set_ylabel('ECG Signal')
        axs[0].set_xlim([start_time, end_time])
        axs[0].legend()
        axs[0].set_title('ECG')

        # --- Subplot 2: PPG ---
        axs[1].plot(self.time_ppg, self.ppg, label='PPG', color='tab:red')
        axs[1].set_xlim([start_time, end_time])
        axs[1].set_ylabel('PPG Signal')
        axs[1].legend()
        axs[1].set_title('PPG')

        # --- Subplot 3: Polar HRM RRIs ---
        axs[2].scatter(self.time_polarHRM, self.polar_HRM['rris1'], label='RRis1', s=10, color='tab:blue')

        valid_rris2 = self.polar_HRM['rris2'] != 0
        axs[2].scatter(self.time_polarHRM[valid_rris2], self.polar_HRM['rris2'][valid_rris2], label='RRis2', s=50, color='tab:red', marker = 'x')
        axs[2].set_ylabel('RR Interval (ms)')
        axs[2].set_xlabel('Time (ms -- aligned units)')
        axs[2].legend()
        axs[2].set_title('Polar HRM RR Intervals')

        plt.tight_layout()
        plt.show()


class Filters:
    def __init__(self):
        pass
    

    def plot(self, time, original_signal, filtered_signal, title, signal_type, start = 0, end = None):
        plt.figure(figsize=(15,4))
        plt.title(title)
        plt.plot(time, original_signal, alpha = 0.8, label = signal_type)
        plt.plot(time, filtered_signal, 'k--', label = 'Filtered' + signal_type)
        plt.xlim([start, end])
        plt.legend()
        plt.show()


    def moving_average(self, time, signal, window_size = 10, signal_type = 'PPG', plot = True, start = 0, end = None):
        filtered_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        if plot:
            self.plot(time, signal, filtered_signal, 'Moving Average Filter', signal_type, start, end)

        return filtered_signal


    def butter_filter(self, time, signal, order = 4, signal_type = 'PPG', plot = True, start = 0, end = None):
        """
            signal   : 1D numpy array (PPG signal)
            fs     : Sampling frequency in Hz
            cutoff : Cutoff frequency (Hz) — single value or tuple for bandpass
            btype  : 'low', 'high', or 'bandpass'
            order  : Filter order (typical: 2-5)
        """

        # low = 0.5
        # high = 10

        # get the most frequent interval between consecutive timestamps
        unique, counts = np.unique(np.diff(time), return_counts = True)
        most_frequent = unique[np.argmax(counts)]
        # convert to frequency 
        fs = 1/(most_frequent*1e-3)
        print (f'Signal Frequency = {fs:.3f} Hz')
        print ('')

        b, a = butter(order, [0.528, 8.0], btype = 'bandpass', fs = fs)    # Charlton Paper.
        w, h = freqz(b, a, fs = fs)
        filtered_signal = filtfilt(b, a, signal)
        if plot:
            self.plot(time, signal, filtered_signal, f'Butterworth: order - {order} Filter', signal_type, start, end)

        return filtered_signal
    

    def cheby2_filter(self, time, signal, order = 4, rs = 40, signal_type = 'PPG', plot = True, start = 0, end = None):
        """
            signal   : 1D numpy array (PPG signal)
            fs     : Sampling frequency in Hz
            cutoff : Cutoff frequency (Hz)
            btype  : 'low', 'high', or 'bandpass'
            order  : Filter order (typical: 2-5)
            rs     : Transition band/ Min attenuation in the stopband (typical: 3-40)
        """

        # get the most frequent interval between consecutive timestamps
        unique, counts = np.unique(np.diff(time), return_counts = True)
        most_frequent = unique[np.argmax(counts)]
        # convert to frequency 
        fs = 1/(most_frequent*1e-3)
        print (f'Signal Frequency = {fs:.3f} Hz')
        print ('')

        sos = cheby2(order, rs, [0.528, 8.0], btype='bandpass', fs=fs, output='sos')
        filtered_signal = sosfiltfilt(sos, signal)
        if plot:
            self.plot(time, signal, filtered_signal, f'Chebyshev Type II: order - {order} Filter', signal_type, start, end)
        
        return filtered_signal
    
    def empirical_mode_decomposition(self, time, signal, signal_type = 'PPG', first = 1, last = 1, plot = True, start = 0, end = None):
        """
            Args:
                time: array of time
                signal: signal values (amplitudes)
                signal_type: 'ECG' or 'PPG'
                first: int (omits the first 'n' IMFs)
                last: int (omits the last 'n' IMFs)
                plot: bool (default = True)
                start: set x-lim start location in the plot
                end: set x-lim end location in the plot
        """
        emd = EEMD()
        imfs = emd(signal)
        n_imfs = imfs.shape[0]

        if plot:
            fig, axs = plt.subplots(n_imfs + 1, 1, figsize = (10, 2 * (n_imfs + 1)))
            axs[0].plot(time, signal)
            axs[0].set_title(f'Original {signal_type}')
            axs[0].set_xlim([start, end])

            for i in range(n_imfs):
                axs[i+1].plot(time, imfs[i])
                axs[i+1].set_title(f'IMF {i+1}')
            
            plt.tight_layout()
            plt.show()
        
        reconstructed_signal = np.sum(imfs[first:-last], axis = 0) # [1:-1] omits the first IMF and last IMF
        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex = True)

            # Plot the original signal on top
            axs[0].plot(time, signal, color='tab:blue')
            axs[0].set_title(f'Original {signal_type}')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_xlim([start, end])

            # Plot reconstructed signal on bottom
            axs[1].plot(time, reconstructed_signal, color='tab:green')
            axs[1].set_title(f'Filtered {signal_type}')
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Amplitude')
            axs[1].set_xlim([start, end])

            plt.tight_layout()

        return reconstructed_signal



class XQRS:
    """
        Peak detection and visualization utility for ECG and PPG signal using the XQRS algorithm.
        This class supports detecting peaks in physiological signals (ECG/PPG),
        refining detected peaks using a local greedy search, and visualizing the results.

        Methods:
            find_peaks: Detects and optionally refines peaks in a time series signal.
            plot_peaks: Plots ECG and PPG signals along with their detected peak locations.
    """
    
    def __init__(self):
        pass


    def find_peaks(self, signal_time, signal_value, search_window = 5, refinement = True):
        """
        Detects R-peaks or pulse peaks using the XQRS algorithm and refines their positions within a local window.

        Args:
            signal_time (np.ndarray): Timestamps of the signal in milliseconds.
            signal_value (np.ndarray): Corresponding amplitude values of the signal.
            search_window (int): Number of sample points to search on each side of the initially detected peak.
            refinement (bool): If True, applies local greedy refinement around each detected peak.

        Returns:
            peak_times (np.ndarray): Refined peak times corresponding to the signal_time axis.
            refined_peaks (np.ndarray): Indices in the signal_value corresponding to refined peak locations.
        """
        
        #fs = 129.99976196332648 if signal_type == 'ECG' else 50

        # get the most frequent interval between consecutive timestamps
        unique, counts = np.unique(np.diff(signal_time), return_counts = True)
        most_frequent = unique[np.argmax(counts)]
        # convert to frequency 
        fs = 1/(most_frequent*1e-3)
        print (f'Signal Frequency = {fs:.3f} Hz')
        print ('')

        rpeaks = wfdb.processing.xqrs_detect(signal_value, fs, verbose = True)
        peak_times = signal_time[rpeaks]

        if refinement == False:
            return peak_times, rpeaks
        else:
            # refine peaks via Local Greedy Refinement
            refined_peaks = []
            for rp in rpeaks:
                start = max(rp - search_window, 0)
                end = min(rp + search_window + 1, len(signal_value))   # +1 to include end index

                local_window = signal_value[start:end]
                if len(local_window) == 0:
                    refined_peaks.append(rp)
                    continue

                # Find local max in the window
                local_max_idx = np.argmax(local_window)
                refined_peak = start + local_max_idx
                refined_peaks.append(refined_peak)

            refined_peaks = np.array(refined_peaks)
            peak_times = signal_time[refined_peaks]

            # Remove duplicate elements (if any)
            unique_times, unique_indices = np.unique(peak_times, return_index=True)
            peak_values = refined_peaks
            unique_peaks = peak_values[unique_indices]

            return unique_times, unique_peaks

    

    def plot_peaks(self, time_ecg, ecg, time_ppg, ppg, peak_times_ecg, refined_peaks_ecg, peak_times_ppg, refined_peaks_ppg, start = 0, end = None):
        """
        Plots ECG and PPG signals alongside their detected peaks.

        Args:
            time_ecg (np.ndarray): Time axis for ECG signal.
            ecg (np.ndarray): ECG signal values.
            time_ppg (np.ndarray): Time axis for PPG signal.
            ppg (np.ndarray): PPG signal values.
            peak_times_ecg (np.ndarray): Timestamps of detected ECG peaks.
            refined_peaks_ecg (np.ndarray): Indices in ECG signal corresponding to peaks.
            peak_times_ppg (np.ndarray): Timestamps of detected PPG peaks.
            refined_peaks_ppg (np.ndarray): Indices in PPG signal corresponding to peaks.
            start (float): Starting value of time window to display.
            end (float): Ending value of time window to display. If None, shows full signal.
        """

        plt.figure(figsize=(20,4))
        plt.subplot(2,1,1)
        plt.plot(time_ecg, ecg, label='ECG', color='tab:blue')
        plt.scatter(peak_times_ecg, ecg[refined_peaks_ecg], marker = 'x', color = 'black')
        plt.xlim([start, end])

        plt.subplot(2,1,2)
        plt.plot(time_ppg, ppg, label='PPG', color='tab:red')
        plt.scatter(peak_times_ppg, ppg[refined_peaks_ppg], marker = 'x', color = 'black')
        plt.xlim([start, end])
        plt.show()