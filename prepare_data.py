import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from utils import LoadDataset, Filters
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.integrate import trapezoid
import pywt
import argparse

class DataPreparationPipeline:
    """
    A pipeline for preparing HRV peak classification data for XGBoost.
    
    This class loads PPG data, detects initial peaks, calculates features
    for each peak, and provides GUIs for manual correction and labeling.

    Additionally, this class loads ECG data, detects initial peaks. Does not apply
    any correction, as ECG data is relatively clean. Using the detected
    peaks it then calculates the SDNN (true label).
    """

    def __init__(self):
        self.initial_features = None
        self.initial_labels = None
        self.final_labels = None
        self.remove_regions = []
        
    def get_data(self, data_loc, plot=False):
        """
        Load raw ECG, PPG, and polar HRM data from a specified directory.

        Args:
            data_loc (str): Path to the directory containing the input CSV files.
            plot (bool): If True, displays the raw signals using a plotting utility.
        """
        self.data_loc = data_loc
        data_loader = LoadDataset(self.data_loc)
        self.time_ecg, self.ecg, self.time_ppg, self.ppg, self.time_polarHRM, self.polarHRM = data_loader.get_data()
        
        if plot:
            data_loader.plot_raw_data(start_time=0, end_time=None)

    def get_peaks(self, filter_type="Chebyshev"):
        """
        Automatically detect peaks in PPG signal using filtering and SciPy's find_peaks.

        Args:
            filter_type (str): Type of filter for processing PPG signal. (Butterworth or Chebyshev)
        """
        if filter_type.lower() == "butterworth":
            filter = Filters()
            print("\nUsing Butterworth Filter\n")
            self.filtered_ppg = filter.butter_filter(self.time_ppg, self.ppg, order=4, signal_type='PPG', plot=False)

        elif filter_type.lower() == "chebyshev":
            filter = Filters()
            print("\nUsing Chebyshev Type-II Filter\n")
            self.filtered_ppg = filter.cheby2_filter(self.time_ppg, self.ppg, order=4, rs=40, signal_type='PPG', plot=False)
        
        # Detect initial peaks
        self.peak_indices_initial, _ = find_peaks(self.filtered_ppg, height=np.mean(self.filtered_ppg))
        self.peak_times_initial = self.time_ppg[self.peak_indices_initial]
        
        print(f"Initially detected {len(self.peak_indices_initial)} peaks.")
        
        # Plot initial detection
        plt.figure(figsize=(15, 6))
        plt.plot(self.time_ppg, self.filtered_ppg, 'r-', lw=1, label='Filtered PPG')
        plt.scatter(self.peak_times_initial, self.filtered_ppg[self.peak_indices_initial], 
                   marker='x', color='black', s=50, label='Detected Peaks')
        plt.title('Initial Peak Detection')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_features_for_peaks(self, peak_idx, all_peak_indices, signal, time, is_new_peak=False):
        """
        Calculates 15 morphological features for a singular peak.
        Robust version with comprehensive error handling.

        Args:
            peak_idx (int): Index of the peak in the signal
            all_peak_indices (np.array): Array of all peak indices
            signal (np.array): The signal data
            time (np.array): Time array
            is_new_peak (bool): Whether this is a newly added peak (not from initial detection)

        Returns:
            list: 15 features for the peak
        """
        features = []

        try:
            # Validate inputs
            if peak_idx < 0 or peak_idx >= len(signal):
                print(f"Warning: peak_idx {peak_idx} out of bounds [0, {len(signal)-1}]")
                return [0] * 15
            
            if len(time) != len(signal):
                print(f"Warning: time and signal length mismatch: {len(time)} vs {len(signal)}")
                return [0] * 15

            # 1. Amplitude
            try:
                amplitude = signal[peak_idx]
                if not np.isfinite(amplitude):
                    amplitude = 0
            except:
                amplitude = 0
            features.append(amplitude)

            # 2. Prominence 
            if is_new_peak:
                # For new peaks, calculate prominence manually from local minima
                try:
                    window = min(50, len(signal) // 10, peak_idx, len(signal) - peak_idx - 1)
                    window = max(1, window)  # Ensure minimum window
                    
                    start_idx = max(0, peak_idx - window)
                    end_idx = min(len(signal), peak_idx + window + 1)
                    
                    peak_val = signal[peak_idx]
                    
                    # Robust prominence calculation
                    if start_idx < peak_idx:
                        left_segment = signal[start_idx:peak_idx+1]
                        left_min = np.min(left_segment) if len(left_segment) > 0 else peak_val
                    else:
                        left_min = peak_val
                    
                    if peak_idx < end_idx - 1:
                        right_segment = signal[peak_idx:end_idx]
                        right_min = np.min(right_segment) if len(right_segment) > 0 else peak_val
                    else:
                        right_min = peak_val
                    
                    prominence_val = peak_val - max(left_min, right_min)
                    prominence_val = max(0, prominence_val)  # Ensure non-negative
                    
                    if not np.isfinite(prominence_val):
                        prominence_val = 0
                except Exception as e:
                    print(f"Error calculating prominence for new peak at {peak_idx}: {e}")
                    prominence_val = 0
            else:
                # For existing peaks, use scipy function with fallback
                try:
                    # Filter valid peak indices
                    valid_peaks = all_peak_indices[
                        (all_peak_indices >= 0) & (all_peak_indices < len(signal))
                    ]
                    if len(valid_peaks) == 0:
                        prominence_val = 0
                    else:
                        prominences = peak_prominences(signal, valid_peaks)[0]
                        if peak_idx in valid_peaks:
                            prominence_idx = list(valid_peaks).index(peak_idx)
                            prominence_val = prominences[prominence_idx]
                            if not np.isfinite(prominence_val):
                                prominence_val = 0
                        else:
                            prominence_val = 0
                except Exception as e:
                    print(f"Error calculating prominence for existing peak at {peak_idx}: {e}")
                    prominence_val = 0
            features.append(prominence_val)

            # 3. Width at Half Prominence
            if is_new_peak:
                # For new peaks, calculate width manually
                try:
                    window = min(50, len(signal) // 10, peak_idx, len(signal) - peak_idx - 1)
                    window = max(1, window)
                    
                    start_idx = max(0, peak_idx - window)
                    end_idx = min(len(signal), peak_idx + window + 1)
                    
                    peak_val = signal[peak_idx]
                    half_height = peak_val - max(0, prominence_val / 2)
                    
                    # Find width at half height
                    left_idx = peak_idx
                    right_idx = peak_idx
                    
                    # Search left
                    for i in range(peak_idx, start_idx, -1):
                        if i < len(signal) and signal[i] <= half_height:
                            left_idx = i
                            break
                    
                    # Search right  
                    for i in range(peak_idx, end_idx):
                        if i < len(signal) and signal[i] <= half_height:
                            right_idx = i
                            break
                    
                    width_half_prom = abs(right_idx - left_idx)
                    if not np.isfinite(width_half_prom):
                        width_half_prom = 0
                except Exception as e:
                    print(f"Error calculating width for new peak at {peak_idx}: {e}")
                    width_half_prom = 0
            else:
                # For existing peaks, use scipy function with fallback
                try:
                    # Filter valid peak indices
                    valid_peaks = all_peak_indices[
                        (all_peak_indices >= 0) & (all_peak_indices < len(signal))
                    ]
                    if len(valid_peaks) == 0:
                        width_half_prom = 0
                    else:
                        widths = peak_widths(signal, valid_peaks, rel_height=0.50)[0]
                        if peak_idx in valid_peaks:
                            width_idx = list(valid_peaks).index(peak_idx)
                            width_half_prom = widths[width_idx]
                            if not np.isfinite(width_half_prom):
                                width_half_prom = 0
                        else:
                            width_half_prom = 0
                except Exception as e:
                    print(f"Error calculating width for existing peak at {peak_idx}: {e}")
                    width_half_prom = 0
            features.append(width_half_prom)

            # 4. Pulse Area (robust integration)
            try:
                window = min(20, len(signal) // 10, peak_idx, len(signal) - peak_idx - 1)
                window = max(1, window)
                
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window + 1)
                
                if start_idx < end_idx:
                    segment = signal[start_idx:end_idx]
                    if len(segment) > 0:
                        baseline = np.min(segment)
                        adjusted = segment - baseline
                        adjusted = np.maximum(adjusted, 0)  # Ensure non-negative
                        pulse_area = trapezoid(adjusted, dx=1)
                        if not np.isfinite(pulse_area):
                            pulse_area = 0
                    else:
                        pulse_area = 0
                else:
                    pulse_area = 0
            except Exception as e:
                print(f"Error calculating pulse area for peak at {peak_idx}: {e}")
                pulse_area = 0
            features.append(pulse_area)

            # 5-6. Rise and Decay Times (robust timing)
            try:
                window = min(20, len(signal) // 10, peak_idx, len(signal) - peak_idx - 1)
                window = max(1, window)
                
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window + 1)

                # Rise Time
                if start_idx < peak_idx:
                    before_segment = signal[start_idx:peak_idx]
                    if len(before_segment) > 0:
                        min_before_idx = np.argmin(before_segment) + start_idx
                        if min_before_idx < len(time) and peak_idx < len(time):
                            rise_time = abs(time[peak_idx] - time[min_before_idx])
                            if not np.isfinite(rise_time):
                                rise_time = 0
                        else:
                            rise_time = 0
                    else:
                        rise_time = 0
                else:
                    rise_time = 0

                # Decay Time
                if peak_idx < end_idx - 1:
                    after_segment = signal[peak_idx:end_idx]
                    if len(after_segment) > 0:
                        min_after_idx = np.argmin(after_segment) + peak_idx
                        if min_after_idx < len(time) and peak_idx < len(time):
                            decay_time = abs(time[min_after_idx] - time[peak_idx])
                            if not np.isfinite(decay_time):
                                decay_time = 0
                        else:
                            decay_time = 0
                    else:
                        decay_time = 0
                else:
                    decay_time = 0
            except Exception as e:
                print(f"Error calculating rise/decay times for peak at {peak_idx}: {e}")
                rise_time = decay_time = 0
            features.extend([rise_time, decay_time])

            # 7-8. Max upslope and max inflection (robust derivatives)
            try:
                window = min(10, len(signal) // 20, peak_idx, len(signal) - peak_idx - 1)
                window = max(1, window)
                
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window + 1)

                # Max Upslope
                if peak_idx > 0 and start_idx < peak_idx:
                    rising_segment = signal[start_idx:(peak_idx + 1)]
                    if len(rising_segment) > 1:
                        rising_deriv1 = np.diff(rising_segment, n=1)
                        if len(rising_deriv1) > 0:
                            max_upslope = np.max(rising_deriv1)
                            if not np.isfinite(max_upslope):
                                max_upslope = 0
                        else:
                            max_upslope = 0
                    else:
                        max_upslope = 0
                else:
                    max_upslope = 0
                
                # Max Inflection
                if start_idx < end_idx:
                    segment = signal[start_idx:end_idx]
                    if len(segment) > 2:
                        segment_deriv2 = np.diff(segment, n=2)
                        if len(segment_deriv2) > 0:
                            max_inflection = np.max(np.abs(segment_deriv2))
                            if not np.isfinite(max_inflection):
                                max_inflection = 0
                        else:
                            max_inflection = 0
                    else:
                        max_inflection = 0
                else:
                    max_inflection = 0
            except Exception as e:
                print(f"Error calculating slopes for peak at {peak_idx}: {e}")
                max_upslope = max_inflection = 0
            features.extend([max_upslope, max_inflection])

            # 9-11. Inter-Beat Intervals (robust IBI calculation)
            try:
                # Find position in sorted peak array
                valid_peaks = all_peak_indices[
                    (all_peak_indices >= 0) & (all_peak_indices < len(signal))
                ]
                valid_peaks = np.sort(valid_peaks)
                
                peak_position = np.where(valid_peaks == peak_idx)[0]
                if len(peak_position) > 0:
                    pos = peak_position[0]

                    # Previous IBI
                    if pos > 0 and (pos - 1) < len(valid_peaks):
                        prev_peak_idx = valid_peaks[pos - 1]
                        if prev_peak_idx < len(time) and peak_idx < len(time):
                            ibi_prev = abs(time[peak_idx] - time[prev_peak_idx])
                            if not np.isfinite(ibi_prev):
                                ibi_prev = 0
                        else:
                            ibi_prev = 0
                    else:
                        ibi_prev = 0

                    # Next IBI
                    if (pos + 1) < len(valid_peaks):
                        next_peak_idx = valid_peaks[pos + 1]
                        if next_peak_idx < len(time) and peak_idx < len(time):
                            ibi_next = abs(time[next_peak_idx] - time[peak_idx])
                            if not np.isfinite(ibi_next):
                                ibi_next = 0
                        else:
                            ibi_next = 0
                    else:
                        ibi_next = 0

                    # IBI Ratio (safe division)
                    if ibi_prev > 0 and ibi_next > 0:
                        ibi_ratio = ibi_next / ibi_prev
                        if not np.isfinite(ibi_ratio):
                            ibi_ratio = 0
                    else:
                        ibi_ratio = 0
                else:
                    ibi_prev = ibi_next = ibi_ratio = 0
            except Exception as e:
                print(f"Error calculating IBIs for peak at {peak_idx}: {e}")
                ibi_prev = ibi_next = ibi_ratio = 0
            features.extend([ibi_prev, ibi_next, ibi_ratio])

            # 12-13. Signal Quality Metrics (robust statistics)
            try:
                window = min(20, len(signal) // 10, peak_idx, len(signal) - peak_idx - 1)
                window = max(1, window)
                
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window + 1)
                
                if start_idx < end_idx:
                    segment = signal[start_idx:end_idx]
                    if len(segment) > 0:
                        # Local Variance
                        local_variance = np.var(segment)
                        if not np.isfinite(local_variance):
                            local_variance = 0
                        
                        # SNR (safe calculation)
                        signal_power = np.mean(segment ** 2)
                        if signal_power > 0 and local_variance > 0:
                            noise_power = local_variance ** 2
                            if noise_power > 0:
                                snr = 10 * np.log10(signal_power / noise_power)
                                if not np.isfinite(snr):
                                    snr = 0
                            else:
                                snr = 0
                        else:
                            snr = 0
                    else:
                        local_variance = snr = 0
                else:
                    local_variance = snr = 0
            except Exception as e:
                print(f"Error calculating signal quality for peak at {peak_idx}: {e}")
                local_variance = snr = 0
            features.extend([local_variance, snr])

            # 14-15. Frequency Features (robust FFT and wavelet)
            try:
                window = min(50, len(signal) // 5, peak_idx, len(signal) - peak_idx - 1)
                window = max(4, window)  # Minimum for FFT
                
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window + 1)
                
                if start_idx < end_idx:
                    segment = signal[start_idx:end_idx]
                    if len(segment) >= 4:
                        # Frequency Energy using FFT
                        try:
                            fft = np.fft.fft(segment)
                            freq_energy = np.sum(np.abs(fft) ** 2)
                            if not np.isfinite(freq_energy):
                                freq_energy = 0
                        except:
                            freq_energy = 0

                        # Wavelet Coefficient
                        try:
                            coeffs = pywt.wavedec(segment, 'db4', level=1)
                            if len(coeffs) > 1 and len(coeffs[1]) > 0:
                                detail_coef = coeffs[1]
                                wavelet_coef = np.mean(np.abs(detail_coef))
                                if not np.isfinite(wavelet_coef):
                                    wavelet_coef = 0
                            else:
                                wavelet_coef = 0
                        except:
                            wavelet_coef = 0
                    else:
                        freq_energy = wavelet_coef = 0
                else:
                    freq_energy = wavelet_coef = 0
            except Exception as e:
                print(f"Error calculating frequency features for peak at {peak_idx}: {e}")
                freq_energy = wavelet_coef = 0
            features.extend([freq_energy, wavelet_coef])

            # Final validation - ensure all features are finite numbers
            for i, feature in enumerate(features):
                if not np.isfinite(feature):
                    features[i] = 0
                    
            if len(features) != 15:
                print(f"Warning: Expected 15 features, got {len(features)}. Padding with zeros.")
                features = (features + [0] * 15)[:15]

        except Exception as e:
            print(f"Critical error calculating features for peak at index {peak_idx}: {e}")
            features = [0] * 15

        return features

    def calculate_all_initial_features(self):
        """Calculate features for all initially deteced peaks before any manual correction."""
        print("Calculating features for all initially detected peaks...")

        self.initial_features = []
        self.initial_labels = np.ones(len(self.peak_indices_initial))

        for i, peak_idx in enumerate(self.peak_indices_initial):
            features = self.calculate_features_for_peaks(peak_idx=peak_idx, 
                                                        all_peak_indices=self.peak_indices_initial,
                                                        signal=self.filtered_ppg,
                                                        time=self.time_ppg,
                                                        is_new_peak=False)
            self.initial_features.append(features)

        self.initial_features = np.array(self.initial_features)
        print(f"Calculated features for {len(self.initial_features)} peaks")
        print(f"Feature matrix shape: {self.initial_features.shape}")

    def remove_unusable_signal(self):
        """
        GUI tool to remove unusable parts in the PPG signal.
        Peaks in removed regions will be labeled as false (0).
        """
        fig, ax = plt.subplots(figsize=(25, 8))

        # Plot PPG with initial peaks
        line_ppg, = ax.plot(self.time_ppg, self.filtered_ppg, color='red', lw=1, label='Filtered PPG')
        markers_ppg = ax.scatter(self.peak_times_initial, self.filtered_ppg[self.peak_indices_initial], 
                                marker='x', color='black', s=50, label='Initial Peaks')
        ax.set_title('PPG Signal - Select Unusable Regions')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(which='major', linestyle='--', linewidth=0.7)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth=0.4)

        self.remove_regions = []  # list to store (xmin, xmax) to remove

        def onselect(xmin, xmax):
            xmin, xmax = xmin.item(), xmax.item()

            if xmin < 0:
                xmin = 0

            self.remove_regions.append((xmin, xmax))
            # Shade selected region
            ax.axvspan(xmin, xmax, color='red', alpha=0.3)
            fig.canvas.draw()

        span = SpanSelector(
            ax,
            onselect,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='red')
        )

        print("Select unusable regions on the PPG plot using mouse. Close the window when done.")
        plt.show()

        # Update labels for peaks in removed regions
        for xmin, xmax in self.remove_regions:
            for i, peak_time in enumerate(self.peak_times_initial):
                if xmin <= peak_time <= xmax:
                    self.initial_labels[i] = 0  # Mark as false peak
                    
        print(f"Removed {len(self.remove_regions)} region(s) from signals.")
        print(f"Peaks labeled as false due to region removal: {np.sum(self.initial_labels == 0)}")

    def correct_peaks(self):
        """
        Launch an interactive matplotlib GUI for manual correction of detected PPG peaks.
        Two-phase process: First remove invalid peaks, then add new peaks.
        ALL initially detected peaks remain in final CSV - only their labels change (1->0 for removed peaks).
        Only newly added peaks are actually added to the dataset.
        """
        # Start with ALL initially detected peaks and their labels
        # Peaks in removed regions already have their labels set to 0 from remove_unusable_signal()
        working_labels = self.initial_labels.copy()
        
        print(f"Starting with {len(self.peak_indices_initial)} total initially detected peaks")
        print(f"Peaks already labeled as false due to region removal: {np.sum(working_labels == 0)}")
        
        # PHASE 1: PEAK REMOVAL (Change labels from 1 to 0, don't actually remove from arrays)
        print("\n=== PHASE 1: PEAK REMOVAL ===")
        print("Right-click on peaks to mark them as false (label=0). Close window when done with removals.")
        
        # Filter display signal (remove unusable regions for display only)
        valid_mask = np.ones_like(self.time_ppg, dtype=bool)
        for xmin, xmax in self.remove_regions:
            valid_mask &= ~((self.time_ppg >= xmin) & (self.time_ppg <= xmax))
        
        time_display = self.time_ppg[valid_mask]
        signal_display = self.filtered_ppg[valid_mask]
        
        # Find display indices for peaks that are NOT in removed regions (for display only)
        display_peak_indices = []
        display_peak_labels = []
        original_peak_map = []  # Map display index to original peak index
        
        for i, (peak_idx, peak_time) in enumerate(zip(self.peak_indices_initial, self.peak_times_initial)):
            # Check if peak is in a removed region
            in_removed_region = False
            for xmin, xmax in self.remove_regions:
                if xmin <= peak_time <= xmax:
                    in_removed_region = True
                    break
            
            if not in_removed_region:
                # Find corresponding index in filtered signal for display
                closest_idx = np.argmin(np.abs(time_display - peak_time))
                display_peak_indices.append(closest_idx)
                display_peak_labels.append(working_labels[i])
                original_peak_map.append(i)
        
        display_peak_indices = np.array(display_peak_indices)
        display_peak_labels = np.array(display_peak_labels)
        original_peak_map = np.array(original_peak_map)
        
        press_event = {'x': None, 'y': None}
        click_threshold = 5  # pixels

        fig, ax = plt.subplots(figsize=(25, 8))
        line, = ax.plot(time_display, signal_display, color='red', lw=1, label='Filtered PPG')
        
        def update_removal_plot():
            # Show peaks with different colors based on labels
            true_peaks = display_peak_indices[display_peak_labels == 1]
            false_peaks = display_peak_indices[display_peak_labels == 0]
            
            # Clear previous scatter plots
            for collection in ax.collections[:]:
                collection.remove()
            
            # Plot true peaks in blue
            if len(true_peaks) > 0:
                ax.scatter(time_display[true_peaks], signal_display[true_peaks], 
                          c='blue', marker='x', s=50, label='True Peaks')
            
            # Plot false peaks in red
            if len(false_peaks) > 0:
                ax.scatter(time_display[false_peaks], signal_display[false_peaks], 
                          c='red', marker='x', s=50, label='False Peaks')
            
            ax.legend()
            fig.canvas.draw_idle()
        
        # Initial plot
        update_removal_plot()
        
        ax.set_title("PHASE 1: Right Click to Mark Peaks as False\nBlue=True, Red=False\nClose window when done")
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(which='major', linestyle='--', linewidth=0.7)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth=0.4)

        def on_press_removal(event):
            if event.inaxes == ax:
                press_event['x'], press_event['y'] = event.x, event.y

        def on_release_removal(event):
            nonlocal display_peak_labels, working_labels
            
            if event.inaxes != ax or event.xdata is None:
                return

            dx = abs(event.x - press_event['x'])
            dy = abs(event.y - press_event['y'])
            if dx > click_threshold or dy > click_threshold:
                return

            if event.button == 3:  # right click → mark peak as false
                click_x = event.xdata
                
                if len(display_peak_indices) > 0:
                    nearest_display_idx = np.argmin(np.abs(time_display[display_peak_indices] - click_x))
                    original_idx = original_peak_map[nearest_display_idx]
                    
                    # Change label to 0 in both display and working arrays
                    display_peak_labels[nearest_display_idx] = 0
                    working_labels[original_idx] = 0
                    
                    peak_time = self.peak_times_initial[original_idx]
                    print(f"Marked peak at time {peak_time:.2f} as false (label=0)")
                    update_removal_plot()
        
        fig.canvas.mpl_connect('button_press_event', on_press_removal)
        fig.canvas.mpl_connect('button_release_event', on_release_removal)
        plt.show()
        
        print(f"After manual removals:")
        print(f"True peaks (label=1): {np.sum(working_labels == 1)}")
        print(f"False peaks (label=0): {np.sum(working_labels == 0)}")
        
        # PHASE 2: PEAK ADDITION (Actually add new peaks to the dataset)
        print("\n=== PHASE 2: PEAK ADDITION ===")
        print("Left-click to add new peaks. Close window when done with additions.")
        
        # Lists to store new peaks (will be inserted in chronological order later)
        new_peak_indices = []
        new_peak_times = []
        
        fig2, ax2 = plt.subplots(figsize=(25, 8))
        line2, = ax2.plot(time_display, signal_display, color='red', lw=1, label='Filtered PPG')
        
        def update_addition_plot():
            # Clear previous scatter plots
            for collection in ax2.collections[:]:
                collection.remove()
            
            # Show existing true peaks in blue
            true_peaks = display_peak_indices[display_peak_labels == 1]
            if len(true_peaks) > 0:
                ax2.scatter(time_display[true_peaks], signal_display[true_peaks], 
                           c='blue', marker='x', s=50, label='Existing True Peaks')
            
            # Show existing false peaks in red  
            false_peaks = display_peak_indices[display_peak_labels == 0]
            if len(false_peaks) > 0:
                ax2.scatter(time_display[false_peaks], signal_display[false_peaks], 
                           c='red', marker='x', s=50, label='Existing False Peaks')
            
            # Show new peaks in green
            if len(new_peak_times) > 0:
                new_display_indices = []
                for new_time in new_peak_times:
                    closest_idx = np.argmin(np.abs(time_display - new_time))
                    new_display_indices.append(closest_idx)
                ax2.scatter(time_display[new_display_indices], signal_display[new_display_indices], 
                           c='green', marker='o', s=50, label='New Peaks')
            
            ax2.legend()
            fig2.canvas.draw_idle()
        
        # Initial plot
        update_addition_plot()
        
        ax2.set_title("PHASE 2: Left Click to Add New Peaks\nBlue=Existing True, Red=Existing False, Green=New")
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(which='major', linestyle='--', linewidth=0.7)
        ax2.minorticks_on()
        ax2.grid(which='minor', linestyle=':', linewidth=0.4)

        def on_press_addition(event):
            if event.inaxes == ax2:
                press_event['x'], press_event['y'] = event.x, event.y

        def on_release_addition(event):
            if event.inaxes != ax2 or event.xdata is None:
                return

            dx = abs(event.x - press_event['x'])
            dy = abs(event.y - press_event['y'])
            if dx > click_threshold or dy > click_threshold:
                return

            if event.button == 1:  # left click → add peak
                click_x = event.xdata
                
                # Find closest point in display signal
                display_idx = np.argmin(np.abs(time_display - click_x))
                click_time = time_display[display_idx]
                
                # Find corresponding index in original signal
                orig_idx = np.argmin(np.abs(self.time_ppg - click_time))
                
                # PEAK REFINEMENT: Find nearest local maximum within window
                refinement_window = min(15, len(self.filtered_ppg) // 30)  # Reduced from 25 to 15 samples
                start_refine = max(0, orig_idx - refinement_window)
                end_refine = min(len(self.filtered_ppg), orig_idx + refinement_window)
                
                # Ensure valid window
                if start_refine >= end_refine or end_refine - start_refine < 3:
                    print(f"Warning: Invalid refinement window at index {orig_idx}")
                    final_idx = orig_idx
                    final_time = click_time
                else:
                    # Find local maximum in the window
                    segment = self.filtered_ppg[start_refine:end_refine]
                    if len(segment) > 0:
                        local_max_offset = np.argmax(segment)
                        refined_idx = start_refine + local_max_offset
                        
                        # Boundary safety check
                        if refined_idx >= len(self.time_ppg) or refined_idx >= len(self.filtered_ppg):
                            print(f"Warning: Refined index {refined_idx} out of bounds")
                            refined_idx = min(refined_idx, len(self.time_ppg) - 1, len(self.filtered_ppg) - 1)
                        
                        refined_time = self.time_ppg[refined_idx]
                        
                        # Check if refinement moved us too close to an existing peak
                        min_dist_to_existing = float('inf')
                        if len(self.peak_times_initial) > 0:
                            min_dist_to_existing = np.min(np.abs(self.peak_times_initial - refined_time))
                        
                        # If refinement moved us very close to an existing peak, use original click position instead
                        if min_dist_to_existing < 50:  # If refined position is too close to existing peak
                            # Check if original click was far enough from existing peaks
                            min_dist_original = float('inf')
                            if len(self.peak_times_initial) > 0:
                                min_dist_original = np.min(np.abs(self.peak_times_initial - click_time))
                            
                            if min_dist_original >= 75:  # Original click was far enough
                                print(f"Refinement moved peak too close to existing peak. Using original position.")
                                final_idx = orig_idx
                                final_time = click_time
                                # Skip further refinement validation
                            else:
                                # Both original and refined are too close
                                final_idx = refined_idx
                                final_time = refined_time
                        else:
                            # Refinement is good, proceed with validation
                            final_idx = refined_idx
                            final_time = refined_time
                            
                            # Verify this is actually a reasonable peak
                            peak_amplitude = self.filtered_ppg[refined_idx]
                            
                            # Check if it's a reasonable peak (positive amplitude and above mean)
                            try:
                                signal_mean = np.mean(self.filtered_ppg)
                                signal_std = np.std(self.filtered_ppg)
                                
                                # More sophisticated peak validation
                                if peak_amplitude < (signal_mean - 0.5 * signal_std):
                                    print(f"Warning: Clicked location at time {click_time:.2f} is in a trough")
                                    print(f"Refined to time {refined_time:.2f} with amplitude {peak_amplitude:.2f}")
                                    print(f"Signal mean: {signal_mean:.2f}, std: {signal_std:.2f}")
                                    # Still allow but warn user
                                
                                # Check if the refined point is actually a local maximum
                                local_check_window = min(5, refined_idx, len(self.filtered_ppg) - refined_idx - 1)
                                if local_check_window > 0:
                                    check_start = max(0, refined_idx - local_check_window)
                                    check_end = min(len(self.filtered_ppg), refined_idx + local_check_window + 1)
                                    local_segment = self.filtered_ppg[check_start:check_end]
                                    
                                    if len(local_segment) > 0:
                                        local_max_idx = np.argmax(local_segment)
                                        if local_max_idx != (refined_idx - check_start):
                                            print(f"Warning: Refined point is not a local maximum")
                                
                            except Exception as e:
                                print(f"Error in peak validation: {e}")
                        
                        # Only print refinement info if there was significant movement
                        if abs(final_time - click_time) > 10.0:  # More than 10ms difference
                            print(f"Refined peak position from {click_time:.2f} to {final_time:.2f}")
                    else:
                        # Fallback to original if refinement fails
                        print(f"Warning: Empty segment during refinement")
                        final_idx = orig_idx
                        final_time = click_time
                
                # Check if peak already exists nearby (among existing OR new peaks)
                min_dist_existing = float('inf')
                if len(self.peak_times_initial) > 0:
                    min_dist_existing = np.min(np.abs(self.peak_times_initial - final_time))
                
                min_dist_new = float('inf')
                if len(new_peak_times) > 0:
                    min_dist_new = np.min(np.abs(np.array(new_peak_times) - final_time))
                
                min_dist = min(min_dist_existing, min_dist_new)
                if min_dist < 50:  # ms threshold (reduced to be less restrictive)
                    print(f"Peak already exists nearby at time {final_time:.2f} (distance: {min_dist:.1f}ms)")
                    return
                
                # Add refined peak
                new_peak_indices.append(final_idx)
                new_peak_times.append(final_time)
                
                print(f"Added new peak at time {final_time:.2f}")
                update_addition_plot()
        
        fig2.canvas.mpl_connect('button_press_event', on_press_addition)
        fig2.canvas.mpl_connect('button_release_event', on_release_addition)
        plt.show()
        
        print(f"Added {len(new_peak_indices)} new peaks")
        
        # PHASE 3: MERGE AND CALCULATE FEATURES FOR NEW PEAKS
        print("\n=== PHASE 3: MERGING AND FEATURE CALCULATION ===")
        
        if len(new_peak_indices) > 0:
            # Convert to numpy arrays
            new_peak_indices = np.array(new_peak_indices)
            new_peak_times = np.array(new_peak_times)
            
            # Combine all peaks and sort by time (chronological order)
            all_final_indices = np.concatenate([self.peak_indices_initial, new_peak_indices])
            all_final_times = np.concatenate([self.peak_times_initial, new_peak_times])
            all_final_labels = np.concatenate([working_labels, np.ones(len(new_peak_indices))])
            
            # Get chronological sort order
            sort_order = np.argsort(all_final_times)
            all_final_indices_sorted = all_final_indices[sort_order]
            all_final_times_sorted = all_final_times[sort_order]
            all_final_labels_sorted = all_final_labels[sort_order]
            
            # Create feature matrix for final dataset
            final_features_list = []
            
            # Track which peaks are new vs existing
            is_new_peak = np.concatenate([np.zeros(len(self.peak_indices_initial), dtype=bool), 
                                        np.ones(len(new_peak_indices), dtype=bool)])[sort_order]
            
            # Calculate features only for new peaks, keep existing features unchanged
            for i, (peak_idx, is_new) in enumerate(zip(all_final_indices_sorted, is_new_peak)):
                if is_new:
                    # Calculate all features for new peak using final peak arrangement
                    features = self.calculate_features_for_peaks(
                        peak_idx, all_final_indices_sorted, 
                        self.filtered_ppg, self.time_ppg, is_new_peak=True
                    )
                    final_features_list.append(features)
                    print(f"Calculated features for new peak at time {self.time_ppg[peak_idx]:.2f}")
                    
                else:
                    # For existing peaks: keep original features completely unchanged
                    original_peak_time = self.time_ppg[peak_idx]
                    feature_idx = np.argmin(np.abs(self.peak_times_initial - original_peak_time))
                    final_features_list.append(self.initial_features[feature_idx])
            
            # Store final results
            self.final_peak_indices = all_final_indices_sorted
            self.final_peak_times = all_final_times_sorted
            self.final_features = np.array(final_features_list)
            self.final_labels = all_final_labels_sorted
            
        else:
            # No new peaks added, just use the existing ones with updated labels
            self.final_peak_indices = self.peak_indices_initial
            self.final_peak_times = self.peak_times_initial
            self.final_features = self.initial_features
            self.final_labels = working_labels
        
        print(f"\nFinal dataset:")
        print(f"Total peaks: {len(self.final_peak_indices)}")
        print(f"True peaks (label=1): {np.sum(self.final_labels == 1)}")
        print(f"False peaks (label=0): {np.sum(self.final_labels == 0)}")

    def save_data(self):
        """Save the feature matrix and labels to CSV file."""
        folder_name = os.path.basename(self.data_loc.rstrip("/"))
        
        # Create simple model-data/participant_datasets structure
        model_data_dir = "model-data"
        participant_dir = os.path.join(model_data_dir, "participant_datasets")
        
        # Create directories if they don't exist
        for directory in [model_data_dir, participant_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
        # Create feature names
        feature_names = [
            'amplitude', 'prominence', 'width_half_prom', 'pulse_area',
            'rise_time', 'decay_time', 'max_upslope', 'max_inflection',
            'ibi_prev', 'ibi_next', 'ibi_ratio', 'local_variance', 'snr',
            'freq_energy', 'wavelet_coef', 'label'
        ]
        
        # Use the final processed data (includes both original peaks and newly added peaks)
        # Data is already in chronological order from correct_peaks method
        final_features = getattr(self, 'final_features', self.initial_features)
        final_labels = getattr(self, 'final_labels', self.initial_labels)
        final_peak_indices = getattr(self, 'final_peak_indices', self.peak_indices_initial)
        final_peak_times = getattr(self, 'final_peak_times', self.peak_times_initial)
        
        # Combine features and labels
        data_matrix = np.column_stack([final_features, final_labels])
        
        # Create DataFrame
        df = pd.DataFrame(data_matrix, columns=feature_names)
        
        # Add metadata columns (not for model training)
        df.insert(0, 'peak_idx', final_peak_indices)
        df.insert(1, 'peak_time', final_peak_times)
        
        # Save to CSV in participant_datasets subfolder
        save_path = os.path.join(participant_dir, f'{folder_name}_features.csv')
        df.to_csv(save_path, index=False)
        
        print(f"Saved feature data to {save_path}")
        print(f"Data shape: {df.shape}")
        print(f"Features saved: {len(feature_names)}")
        
        # Print summary
        print(f"\nDataset Summary:")
        print(f"Total final peaks: {len(final_peak_indices)}")
        print(f"True peaks (label=1): {np.sum(final_labels == 1)}")
        print(f"False peaks (label=0): {np.sum(final_labels == 0)}")
        print(f"Peaks are stored in chronological order")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HRV peak classification data for XGBoost")
    parser.add_argument("--data_loc", type=str, required=True, help="Path to data folder")
    parser.add_argument("--plot", action='store_true', help="Plot raw data")
    parser.add_argument("--filter_type", type=str, default='Chebyshev', help="PPG filter type (Butterworth or Chebyshev)")
    
    args = parser.parse_args()

    # initialize pipeline
    pipeline = DataPreparationPipeline()
    
    # load data
    pipeline.get_data(args.data_loc, plot=args.plot)
    
    # detect initial peaks and calculate features
    pipeline.get_peaks(args.filter_type)
    pipeline.calculate_all_initial_features()
    
    # corrections
    pipeline.remove_unusable_signal()
    pipeline.correct_peaks()
    
    # save results
    pipeline.save_data()
    
    print("\nData preparation complete!") 