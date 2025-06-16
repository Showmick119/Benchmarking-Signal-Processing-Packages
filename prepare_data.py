import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from utils import LoadDataset, Filters
from scipy.signal import find_peaks, peak_widths, peak_prominences
import pywt
import argparse


class DataPreparationPipeline:
    """
    A pipeline for preparing HRV peak classification data for XGBoost.
    
    This class loads PPG data, detects initial peaks, calculates features
    for each peak, and provides GUIs for manual correction and labeling.
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

    def get_peaks(self, filter_type='Chebyshev'):
        """
        Automatically detect peaks in PPG signal using filtering and find_peaks.

        Args:
            filter_type (str): Type of filter for processing PPG signal. (Butterworth or Chebyshev)
        """
        if filter_type.lower() == 'butterworth':
            filter = Filters()
            print('\nUsing Butterworth Type-II Filter')
            print('')
            self.filtered_ppg = filter.butter_filter(self.time_ppg, self.ppg, order=4, signal_type='PPG', plot=False)
            
        elif filter_type.lower() == 'chebyshev':
            filter = Filters()
            print('\nUsing Chebyshev Type-II Filter')
            print('')
            self.filtered_ppg = filter.cheby2_filter(self.time_ppg, self.ppg, order=4, rs=40, signal_type='PPG', plot=False)
        
        # Detect initial peaks
        self.peak_indices_initial, _ = find_peaks(self.filtered_ppg, height=np.mean(self.filtered_ppg))
        self.peak_times_initial = self.time_ppg[self.peak_indices_initial]
        
        print(f"Initially detected {len(self.peak_indices_initial)} peaks")
        
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

    def calculate_features_for_peak(self, peak_idx, all_peak_indices, signal, time, prominence=None):
        """
        Calculate 15 features for a single peak.
        
        Args:
            peak_idx (int): Index of the peak in the signal
            all_peak_indices (np.array): Array of all peak indices 
            signal (np.array): The signal data
            time (np.array): Time array
            prominence (float): Pre-calculated prominence value (optional)
            
        Returns:
            list: 15 features for the peak
        """
        features = []
        
        try:
            # 1. Amplitude
            amplitude = signal[peak_idx]
            features.append(amplitude)
            
            # 2. Prominence (use pre-calculated value if provided, otherwise calculate individually)
            if prominence is not None:
                features.append(prominence)
            else:
                try:
                    prominences, _ = peak_prominences(signal, [peak_idx])
                    prominence_val = prominences[0] if len(prominences) > 0 else 0
                except:
                    prominence_val = 0
                features.append(prominence_val)
            
            # 3. Width at half prominence
            try:
                widths, _, _, _ = peak_widths(signal, [peak_idx], rel_height=0.5)
                width_half_prom = widths[0] if len(widths) > 0 else 0
            except:
                width_half_prom = 0
            features.append(width_half_prom)
            
            # 4. Pulse area (approximate using width and amplitude)
            'Can extend this to use integration later on, for more accuracy'
            try:
                pulse_area = amplitude * width_half_prom
            except:
                pulse_area = 0
            features.append(pulse_area)
            
            # 5-6. Rise and decay times
            try:
                window = min(20, len(signal) // 10)  # Adaptive window
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window)
                
                # Rise time: from local min before to peak
                before_segment = signal[start_idx:peak_idx]
                if len(before_segment) > 1:
                    min_before_idx = np.argmin(before_segment) + start_idx
                    rise_time = time[peak_idx] - time[min_before_idx]
                else:
                    rise_time = 0
                    
                # Decay time: from peak to local min after
                after_segment = signal[peak_idx:end_idx]
                if len(after_segment) > 1:
                    min_after_idx = np.argmin(after_segment) + peak_idx
                    decay_time = time[min_after_idx] - time[peak_idx]
                else:
                    decay_time = 0
            except:
                rise_time = 0
                decay_time = 0
            
            features.extend([rise_time, decay_time])
            
            # 7-8. Max upslope and max inflection
            try:
                window = min(10, len(signal) // 20)
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window)
                
                # Max upslope
                if peak_idx > 0:
                    slopes = np.diff(signal[start_idx:peak_idx+1])
                    max_upslope = np.max(slopes) if len(slopes) > 0 else 0
                else:
                    max_upslope = 0
                    
                # Max inflection (second derivative)
                segment = signal[start_idx:end_idx]
                if len(segment) > 2:
                    second_deriv = np.diff(segment, n=2)
                    max_inflection = np.max(np.abs(second_deriv)) if len(second_deriv) > 0 else 0
                else:
                    max_inflection = 0
            except:
                max_upslope = 0
                max_inflection = 0
                
            features.extend([max_upslope, max_inflection])
            
            # 9-11. Inter-beat intervals
            try:
                peak_position = np.where(all_peak_indices == peak_idx)[0]
                if len(peak_position) > 0:
                    pos = peak_position[0]
                    
                    # Previous IBI
                    if pos > 0:
                        ibi_prev = time[peak_idx] - time[all_peak_indices[pos-1]]
                    else:
                        ibi_prev = 0
                        
                    # Next IBI  
                    if pos < len(all_peak_indices) - 1:
                        ibi_next = time[all_peak_indices[pos+1]] - time[peak_idx]
                    else:
                        ibi_next = 0
                        
                    # IBI ratio
                    if ibi_prev > 0 and ibi_next > 0:
                        ibi_ratio = ibi_next / ibi_prev
                    else:
                        ibi_ratio = 0
                else:
                    ibi_prev = ibi_next = ibi_ratio = 0
            except:
                ibi_prev = ibi_next = ibi_ratio = 0
                
            features.extend([ibi_prev, ibi_next, ibi_ratio])
            
            # 12-13. Signal quality metrics
            try:
                window = min(20, len(signal) // 10)
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window)
                segment = signal[start_idx:end_idx]
                
                # Local variance
                local_variance = np.var(segment) if len(segment) > 1 else 0
                
                # SNR (simplified)
                signal_power = amplitude ** 2
                noise_power = local_variance
                snr = signal_power / noise_power if noise_power > 0 else 0
            except:
                local_variance = 0
                snr = 0
                
            features.extend([local_variance, snr])
            
            # 14-15. Frequency features
            try:
                window = min(50, len(signal) // 5)
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(signal), peak_idx + window)
                segment = signal[start_idx:end_idx]
                
                if len(segment) > 4:
                    # Frequency energy using FFT
                    fft = np.fft.fft(segment)
                    freq_energy = np.sum(np.abs(fft)**2)
                    
                    # Wavelet coefficient
                    try:
                        coeffs = pywt.dwt(segment, 'db4')
                        wavelet_coef = np.mean(np.abs(coeffs[1])) if len(coeffs[1]) > 0 else 0
                    except:
                        wavelet_coef = 0
                else:
                    freq_energy = 0
                    wavelet_coef = 0
            except:
                freq_energy = 0
                wavelet_coef = 0
                
            features.extend([freq_energy, wavelet_coef])
            
        except Exception as e:
            print(f"Error calculating features for peak at index {peak_idx}: {e}")
            # Return zeros for all features if calculation fails
            features = [0] * 15
            
        return features

    def calculate_all_initial_features(self):
        """Calculate features for all initially detected peaks before any manual correction."""
        print("Calculating features for all initially detected peaks...")
        
        # Pre-calculate prominences for all peaks at once (this is key for proper prominence calculation)
        try:
            self.all_prominences, _ = peak_prominences(self.filtered_ppg, self.peak_indices_initial)
            print(f"Successfully calculated prominences for all peaks")
        except Exception as e:
            print(f"Error calculating prominences: {e}")
            self.all_prominences = np.zeros(len(self.peak_indices_initial))
        
        self.initial_features = []
        self.initial_labels = np.ones(len(self.peak_indices_initial))  # Initially all peaks are considered true
        
        for i, peak_idx in enumerate(self.peak_indices_initial):
            features = self.calculate_features_for_peak(peak_idx, self.peak_indices_initial, 
                                                      self.filtered_ppg, self.time_ppg, 
                                                      prominence=self.all_prominences[i])
            self.initial_features.append(features)
            
        self.initial_features = np.array(self.initial_features)
        print(f"Calculated features for {len(self.initial_features)} peaks")
        print(f"Feature matrix shape: {self.initial_features.shape}")
        print(f"Prominence range: {np.min(self.all_prominences):.3f} to {np.max(self.all_prominences):.3f}")

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
        This updates the labels and can add new peaks with their features.
        """
        # Create working copies
        self.final_labels = self.initial_labels.copy()
        current_peak_indices = self.peak_indices_initial.copy()
        current_features = self.initial_features.copy()
        
        # Filter out peaks in removed regions for display
        valid_mask = np.ones_like(self.time_ppg, dtype=bool)
        for xmin, xmax in self.remove_regions:
            valid_mask &= ~((self.time_ppg >= xmin) & (self.time_ppg <= xmax))
        
        # Apply mask to signals for display
        time_display = self.time_ppg[valid_mask]
        signal_display = self.filtered_ppg[valid_mask]
        
        # Get peaks that are still visible (not in removed regions)
        visible_peak_mask = []
        visible_peak_indices_display = []
        
        for i, peak_idx in enumerate(current_peak_indices):
            peak_time = self.time_ppg[peak_idx]
            in_removed_region = False
            for xmin, xmax in self.remove_regions:
                if xmin <= peak_time <= xmax:
                    in_removed_region = True
                    break
            
            if not in_removed_region:
                visible_peak_mask.append(i)
                # Find corresponding index in filtered signal
                closest_idx = np.argmin(np.abs(time_display - peak_time))
                visible_peak_indices_display.append(closest_idx)
        
        visible_peak_mask = np.array(visible_peak_mask)
        visible_peak_indices_display = np.array(visible_peak_indices_display)
        
        press_event = {'x': None, 'y': None}
        click_threshold = 5  # pixels

        fig, ax = plt.subplots(figsize=(25, 8))

        # Plot filtered signal without removed regions
        line, = ax.plot(time_display, signal_display, color='red', lw=1, label='Filtered PPG')
        
        # Plot visible peaks
        if len(visible_peak_indices_display) > 0:
            scatter = ax.scatter(time_display[visible_peak_indices_display], 
                               signal_display[visible_peak_indices_display], 
                               c='b', marker='x', s=50, label='Peaks')
        else:
            scatter = ax.scatter([], [], c='b', marker='x', s=50, label='Peaks')
            
        ax.set_title("Left Click: Add peak, Right Click: Remove peak\nClose window when done")
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(which='major', linestyle='--', linewidth=0.7)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth=0.4)

        def update_plot():
            if len(visible_peak_indices_display) > 0:
                scatter.set_offsets(np.c_[time_display[visible_peak_indices_display], 
                                        signal_display[visible_peak_indices_display]])
            else:
                scatter.set_offsets(np.empty((0, 2)))
            fig.canvas.draw_idle()
        
        def on_press(event):
            if event.inaxes == ax:
                press_event['x'], press_event['y'] = event.x, event.y

        def on_release(event):
            nonlocal visible_peak_indices_display, current_peak_indices, current_features
            
            if event.inaxes != ax or event.xdata is None:
                return

            dx = abs(event.x - press_event['x'])
            dy = abs(event.y - press_event['y'])
            if dx > click_threshold or dy > click_threshold:
                return  # it was a drag, not a click

            click_x = event.xdata

            if event.button == 1:  # left click → add peak
                # Find closest point in display signal
                display_idx = np.argmin(np.abs(time_display - click_x))
                
                # Find corresponding index in original signal
                click_time = time_display[display_idx]
                orig_idx = np.argmin(np.abs(self.time_ppg - click_time))
                
                # Check if peak already exists nearby
                if len(visible_peak_indices_display) > 0:
                    min_dist = np.min(np.abs(time_display[visible_peak_indices_display] - click_time))
                    if min_dist < 50:  # ms threshold
                        print(f"Peak already exists nearby at time {click_time:.2f}")
                        return
                
                # Add new peak
                visible_peak_indices_display = np.append(visible_peak_indices_display, display_idx)
                visible_peak_indices_display = np.sort(visible_peak_indices_display)
                
                # Calculate features for new peak (no pre-calculated prominence for new peaks)
                new_features = self.calculate_features_for_peak(orig_idx, current_peak_indices, 
                                                              self.filtered_ppg, self.time_ppg, 
                                                              prominence=None)
                
                # Add to data structures
                current_peak_indices = np.append(current_peak_indices, orig_idx)
                current_features = np.vstack([current_features, new_features])
                self.final_labels = np.append(self.final_labels, 1)  # New peak is true
                
                print(f"Added peak at time {click_time:.2f}, index {orig_idx}")
                update_plot()

            elif event.button == 3 and len(visible_peak_indices_display) > 0:  # right click → remove peak
                # Find nearest visible peak
                nearest_display_idx = np.argmin(np.abs(time_display[visible_peak_indices_display] - click_x))
                removed_display_idx = visible_peak_indices_display[nearest_display_idx]
                removed_time = time_display[removed_display_idx]
                
                # Find corresponding peak in original arrays
                for i, peak_idx in enumerate(current_peak_indices):
                    if abs(self.time_ppg[peak_idx] - removed_time) < 10:  # ms tolerance
                        self.final_labels[i] = 0  # Mark as false
                        break
                
                # Remove from display
                visible_peak_indices_display = np.delete(visible_peak_indices_display, nearest_display_idx)
                
                print(f"Removed peak at time {removed_time:.2f}")
                update_plot()
        
        # Connect mouse events
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)

        plt.show()

        print(f"\nFinal peak count: {np.sum(self.final_labels == 1)} true, {np.sum(self.final_labels == 0)} false")

    def save_data(self):
        """Save the feature matrix and labels to CSV file."""
        folder_name = os.path.basename(self.data_loc.rstrip("/"))
        
        # Create model-data directory if it doesn't exist
        model_data_dir = "model-data"
        if not os.path.exists(model_data_dir):
            os.makedirs(model_data_dir)
            print(f"Created directory: {model_data_dir}")
        
        # Create feature names
        feature_names = [
            'amplitude', 'prominence', 'width_half_prom', 'pulse_area',
            'rise_time', 'decay_time', 'max_upslope', 'max_inflection',
            'ibi_prev', 'ibi_next', 'ibi_ratio', 'local_variance', 'snr',
            'freq_energy', 'wavelet_coef', 'label'
        ]
        
        # Only use the original peaks and their labels (no newly added peaks)
        # This ensures consistency between initial detection and final labeling
        final_labels_trimmed = self.final_labels[:len(self.peak_indices_initial)]
        
        # Combine features and labels
        data_matrix = np.column_stack([self.initial_features, final_labels_trimmed])
        
        # Create DataFrame
        df = pd.DataFrame(data_matrix, columns=feature_names)
        
        # Add metadata columns (not for model training)
        df.insert(0, 'peak_idx', self.peak_indices_initial)
        df.insert(1, 'peak_time', self.peak_times_initial)
        
        # Save to CSV in model-data folder
        save_path = os.path.join(model_data_dir, f'{folder_name}_features.csv')
        df.to_csv(save_path, index=False)
        
        print(f"Saved feature data to {save_path}")
        print(f"Data shape: {df.shape}")
        print(f"Features saved: {len(feature_names)}")
        
        # Print summary
        print(f"\nDataset Summary:")
        print(f"Total initial peaks: {len(self.peak_indices_initial)}")
        print(f"True peaks (label=1): {np.sum(final_labels_trimmed == 1)}")
        print(f"False peaks (label=0): {np.sum(final_labels_trimmed == 0)}")


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