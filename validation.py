import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from utils import LoadDataset, XQRS, Filters
from scipy.signal import find_peaks

class Validation:
    """
    Class to process raw PPG/ECG signals and validate PPG signals.
    """

    def __init__(self):
        pass


    def get_data(self, data_loc, plot = False):
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


    def get_peaks(self, ecg_window = 20, ppg_window = 20, filter_type = 'Butterworth'):
        """
        Automatically detect peaks in ECG and PPG signals using the XQRS algorithm with optional refinement.

        Args:
            ecg_window (int): Window size (in sample points) for refining ECG peaks.
            ppg_window (int): Window size (in sample points) for refining PPG peaks.
        """

        if filter_type.lower() == 'xqrs':    
            xqrs = XQRS()
            print ('**Using XQRS Filter**')
            self.peak_times_ecg, self.refined_peaks_ecg = xqrs.find_peaks(self.time_ecg, self.ecg, search_window=ecg_window, refinement=True)
            self.peak_times_ppg, self.refined_peaks_ppg = xqrs.find_peaks(self.time_ppg, self.ppg, search_window=ppg_window, refinement=True)
            
            xqrs.plot_peaks(self.time_ecg, self.ecg,
                            self.time_ppg, self.ppg,
                            self.peak_times_ecg, self.refined_peaks_ecg,
                            self.peak_times_ppg, self.refined_peaks_ppg,
                            start = 0, end = None)
            
        elif filter_type.lower() == 'butterworth':
            filter = Filters()
            print ('**Using Butterworth Type-II Filter**')
            print ('')
            filtered_ecg = filter.butter_filter(self.time_ecg, self.ecg, order=4, signal_type='ECG', plot=False)
            filtered_ppg = filter.butter_filter(self.time_ppg, self.ppg, order=4, signal_type='PPG', plot=False)

            self.refined_peaks_ecg, _ = find_peaks(filtered_ecg, height = np.mean(filtered_ecg))
            self.refined_peaks_ppg, _ = find_peaks(filtered_ppg, height = np.mean(filtered_ppg))

            self.peak_times_ecg = self.time_ecg[self.refined_peaks_ecg]
            self.peak_times_ppg = self.time_ppg[self.refined_peaks_ppg]

        
        elif filter_type.lower() == 'chebyshev':
            filter = Filters()
            print ('**Using Chebyshev Type-II Filter**')
            print ('')
            filtered_ecg = filter.cheby2_filter(self.time_ecg, self.ecg, order=4, rs=40, signal_type='ECG', plot=False)
            filtered_ppg = filter.cheby2_filter(self.time_ppg, self.ppg, order=4, rs=40, signal_type='PPG', plot=False)

            self.refined_peaks_ecg, _ = find_peaks(filtered_ecg, height = np.mean(filtered_ecg))
            self.refined_peaks_ppg, _ = find_peaks(filtered_ppg, height = np.mean(filtered_ppg))

            self.peak_times_ecg = self.time_ecg[self.refined_peaks_ecg]
            self.peak_times_ppg = self.time_ppg[self.refined_peaks_ppg]

    
    def remove_unusable_signal(self):
        """
        GUI tool to remove unsuable part in the PPG signal. Also, removes corresponding part in the ECG signal.
        """

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(25,8), sharex=True)

         # Plot ECG
        line_ecg, = ax1.plot(self.time_ecg, self.ecg, lw=1)
        markers_ecg = ax1.scatter(self.peak_times_ecg, self.ecg[self.refined_peaks_ecg], marker = 'x', color = 'black')
        ax1.set_title('ECG Signal')
        ax1.grid(which='major', linestyle='--', linewidth=0.7)
        ax1.minorticks_on()
        ax1.grid(which='minor', linestyle=':', linewidth=0.4)

        # Plot PPG
        line_ppg, = ax2.plot(self.time_ppg, self.ppg, color = 'red', lw=1)
        markers_ppg = ax2.scatter(self.peak_times_ppg, self.ppg[self.refined_peaks_ppg], marker = 'x', color = 'black')
        ax2.set_title('PPG Signal (Select Unsuable Regions)')
        ax2.grid(which='major', linestyle='--', linewidth=0.7)
        ax2.minorticks_on()
        ax2.grid(which='minor', linestyle=':', linewidth=0.4)

        self.remove_regions = []  # list to store (xmin, xmax) to remove

        def onselect(xmin, xmax):
            xmin, xmax = xmin.item(), xmax.item()

            if xmin < 0:
                xmin = 0

            self.remove_regions.append((xmin, xmax))
            # shade selected region
            ax1.axvspan(xmin, xmax, color = 'red', alpha = 0.3)
            ax2.axvspan(xmin, xmax, color = 'red', alpha = 0.3)
            fig.canvas.draw()

        span = SpanSelector(
            ax2,
            onselect,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='red')
        )

        print ("Select unsuable regions on the PPG plot using mouse. Close the window when done.\n")
        plt.show()

        # After selection, remove all the selected regions from signals.
        def mask_signal(time, regions):
            mask = np.ones_like(time, dtype=bool)
            for xmin, xmax in regions:
                mask &= ~((time >= xmin) & (time <= xmax))
            return mask
        

        # Create masks
        ppg_mask = mask_signal(self.time_ppg, self.remove_regions)
        ecg_mask = mask_signal(self.time_ecg, self.remove_regions)
        
        # Save originals for later mapping
        self.time_ecg_orig = self.time_ecg.copy()
        self.time_ppg_orig = self.time_ppg.copy()
        self.refined_peaks_ecg_orig = self.refined_peaks_ecg.copy()
        self.refined_peaks_ppg_orig = self.refined_peaks_ppg.copy()
        
        # Apply masks
        self.time_ppg = self.time_ppg[ppg_mask]
        self.ppg = self.ppg[ppg_mask]
        self.time_ecg = self.time_ecg[ecg_mask]
        self.ecg = self.ecg[ecg_mask]

        # Get times of original peaks
        refined_peaks_ecg_times = self.time_ecg_orig[self.refined_peaks_ecg_orig]
        refined_peaks_ppg_times = self.time_ppg_orig[self.refined_peaks_ppg_orig]

        # Keep only those peaks whose time remains in masked signal
        self.refined_peaks_ecg = np.where(np.isin(self.time_ecg, refined_peaks_ecg_times))[0]
        self.refined_peaks_ppg = np.where(np.isin(self.time_ppg, refined_peaks_ppg_times))[0]

        print(f"Removed {len(self.remove_regions)} region(s) from signals.")
        print ('')


    def correct_peaks(self):
        """
        Launch an interactive matplotlib GUI for manual correction of detected PPG peaks.

        This method displays two subplots:
            - Top: ECG signal with automatically detected peaks (read-only, for reference).
            - Bottom: PPG signal with detected peaks that can be manually edited.

        Interaction instructions:
            - **Left Click**: Add a peak at the nearest signal point to the clicked x-coordinate.
            - **Right Click**: Remove the nearest existing peak.
            - **Close the Window**: Finalize editing and store the corrected peak indices.

        Side Effects:
            - Updates the following instance variables:
                self.corrected_peaks_ppg (np.ndarray): Indices of manually corrected PPG peaks.
                self.corrected_peak_times_ppg (np.ndarray): Corresponding timestamps of the corrected peaks.

        Notes:
            - Only interactions on the PPG subplot are processed.
            - Original peaks are shown in blue, and manually added peaks are marked in red.
            - ECG plot is shown for reference and cannot be edited.

        Raises:
            ValueError: If time or signal data are not initialized (e.g., get_data() not called first).
        """
        
        press_event = {'x': None, 'y': None}
        click_threshold = 5  # pixels

        x, y = self.time_ppg, self.ppg
        peak_indices = self.refined_peaks_ppg.copy()

        x_ecg, y_ecg = self.time_ecg, self.ecg
        peak_indices_ecg = self.refined_peaks_ecg.copy()


        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(25,8), sharex=True)

        # ECG (reference only)
        ax1.plot(x_ecg, y_ecg, lw=1)
        ax1.scatter(x_ecg[peak_indices_ecg], y_ecg[peak_indices_ecg], c='k', marker='x')
        ax1.set_title('ECG (Reference Only)')
        ax1.grid(which='major', linestyle='--', linewidth=0.7)
        ax1.minorticks_on()
        ax1.grid(which='minor', linestyle=':', linewidth=0.4)

        # PPG (interactive)
        line, = ax2.plot(x, y, color = 'red', lw=1)
        scatter = ax2.scatter(x[peak_indices], y[peak_indices], c = 'b', marker = 'x')
        ax2.set_title("Left Click: Add peak, Right Click: Remove peak\nClose window when done")
        ax2.grid(which='major', linestyle='--', linewidth=0.7)
        ax2.minorticks_on()
        ax2.grid(which='minor', linestyle=':', linewidth=0.4)


        def update_plot():
            scatter.set_offsets(np.c_[x[peak_indices], y[peak_indices]])
            fig.canvas.draw_idle()
        
        def on_press(event):
            if event.inaxes == ax2:
                press_event['x'], press_event['y'] = event.x, event.y

        def on_release(event):
            if event.inaxes != ax2 or event.xdata is None:
                return

            dx = abs(event.x - press_event['x'])
            dy = abs(event.y - press_event['y'])
            if dx > click_threshold or dy > click_threshold:
                return  # it was a drag, not a click

            click_x = event.xdata
            nonlocal peak_indices

            if event.button == 1:  # left click → add peak
                idx = np.argmin(np.abs(x - click_x))
                if idx not in peak_indices:
                    peak_indices = np.sort(np.append(peak_indices, idx))
                    print(f"Added peak at index {idx}, time {x[idx]:.2f}")
                    update_plot()

            elif event.button == 3 and len(peak_indices) > 0:  # right click → remove peak
                nearest_idx = np.argmin(np.abs(x[peak_indices] - click_x))
                print(f"Removed peak at index {peak_indices[nearest_idx]}, time {x[peak_indices[nearest_idx]]:.2f}")
                peak_indices = np.delete(peak_indices, nearest_idx)
                update_plot()

        # Connect mouse click event
        #cid = fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)


        # Block execution until the window is closed
        plt.show()

        # store corrected peaks back
        self.corrected_peaks_ppg = peak_indices
        self.corrected_peak_times_ppg = x[peak_indices]
        print(f"\nFinal number of corrected peaks: {len(self.corrected_peaks_ppg)}")



    def compute_rr_intervals(self, peak_times, remove_regions):
        """
        Compute RR intervals from peak times, excluding intervals that span removed regions.

        Args:
            peak_times (np.ndarray): 1D array of timestamps (ECG or PPG peaks in ms)
            remove_regions (list of (float, float)): List of time intervals that were removed.

        Returns:
            rr_intervals (list): RR intervals computed only within continuous segments.
        """

        rr_intervals = []

        # sort remove_regions just in case
        remove_regions = sorted(remove_regions, key=lambda x:x[0])

        print (f"\nRemoved regions: {remove_regions}")

        # Iterate over peak intervals
        for i in range(len(peak_times) - 1):
            t1 = peak_times[i]
            t2 = peak_times[i+1]

            # Check if t1 to t2 crosses a removed region
            crosses_gap = False
            for start, end in remove_regions:
                if t1 < start < t2 or t1 < end < t2 or (t1 >= start and t1 <= end) or (t2 >= start and t2 <= end):
                    crosses_gap = True
                    print (f'Gap Crossed. Not computing RR between {t1:.3f} and {t2:.3f}')
                    break
            
            if not crosses_gap:
                rr_intervals.append((t2 - t1).item())

        return rr_intervals

    
    def get_statistics(self):
        """
        https://knowledge.time2tri.me/en/articles/main-hrv-parameters-part-1-time-related-parameters
        """

        rr_ecg = self.compute_rr_intervals(self.peak_times_ecg, self.remove_regions)

        if hasattr(self, 'corrected_peak_times_ppg'):
            rr_ppg = self.compute_rr_intervals(self.corrected_peak_times_ppg, self.remove_regions)
        else:
            rr_ppg = self.compute_rr_intervals(self.peak_times_ppg, self.remove_regions)


        def hrv_metrics(rr_intervals):
            if len(rr_intervals) < 2:
                return np.nan, np.nan, np.nan, np.nan

            # RMSSD
            diff_rr = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(diff_rr**2))

            # SDNN
            sdnn = np.std(rr_intervals, ddof=1)

            # pNN20
            pnn20 = (np.sum(np.abs(diff_rr) > 20) / len(diff_rr) ) * 100

            # pNN50
            pnn50 = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr) ) * 100

            return rmssd.item(), sdnn.item(), pnn20.item(), pnn50.item()

        rmssd_ecg, sdnn_ecg, pNN20_ecg, pNN50_ecg = hrv_metrics(rr_ecg)
        rmssd_ppg, sdnn_ppg, pNN20_ppg, pNN50_ppg = hrv_metrics(rr_ppg)

        print ('======================================================================')
        print (f'RMSSD ECG: {rmssd_ecg:.3f} \nSDNN ECG: {sdnn_ecg:.3f} \npNN20 ECG: {pNN20_ecg:.3f} \npNN50 ECG: {pNN50_ecg:.3f}')
        print ('')
        print (f'RMSSD PPG: {rmssd_ppg:.3f} \nSDNN PPG: {sdnn_ppg:.3f} \npNN20 PPG: {pNN20_ppg:.3f} \npNN50 PPG: {pNN50_ppg:.3f}')
        print ('======================================================================')
        
        # return {
        #     'ECG': {
        #         'RMSSD': rmssd_ecg,
        #         'SDNN': sdnn_ecg,
        #         'pNN50': pNN50_ecg,
        #         'pNN20': pNN20_ecg,
        #     },
        #     'PPG': {
        #         'RMSSD': rmssd_ppg,
        #         'SDNN': sdnn_ppg,
        #         'pNN50': pNN50_ppg,
        #         'pNN20': pNN20_ppg,
        #     }
        # }
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison of HRV estimates from PPG signals between processing approaches.")
    parser.add_argument("--data_loc", type=str, required=True, help="Path to data")
    parser.add_argument("--filter_type", type=str, required=True, help="Butterworth, Chebyshev or XQRS")

    args = parser.parse_args()

    validate = Validation()
    validate.get_data(args.data_loc)
    validate.get_peaks(filter_type=args.filter_type)
    validate.remove_unusable_signal()
    validate.correct_peaks()
    validate.get_statistics()