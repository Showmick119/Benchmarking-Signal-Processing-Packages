# HRV computation (SDNN, RMSSD etc.)
import numpy as np

def get_segments(time_ecg, ecg, time_ppg, ppg, peak_indices_ecg = None, peak_indices_ppg = None, window_time = 15, overlap_time = 5, path = None):
    """
        Segment ECG and PPG signals into aligned overlapping time windows
        of fixed duration (window_time), with a user-defined overlap (overlap_time).
        Each segment is defined based on time, not number of samples, to account for potnetial
        differences in sampling rates between ECG and PPG signals.

        Args:
            time_ecg (array): Time vector for ECG signal, in milliseconds.
            ecg (array): ECG signal amplitudes.
            time_ppg (array): Time vector for PPG signal, in milliseconds.
            ppg (array): PPG signal amplitudes.
            peak_indices_ecg (array): location of peaks in ECG (optional)
            peak_indices_ppg (array): location of peaks in PPG (optional)
            window_time (float): Segment window duration in seconds (default = 15).
            overlap_time (float): Overlap between consecutive windows in seconds (default = 5).

        Returns:
            segments (list): List of valid overlapping segments. 
            Each element is a tuple:
                If peaks are provided:
            (ppg_segment, ecg_segment, ppg_time_segment, ecg_time_segment, segment_peak_indices_ppg, segment_peak_indices_ecg, segment_start_time, segment_end_time)

                If peaks are not provided:
                    (ppg_segment, ecg_segment, ppg_time_segment, ecg_time_segment, segment_start_time, segment_end_time)
            where:
                ppg_segment (array): PPG samples within the segment.
                ecg_segment (array): ECG samples within the segment.
                ppg_time_segment (array): Corresponding PPG time vector.
                ecg_time_segment (array): Corresponding ECG time vector.
                segment_peak_indices_ppg (array): Indices for peaks in PPG.
                segment_peak_indices_ecg (array): Indices for peaks in ECG.
                segment_start_time (float): Start time of the segment (ms).
                segment_end_time (float): End time of the segment (ms).

        Notes:
            - The final segment is forced to align with the end of the signal.
            - Segments with insufficient data are automatically discarded.
            - PPG and ECG signals can have different sampling rates.
    """

    # get segments of ECG and PPG based on window time in milliseconds
    window_time_ms = window_time * 1000
    overlap_time_ms = overlap_time * 1000

    # Chack for valid parameters
    if overlap_time >= window_time:
        raise ValueError("Overlap_time must be smaller than window_time.")
    
    # Determine start/end times
    start_time = 0
    end_time = window_time_ms

    # Find max possible time range from both signals
    max_time = min(time_ecg[-1], time_ppg[-1])

    segments = []

    def get_segment_peaks(peak_indices, time_array, seg_start, seg_end):
        peak_mask = (time_array[peak_indices] >= seg_start) & (time_array[peak_indices] < seg_end)
        global_indidces = peak_indices[peak_mask]
        first_sample_idx = np.searchsorted(time_array, seg_start)
        local_indices = global_indidces - first_sample_idx
        return local_indices

    while end_time <= max_time:
        # ECG segment
        ecg_mask = (time_ecg >= start_time) & (time_ecg < end_time)
        ecg_segment = ecg[ecg_mask]
        ecg_time_segment = time_ecg[ecg_mask]

        # PPG segment
        ppg_mask = (time_ppg >= start_time) & (time_ppg < end_time)
        ppg_time_segment = time_ppg[ppg_mask]
        ppg_segment = ppg[ppg_mask]

        # min-max normalization for each ppg_segment
        try:
            ppg_segment = (ppg_segment - np.min(ppg_segment))/(np.max(ppg_segment) - np.min(ppg_segment))
        except Exception as e:
            print (f'\nPath: {path}')
            print (f'Failed min-max normalization for segment starting at {start_time} ms')
            start_time += (window_time_ms - overlap_time_ms)
            end_time = start_time + window_time_ms
            continue

        # Only append segments if both ECG and PPG have enough samples
        min_ppg_samples = 0.8 * (window_time * (len(time_ppg) / (time_ppg[-1] / 1000)))
        min_ecg_samples = 0.8 * (window_time * (len(time_ecg) / (time_ecg[-1] / 1000)))

        if len(ppg_segment) >= min_ppg_samples and len(ecg_segment) >= min_ecg_samples:
            if peak_indices_ecg is not None and peak_indices_ppg is not None:
                segment_peak_indices_ecg = get_segment_peaks(peak_indices_ecg, time_ecg, start_time, end_time)
                segment_peak_indices_ppg = get_segment_peaks(peak_indices_ppg, time_ppg, start_time, end_time)
                # segments.append(
                #     (ppg_segment, ecg_segment, ppg_time_segment, ecg_time_segment, 
                #      segment_peak_indices_ppg, segment_peak_indices_ecg,
                #      start_time, end_time)
                # )
                segments.append({
                    "ppg_segment": ppg_segment,
                    "ecg_segment": ecg_segment,
                    "ppg_time_segment": ppg_time_segment,
                    "ecg_time_segment": ecg_time_segment,
                    "peak_indices_ppg": segment_peak_indices_ppg,
                    "peak_indices_ecg": segment_peak_indices_ecg,
                    "start_time": start_time,
                    "end_time": end_time
                })
            else:
                segment_peak_indices_ecg = get_segment_peaks(peak_indices_ecg, time_ecg, start_time, end_time)
                # segments.append(
                #     (ppg_segment, ecg_segment, ppg_time_segment, ecg_time_segment, segment_peak_indices_ecg,
                #      start_time, end_time)
                # )
                segments.append({
                    "ppg_segment": ppg_segment,
                    "ecg_segment": ecg_segment,
                    "ppg_time_segment": ppg_time_segment,
                    "ecg_time_segment": ecg_time_segment,
                    "peak_indices_ecg": segment_peak_indices_ecg,
                    "start_time": start_time,
                    "end_time": end_time
                })

        # Move window
        start_time += (window_time_ms - overlap_time_ms)
        end_time = start_time + window_time_ms

    
    # Use all available data even if the final window heavily overlaps with the previous window
    if end_time - (window_time_ms - overlap_time_ms) < max_time:
        # Add final right-aligned segment
        final_start = max_time - window_time_ms
        final_end = max_time
        
        ecg_mask = (time_ecg >= final_start) & (time_ecg < final_end)
        ecg_segment = ecg[ecg_mask]
        ecg_time_segment = time_ecg[ecg_mask]

        ppg_mask = (time_ppg >= final_start) & (time_ppg < final_end)
        ppg_time_segment = time_ppg[ppg_mask]
        ppg_segment = ppg[ppg_mask]

        # min-max normalization for each ppg_segment
        try:
            ppg_segment = (ppg_segment - np.min(ppg_segment))/(np.max(ppg_segment) - np.min(ppg_segment))
        except Exception as e:
            print (f'Failed min-max normalization for last segment')


        # Robustness check as before
        min_ppg_samples = 0.8 * (window_time * (len(time_ppg) / (time_ppg[-1] / 1000)))
        min_ecg_samples = 0.8 * (window_time * (len(time_ecg) / (time_ecg[-1] / 1000)))

        if len(ppg_segment) >= min_ppg_samples and len(ecg_segment) >= min_ecg_samples:
            #segments.append( (ppg_segment, ecg_segment, ppg_time_segment, ecg_time_segment, final_start, final_end) )
            if peak_indices_ecg is not None and peak_indices_ppg is not None:
                segment_peak_indices_ecg = get_segment_peaks(peak_indices_ecg, time_ecg, final_start, final_end)
                segment_peak_indices_ppg = get_segment_peaks(peak_indices_ppg, time_ppg, final_start, final_end)
                # segments.append((
                #     ppg_segment, ecg_segment, ppg_time_segment, ecg_time_segment,
                #     segment_peak_indices_ppg, segment_peak_indices_ecg,
                #     final_start, final_end
                # ))
                segments.append({
                    "ppg_segment": ppg_segment,
                    "ecg_segment": ecg_segment,
                    "ppg_time_segment": ppg_time_segment,
                    "ecg_time_segment": ecg_time_segment,
                    "peak_indices_ppg": segment_peak_indices_ppg,
                    "peak_indices_ecg": segment_peak_indices_ecg,
                    "start_time": final_start,
                    "end_time": final_end
                })
            else:
                segment_peak_indices_ecg = get_segment_peaks(peak_indices_ecg, time_ecg, final_start, final_end)
                # segments.append((
                #     ppg_segment, ecg_segment, ppg_time_segment, ecg_time_segment, segment_peak_indices_ecg,
                #     final_start, final_end
                # ))
                segments.append({
                    "ppg_segment": ppg_segment,
                    "ecg_segment": ecg_segment,
                    "ppg_time_segment": ppg_time_segment,
                    "ecg_time_segment": ecg_time_segment,
                    "peak_indices_ecg": segment_peak_indices_ecg,
                    "start_time": final_start,
                    "end_time": final_end
                })

    return segments



def remove_noisy_segments(segments):
    """
    Filter out noisy segments based on physiological feasibility rules.
    NOTE: This method does not produce good results !!! 
    Args:
        segments (list): List of segments from ECG and corresponding PPG signal

    Returns:
        clean_segments (list): List of segments that pass all quality checks
    """

    clean_segments = []

    for segment in segments:
        #_, _, ppg_time, ecg_time, peak_ppg_idx, peak_ecg_idx, start, end = segment
        ppg = segment['ppg_segment']
        ecg = segment['ecg_segment']
        ppg_time = segment['ppg_time_segment']
        ecg_time = segment['ecg_time_segment']
        peak_ppg_idx = segment['peak_indices_ppg']
        peak_ecg_idx = segment['peak_indices_ecg']
        start = segment['start_time']
        end = segment['end_time']

        segment_duration = (end - start)/1000.0  # Segment duration in seconds
        peak_ppg_times = ppg_time[peak_ppg_idx]
        peak_ecg_times = ecg_time[peak_ecg_idx]

        rr_ppg = np.diff(peak_ppg_times)
        rr_ecg = np.diff(peak_ecg_times)

        if len(peak_ppg_idx) < 2 or len(peak_ecg_idx) < 2:
            continue # skip segment with insufficient peaks
        
        avg_hr_ecg = 60_000 / np.mean(rr_ecg)
        avg_hr_ppg = 60_000 / np.mean(rr_ppg)

        # Check 1: Average HR must be within 40-220 bpm
        if not (40 <= avg_hr_ecg <= 220) or not (40 <= avg_hr_ppg <= 220):
            print ('Check 1 Failed')
            continue

        # Check 2: Maximum interval between peaks
        if np.max(rr_ecg) > 3000 or np.max(rr_ppg) > 3000:
            print ('Check 2 Failed')
            continue
        
        # # Check 3: Ratio of max to min interval
        # if segment_duration <= 30:
        #     if len(rr_ppg) >= 2:
        #         ratio_ppg = np.max(rr_ppg) / np.min(rr_ppg)
        #         if ratio_ppg > 2.2:
        #             print ('Check 3: PPG Failed')
        #             continue
        

        #     if len(rr_ecg) >= 2:
        #         ratio_ecg = np.max(rr_ecg) / np.min(rr_ecg)
        #         if ratio_ecg > 1.1:
        #             print ('Check 3: ECG Failed')
        #             continue

        # # For longer segments
        # if segment_duration > 30:
        #     # HR variation <= 25 for ECG
        #     min_hr_ecg = 60_000 / np.max(rr_ecg)
        #     max_hr_ecg = 60_000 / np.min(rr_ecg)
        #     if (max_hr_ecg - min_hr_ecg) > 0.25 * avg_hr_ecg:
        #         continue
            
        #     # HR variation <= 30 for PPG
        #     min_hr_ppg = 60000 / np.max(rr_ppg)
        #     max_hr_ppg = 60000 / np.min(rr_ppg)
        #     if (max_hr_ppg - min_hr_ppg) > 0.30 * avg_hr_ppg:
        #         continue

        # If all checks passed, keep the segment
        clean_segments.append(segment)

    return clean_segments


def hrv_metrics(time, peak_indices):
    """
    Get HRV statistics
    """
    peak_times = time[peak_indices]
    rr_intervals = np.diff(peak_times)

    if len(rr_intervals) < 2:
        return None
    
    # RMSSD
    #diff_rr = np.diff(rr_intervals)
    #rmssd = np.sqrt(np.mean(diff_rr**2))

    # SDNN
    sdnn = np.std(rr_intervals, ddof=1)

    return sdnn.item()


    

