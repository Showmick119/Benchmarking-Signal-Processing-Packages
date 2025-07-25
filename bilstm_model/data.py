# signal loading, preprocessing, segmentation
# Creates .npz files from Raw Data
import os
import yaml
import numpy as np
from scipy.signal import find_peaks
from misc import LoadDataset, Filters, XQRS
from helper import get_segments, remove_noisy_segments, hrv_metrics

class GenerateDataset:
    """
    Creates dataset for training from raw data.
    Drops unusable segment(s) in the PPG signal (optional)..see config file.
    """
    def __init__(self, config_path = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_path = self._resolve_data_path()
        self.quality_assessment = self.config["generate_dataset"]["quality_assessment"]
        self.ecg_window = self.config["generate_dataset"]["Filter"]["ecg_window"]
        self.window_time = self.config["generate_dataset"]["Segmentation"]["window_size"]
        self.overlap_time = self.config["generate_dataset"]["Segmentation"]["overlap"]
        
        print ('\nGenerating dataset with the following parameters:')
        print (f'Data Path: {self.data_path}')
        print (f'Quality Assessment of PPG segments: {self.quality_assessment}\nECG window_time: {self.window_time} sec\nOverlap_time between segments: {self.overlap_time}\n')

    def _resolve_data_path(self):
        path = self.config["generate_dataset"]["path"]
        root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        return os.path.join(root, path)
    

    def _get_sample_paths(self):
        data_samples = []
        for folder in os.listdir(self.data_path):
            for individual in os.listdir(os.path.join(self.data_path, folder)):
                if individual.endswith(".zip"):
                    continue

                id_path = os.path.join(self.data_path, folder, individual)

                for state in os.listdir(id_path):
                    sample = os.path.join(id_path, state)
                    data_samples.append(sample)
        
        return data_samples


    def _process_signal(self, ecg_time, ecg, ppg_time, ppg):
        xqrs = XQRS()
        peak_times_ecg, peak_indices_ecg = xqrs.find_peaks(ecg_time, ecg, self.ecg_window, refinement=True)
        global_sdnn = hrv_metrics(ecg_time, peak_indices_ecg)

        filter = Filters()
        if self.config["generate_dataset"]["Filter"]["filter_type"].lower() == "chebyshev":
            filtered_ppg = filter.cheby2_filter(ppg_time, ppg, plot = False)
        else:
            filtered_ppg = filter.butter_filter(ppg_time, ppg, plot = False)
        
        norm_ppg = (filtered_ppg - np.mean(filtered_ppg))/np.std(filtered_ppg)

        return norm_ppg, global_sdnn, peak_indices_ecg


    def _segment_signals(self, ecg_time, ecg, ppg_time, norm_ppg, peak_indices_ecg, sample_path):
        try:
            if self.quality_assessment:
                peak_indices_ppg, _ = find_peaks(norm_ppg, height = np.mean(norm_ppg))
                segments = get_segments(
                    ecg_time, ecg, ppg_time, norm_ppg,
                    peak_indices_ecg, peak_indices_ppg,
                    self.window_time, self.overlap_time,
                    path=sample_path
                )
                return remove_noisy_segments(segments)
            else:
                return get_segments(
                    ecg_time, ecg, ppg_time, norm_ppg,
                    peak_indices_ecg, 
                    window_time = self.window_time, overlap_time = self.overlap_time,
                    path = sample_path
                )
            
        except Exception as e:
            print (f'Segmentation Failed for {sample_path}: {str(e)}')
            return None
            

    def process_dataset(self):
        output_dir = "Data/Dataset_processed" if self.quality_assessment else "Data/Dataset_raw"
        os.makedirs(output_dir, exist_ok = True)

        processed_samples = []
        for idx, sample_path in enumerate(self._get_sample_paths()):
            try:
                load_sample = LoadDataset(sample_path)
                ecg_time, ecg, ppg_time, ppg, _, _ = load_sample.get_data()
                norm_ppg, global_sdnn, peak_indices_ecg = self._process_signal(ecg_time, ecg, ppg_time, ppg)

                if global_sdnn > 250:
                    print (f'Abnormal True SDNN: {global_sdnn}. Skipping to next sample...')
                    continue
                segments = self._segment_signals(ecg_time, ecg, ppg_time, norm_ppg, peak_indices_ecg, sample_path)

                output_path = os.path.join(output_dir, f"sample_{idx}.npz")
                np.savez_compressed(output_path, segments = np.array(segments, dtype=object), label = global_sdnn)

                processed_samples.append(output_path)

            except Exception as e:
                print (f'\nFailed to process {sample_path}: {str(e)}')
                continue

        return processed_samples
    

if __name__ == "__main__":
    generate_dataset = GenerateDataset()
    processed_files = generate_dataset.process_dataset()
    print (f"\nSuccessfully processed {len(processed_files)} files.")