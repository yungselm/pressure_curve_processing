import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os


class SignalProcessing:
    def __init__(self, file_path, output_path):
        """
        Initializes the SignalProcessing object with a file path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        self.file_path = file_path
        self.output_path = output_path
        self.data = pd.read_csv(file_path)
        self.processed_data = None
    
    def __call__(self):
        """
        Runs all processing steps and saves results.
        """
        self._smooth_signals()
        self.preprocess_data()
        self.refind_peaks()
        self.find_saddle_point_with_trimmed_interval()
        # print(f'Number of saddle points: {len(self.data[self.data["peaks"] == 3])}')
        self.calculate_ifr()
        # print(f'iFR: {round(self.data["iFR"].mean(), 2)}')
        self.calculate_systolic_measures()
        # print(f'Mid-systolic ratio: {round(self.data["mid_systolic_ratio"].mean(), 2)}')
        self.save_results(self.output_path)

        return self.data

    def _smooth_signals(self, window=10):
        """
        Smooths the signals using a rolling mean.
        """
        self.data['p_aortic_smooth'] = self.data['p_aortic'].rolling(window=window).mean()
        self.data['p_distal_smooth'] = self.data['p_distal'].rolling(window=window).mean()

    def preprocess_data(self):
        """
        Checks for the longest segment where peaks == 1 and 2 alternate, and keeps only that segment.
        Peaks can have values of 1, 2, or 0 (non-peaks), and only alternating 1 and 2 are considered valid.
        """
        df_copy = self.data.copy()
        peaks = df_copy['peaks'].tolist()
        
        longest_segment_indices = []
        current_segment_indices = []
        
        for idx, peak in enumerate(peaks):
            # Skip 0s since they do not represent valid peaks
            if peak == 0:
                continue
            
            if not current_segment_indices:
                current_segment_indices.append(idx)
            else:
                last_peak = df_copy.loc[current_segment_indices[-1], 'peaks']
                
                # Check if current peak alternates with the last valid peak
                if peak != last_peak:
                    current_segment_indices.append(idx)
                else:
                    # If alternation breaks, finalize the current segment
                    if len(current_segment_indices) > len(longest_segment_indices):
                        longest_segment_indices = current_segment_indices
                    current_segment_indices = [idx]  # Start a new segment with the current index
        
        # Check the last segment
        if len(current_segment_indices) > len(longest_segment_indices):
            longest_segment_indices = current_segment_indices
        
        # Keep only the rows corresponding to the longest alternating segment
        start_idx = longest_segment_indices[0]
        end_idx = longest_segment_indices[-1]
        self.data = df_copy.iloc[start_idx:end_idx + 1].reset_index(drop=True)

    def refind_peaks(self, signal='p_aortic_smooth'):
        """
        Refines systolic and diastolic peaks by resetting them between intervals.
        """
        df_copy_systole = self.data.copy()
        df_copy_systole.loc[df_copy_systole['peaks'] == 1, 'peaks'] = 0
        df_copy_diastole = self.data.copy()
        df_copy_diastole.loc[df_copy_diastole['peaks'] == 2, 'peaks'] = 0

        systole_indices = df_copy_systole.index[self.data['peaks'] == 2]

        for i in range(len(systole_indices) - 1):
            start_idx = systole_indices[i]
            end_idx = systole_indices[i + 1]
            interval_data = df_copy_systole.loc[start_idx:end_idx, signal]
            peak_idx = interval_data.idxmax()
            df_copy_systole.at[peak_idx, 'peaks'] = 1

        diastole_indices = df_copy_diastole.index[self.data['peaks'] == 1]

        for i in range(len(diastole_indices) - 1):
            start_idx = diastole_indices[i]
            end_idx = diastole_indices[i + 1]
            interval_data = df_copy_diastole.loc[start_idx:end_idx, signal]
            peak_idx = interval_data.idxmin()
            df_copy_diastole.at[peak_idx, 'peaks'] = 2

        self.data['peaks'] = 0
        self.data['peaks'] = self.data['peaks'].mask(df_copy_systole['peaks'] == 1, 1)
        self.data['peaks'] = self.data['peaks'].mask(df_copy_diastole['peaks'] == 2, 2)

    def find_saddle_point_with_trimmed_interval(self, signal='p_aortic_smooth'):
        """
        Identifies saddle points between systolic and diastolic peaks.
        """
        df_copy = self.data.copy()

        # Reset any previous saddle points (peaks == 3)
        df_copy.loc[df_copy['peaks'] == 3, 'peaks'] = 0

        systole_indices = df_copy.index[df_copy['peaks'] == 1].tolist()
        diastole_indices = df_copy.index[df_copy['peaks'] == 2].tolist()

        # check if time at first systole < time at first diastole if not remove first diastole
        if df_copy.loc[systole_indices[0], 'time'] > df_copy.loc[diastole_indices[0], 'time']:
            diastole_indices.pop(0)

        for i in range(min(len(systole_indices), len(diastole_indices))):
            systole_idx = systole_indices[i]
            diastole_idx = diastole_indices[i]

            # Get the data range between systole and diastole
            interval_data = df_copy.loc[systole_idx:diastole_idx, signal]
            interval_time = df_copy.loc[systole_idx:diastole_idx, 'time']

            # remove 10 percent at each end
            interval_data = interval_data[int(len(interval_data) * 0.10) : int(len(interval_data) * 0.90)]
            interval_time = interval_time[int(len(interval_time) * 0.10) : int(len(interval_time) * 0.90)]

            # Detect the saddle point in the interval data
            peaks, _ = find_peaks(-interval_data)  # Find local minima by inverting the signal
            if len(peaks) > 0:
                saddle_idx = interval_data.index[peaks[0]]  # Take the first local minimum
            else:
                saddle_idx = None

            # If no saddle point is found, place it in the midpoint of systole and diastole
            if saddle_idx is None:
                saddle_idx = int((systole_idx + diastole_idx) / 2)

            # Mark the saddle point as '3' in the modified peaks column
            df_copy.at[saddle_idx, 'peaks'] = 3

        self.data = df_copy

    def calculate_ifr(self):
        """
        Calculates the iFR from the processed data.
        """
        df_copy = self.data.copy()

        aortic_indices = df_copy.index[df_copy['peaks'] == 3].tolist()
        diastole_indices = df_copy.index[df_copy['peaks'] == 2].tolist()

        # check if time at first systole < time at first diastole if not remove first diastole
        if df_copy.loc[aortic_indices[0], 'time'] > df_copy.loc[diastole_indices[0], 'time']:
            diastole_indices.pop(0)

        # Vectorize the iFR calculations
        min_len = min(len(aortic_indices), len(diastole_indices))

        # Calculate systolic integral and diastolic ratio for all pairs of aortic and diastole indices
        for i in range(min_len):
            aortic_idx = aortic_indices[i]
            diastole_idx = diastole_indices[i]

            # Get the data range between systole and diastole
            interval_aortic = self.data.loc[aortic_idx:diastole_idx, 'p_aortic_smooth']
            interval_distal = self.data.loc[aortic_idx:diastole_idx, 'p_distal_smooth']

            # Calculate systolic integral as the sum of the difference between p_aortic_smooth and p_distal_smooth
            # Calculate the baseline value as the mean of the values at diastole_indices[i] and diastole_indices[i+1]
            if i + 1 < len(diastole_indices):
                baseline_value = (self.data.at[diastole_indices[i], 'p_aortic_smooth'] + self.data.at[diastole_indices[i + 1], 'p_aortic_smooth']) / 2
                self.data.at[diastole_idx, 'diastolic_integral_aortic'] = np.sum(interval_aortic - baseline_value)

                baseline_value = (self.data.at[diastole_indices[i], 'p_distal_smooth'] + self.data.at[diastole_indices[i + 1], 'p_distal_smooth']) / 2
                self.data.at[diastole_idx, 'diastolic_integral_distal'] = np.sum(interval_distal - baseline_value)

                self.data.at[diastole_idx, 'diastolic_integral_diff'] = self.data.at[diastole_idx, 'diastolic_integral_aortic'] - self.data.at[diastole_idx, 'diastolic_integral_distal']

            # Remove 25% at each end and 10% from the other end (keeping 90% of the middle part)
            interval_aortic = interval_aortic[int(len(interval_aortic) * 0.25):int(len(interval_aortic) * 0.90)]
            interval_distal = interval_distal[int(len(interval_distal) * 0.25):int(len(interval_distal) * 0.90)]

            # Calculate the diastolic ratio as p_distal_smooth / p_aortic_smooth for every point in the interval
            self.data.loc[aortic_idx:diastole_idx, 'diastolic_ratio'] = (interval_distal / interval_aortic).reindex(self.data.loc[aortic_idx:diastole_idx].index).values

            # Calculate iFR as the mean of the interval ratios
            self.data.at[aortic_idx, 'iFR'] = interval_distal.mean() / interval_aortic.mean()

    def calculate_systolic_measures(self):
        """
        Calculates systolic measures and updates the DataFrame.
        """
        df_copy = self.data.copy()

        diastole_indices = df_copy.index[df_copy['peaks'] == 2].tolist()
        aortic_indices = df_copy.index[df_copy['peaks'] == 3].tolist()

        # Ensure diastole occurs before aortic peaks
        if df_copy.loc[diastole_indices[0], 'time'] > df_copy.loc[aortic_indices[0], 'time']:
            aortic_indices.pop(0)
        
        min_len = min(len(diastole_indices), len(aortic_indices))

        for i in range(min_len):
            try:
                diastole_idx = diastole_indices[i]
                aortic_idx = aortic_indices[i]
                
                # Get the data range between diastole and aortic peaks
                interval_aortic = self.data.loc[diastole_idx:aortic_idx, 'p_aortic_smooth']
                interval_distal = self.data.loc[diastole_idx:aortic_idx, 'p_distal_smooth']

                # Skip calculation if intervals are empty or have mismatched lengths
                if interval_aortic.empty or interval_distal.empty or len(interval_aortic) != len(interval_distal):
                    continue

                # Calculate systolic integral as the sum of differences
                # Calculate the baseline value as the mean of the values at diastole_indices[i] and diastole_indices[i+1]
                if i + 1 < len(diastole_indices):
                    baseline_value = (self.data.at[diastole_indices[i], 'p_aortic_smooth'] + self.data.at[diastole_indices[i + 1], 'p_aortic_smooth']) / 2
                    self.data.at[diastole_idx, 'systolic_integral_aortic'] = np.sum(interval_aortic - baseline_value)

                    baseline_value = (self.data.at[diastole_indices[i], 'p_distal_smooth'] + self.data.at[diastole_indices[i + 1], 'p_distal_smooth']) / 2
                    self.data.at[diastole_idx, 'systolic_integral_distal'] = np.sum(interval_distal - baseline_value)

                    self.data.at[diastole_idx, 'systolic_integral_diff'] = self.data.at[diastole_idx, 'systolic_integral_aortic'] - self.data.at[diastole_idx, 'systolic_integral_distal']

                # Remove 25% at each end
                interval_aortic = interval_aortic[int(len(interval_aortic) * 0.25):int(len(interval_aortic) * 0.75)]
                interval_distal = interval_distal[int(len(interval_distal) * 0.25):int(len(interval_distal) * 0.75)]

                # Skip if intervals are empty after trimming
                if interval_aortic.empty or interval_distal.empty:
                    continue

                # Calculate diastolic ratio
                diastolic_ratio = (interval_distal / interval_aortic).reindex(self.data.loc[diastole_idx:aortic_idx].index, fill_value=np.nan)

                # Set calculated values to the DataFrame
                self.data.loc[diastole_idx:aortic_idx, 'aortic_ratio'] = diastolic_ratio.values
                self.data.at[aortic_idx, 'mid_systolic_ratio'] = interval_distal.mean() / interval_aortic.mean()

            except Exception as e:
                print(f"Error processing indices {diastole_idx} to {aortic_idx}: {e}")
                continue

    def save_results(self, output_path):
        """
        Saves the processed data to a CSV file.
        """
        self.data.to_csv(output_path, index=False)
