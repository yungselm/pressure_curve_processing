from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import butter, filtfilt

# ifr_df = pd.read_csv('NARCO_10_eval/narco_10_pressure_rest_1.csv')
ifr_df = pd.read_csv('NARCO_10_eval/narco_10_pressure_dobu.csv')
# ifr_df = pd.read_csv('NARCO_119_eval/narco_119_pressure_dobu.csv')
# ifr_df = pd.read_csv('NARCO_10_eval/narco_10_pressure_ade.csv')
# ifr_df = ifr_df.head(2000)

ifr_df['p_aortic_smooth'] = ifr_df['p_aortic'].rolling(window=10).mean()
ifr_df['p_distal_smooth'] = ifr_df['p_distal'].rolling(window=10).mean()

def preprocess_data(data):
    """
    Checks for the longest segment where peaks == 1 and 2 alternate, and keeps only that segment.
    Peaks can have values of 1, 2, or 0 (non-peaks), and only alternating 1 and 2 are considered valid.
    """
    df_copy = data.copy()
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
    
    # Keep only the rows corresponding to the longest alternating segment, including peaks == 0
    start_idx = longest_segment_indices[0]
    end_idx = longest_segment_indices[-1]
    data = df_copy.iloc[start_idx:end_idx + 1].reset_index(drop=True)

    return data


def refind_peaks(df, signal='p_aortic_smooth'):
    """
    Refines systolic and diastolic peaks by processing in batches between existing systolic peaks
    (peaks == 1) or diastolic peaks (peaks == 2), and resetting peaks consistently.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - signal (str): The column name of the input signal to analyze.

    Returns:
    - df (pd.DataFrame): The updated DataFrame with refined systolic and diastolic peaks.
    """
    # Work on a copy to avoid interfering with original peaks during processing
    df_copy_systole = df.copy()
    df_copy_systole.loc[df_copy_systole['peaks'] == 1, 'peaks'] = 0
    df_copy_diastole = df.copy()
    df_copy_diastole.loc[df_copy_diastole['peaks'] == 2, 'peaks'] = 0

    # find maximum between 'peaks' == 2 (diastole) for systolic peak
    # find minimum between 'peaks' == 1 (systole) for diastolic peak
    systole_indices = df_copy_systole.index[df['peaks'] == 2]

    for i in range(len(systole_indices) - 1):
        start_idx = systole_indices[i]
        end_idx = systole_indices[i + 1]

        # Get the data range between systole and diastole
        interval_data = df_copy_systole.loc[start_idx:end_idx, signal]

        # Detect the peak in the interval data
        peak_idx = interval_data.idxmax()

        # Mark the peak as '1' in the modified peaks column
        df_copy_systole.at[peak_idx, 'peaks'] = 1
    
    diastole_indices = df_copy_diastole.index[df['peaks'] == 1]

    for i in range(len(diastole_indices) - 1):
        start_idx = diastole_indices[i]
        end_idx = diastole_indices[i + 1]

        # Get the data range between systole and diastole
        interval_data = df_copy_diastole.loc[start_idx:end_idx, signal]

        # Detect the peak in the interval data
        peak_idx = interval_data.idxmin()

        # Mark the peak as '2' in the modified peaks column
        df_copy_diastole.at[peak_idx, 'peaks'] = 2
    
    # Combine the refined systolic and diastolic peaks
    df['peaks'] = 0
    df['peaks'] = df['peaks'].mask(df_copy_systole['peaks'] == 1, 1)
    df['peaks'] = df['peaks'].mask(df_copy_diastole['peaks'] == 2, 2)

    return df

def find_saddle_point_with_trimmed_interval(df, signal='p_aortic_smooth'):
    """
    Finds the saddle points between systolic and diastolic peaks by removing 
    the first and last 10% of the data before detecting local minima (saddle points).
    If no saddle point is found, places it in the midpoint of the systolic and diastolic peaks.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the signal and peaks.
    - signal (str): The column name of the input signal to analyze.

    Returns:
    - df (pd.DataFrame): The updated DataFrame with identified saddle points.
    """
    # Work on a copy to avoid modifying original peaks during processing
    df_copy = df.copy()

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
        interval_data = interval_data[int(len(interval_data)*0.10):int(len(interval_data)*0.90)]
        interval_time = interval_time[int(len(interval_time)*0.10):int(len(interval_time)*0.90)]

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

    return df_copy


def calculate_ifr(ifr_df):
    """
    Calculates the instantaneous wave-free ratio (iFR) from the input signal.

    Parameters:
    - signal (np.ndarray): The input signal to calculate iFR from.

    Returns:
    - ifr_df (pd.DataFrame): The updated DataFrame with calculated iFR values.
    """
    df_copy = ifr_df.copy()

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
        interval_aortic = ifr_df.loc[aortic_idx:diastole_idx, 'p_aortic_smooth']
        interval_distal = ifr_df.loc[aortic_idx:diastole_idx, 'p_distal_smooth']

        # Calculate systolic integral as the sum of the difference between p_aortic_smooth and p_distal_smooth
        # Calculate the baseline value as the mean of the values at diastole_indices[i] and diastole_indices[i+1]
        if i + 1 < len(diastole_indices):
            baseline_value = (ifr_df.at[diastole_indices[i], 'p_aortic_smooth'] + ifr_df.at[diastole_indices[i + 1], 'p_aortic_smooth']) / 2
            ifr_df.at[diastole_idx, 'diastolic_integral_aortic'] = np.sum(interval_aortic - baseline_value)

            baseline_value = (ifr_df.at[diastole_indices[i], 'p_distal_smooth'] + ifr_df.at[diastole_indices[i + 1], 'p_distal_smooth']) / 2
            ifr_df.at[diastole_idx, 'diastolic_integral_distal'] = np.sum(interval_distal - baseline_value)

            ifr_df.at[diastole_idx, 'diastolic_integral_diff'] = ifr_df.at[diastole_idx, 'diastolic_integral_aortic'] - ifr_df.at[diastole_idx, 'diastolic_integral_distal']

        # Remove 25% at each end and 10% from the other end (keeping 90% of the middle part)
        interval_aortic = interval_aortic[int(len(interval_aortic) * 0.25):int(len(interval_aortic) * 0.90)]
        interval_distal = interval_distal[int(len(interval_distal) * 0.25):int(len(interval_distal) * 0.90)]

        # Calculate the diastolic ratio as p_distal_smooth / p_aortic_smooth for every point in the interval
        ifr_df.loc[aortic_idx:diastole_idx, 'diastolic_ratio'] = (interval_distal / interval_aortic).reindex(ifr_df.loc[aortic_idx:diastole_idx].index).values

        # Calculate iFR as the mean of the interval ratios
        ifr_df.at[aortic_idx, 'iFR'] = interval_distal.mean() / interval_aortic.mean()

    return ifr_df


def calculate_systolic_measures(ifr_df):
    df_copy = ifr_df.copy()

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
            interval_aortic = ifr_df.loc[diastole_idx:aortic_idx, 'p_aortic_smooth']
            interval_distal = ifr_df.loc[diastole_idx:aortic_idx, 'p_distal_smooth']

            # Skip calculation if intervals are empty or have mismatched lengths
            if interval_aortic.empty or interval_distal.empty or len(interval_aortic) != len(interval_distal):
                continue

            # Calculate systolic integral as the sum of differences
            # Calculate the baseline value as the mean of the values at diastole_indices[i] and diastole_indices[i+1]
            if i + 1 < len(diastole_indices):
                baseline_value = (ifr_df.at[diastole_indices[i], 'p_aortic_smooth'] + ifr_df.at[diastole_indices[i + 1], 'p_aortic_smooth']) / 2
                ifr_df.at[diastole_idx, 'systolic_integral_aortic'] = np.sum(interval_aortic - baseline_value)

                baseline_value = (ifr_df.at[diastole_indices[i], 'p_distal_smooth'] + ifr_df.at[diastole_indices[i + 1], 'p_distal_smooth']) / 2
                ifr_df.at[diastole_idx, 'systolic_integral_distal'] = np.sum(interval_distal - baseline_value)

                ifr_df.at[diastole_idx, 'systolic_integral_diff'] = ifr_df.at[diastole_idx, 'systolic_integral_aortic'] - ifr_df.at[diastole_idx, 'systolic_integral_distal']

            # Remove 25% at each end
            interval_aortic = interval_aortic[int(len(interval_aortic) * 0.25):int(len(interval_aortic) * 0.75)]
            interval_distal = interval_distal[int(len(interval_distal) * 0.25):int(len(interval_distal) * 0.75)]

            # Skip if intervals are empty after trimming
            if interval_aortic.empty or interval_distal.empty:
                continue

            # Calculate diastolic ratio
            diastolic_ratio = (interval_distal / interval_aortic).reindex(ifr_df.loc[diastole_idx:aortic_idx].index, fill_value=np.nan)

            # Set calculated values to the DataFrame
            ifr_df.loc[diastole_idx:aortic_idx, 'aortic_ratio'] = diastolic_ratio.values
            ifr_df.at[aortic_idx, 'mid_systolic_ratio'] = interval_distal.mean() / interval_aortic.mean()

        except Exception as e:
            print(f"Error processing indices {diastole_idx} to {aortic_idx}: {e}")
            continue

    return ifr_df


def get_average_curve_between_diastolic_peaks(ifr_df, signal='p_aortic_smooth', num_points=100):
    """
    Computes the average curve between diastolic peaks by scaling each interval to the same time length.

    Parameters:
    - ifr_df (pd.DataFrame): The DataFrame containing the signal and peak information.
    - signal (str): The column name of the input signal to analyze.
    - num_points (int): The number of points to normalize each interval to.

    Returns:
    - avg_curve (np.ndarray): The average curve of the signal.
    - avg_time (np.ndarray): The normalized time axis corresponding to the average curve.
    """
    # Extract indices of diastolic peaks
    diastolic_indices = ifr_df.index[ifr_df['peaks'] == 2].tolist()

    if len(diastolic_indices) < 2:
        raise ValueError("Not enough diastolic peaks to calculate intervals.")

    # Initialize a list to store all rescaled intervals
    rescaled_curves = []

    for i in range(len(diastolic_indices) - 1):
        start_idx = diastolic_indices[i]
        end_idx = diastolic_indices[i + 1]

        # Extract the interval data
        interval_data = ifr_df.loc[start_idx:end_idx, signal].values

        if len(interval_data) < 2:
            # Skip intervals with insufficient data
            continue

        # Rescale the interval to have `num_points` using interpolation
        original_time = np.linspace(0, 1, len(interval_data))
        scaled_time = np.linspace(0, 1, num_points)
        rescaled_curve = np.interp(scaled_time, original_time, interval_data)

        rescaled_curves.append(rescaled_curve)

    if len(rescaled_curves) == 0:
        raise ValueError("No valid intervals found for averaging.")

    # Convert the list of rescaled curves to a 2D NumPy array
    rescaled_curves = np.array(rescaled_curves)

    # Compute the average curve across all rescaled intervals
    avg_curve = np.mean(rescaled_curves, axis=0)
    avg_time = np.linspace(0, 1, num_points)

    return avg_time, avg_curve


def split_df_by_pdpa(data):
    """
    Splits the input DataFrame into two separate DataFrames based on the pd/pa ratio.
    """
    df_copy = data.copy()

    # get lower 25% of pd/pa ratio
    lower_bound = df_copy['pd/pa'].quantile(0.25)
    upper_bound = df_copy['pd/pa'].quantile(0.75)

    df_low = df_copy[df_copy['pd/pa'] < lower_bound]
    df_high = df_copy[df_copy['pd/pa'] > upper_bound]
    
    return df_low, df_high

def get_measurements(data):
    """
    Extracts iFR, mid_systolic_ratio and calculates mean, plus mean of pd/pa ratio.
    For diastolic_ratio and aortic_ratio, gets their start and end time within the cardiac cycle.
    """
    df_copy = data.copy()

    # Get the mean of iFR, mid_systolic_ratio, and pd/pa
    iFR_mean = df_copy['iFR'].mean()
    mid_systolic_ratio_mean = df_copy['mid_systolic_ratio'].mean()
    pdpa_mean = df_copy['pd/pa'].mean()

    return iFR_mean, mid_systolic_ratio_mean, pdpa_mean


def plot_average_curve(data, name='all'):
    """
    Plots the average curve between diastolic peaks for p_aortic_smooth and p_distal_smooth.
    """
    if name == 'all':
        avg_time, avg_curve_aortic = get_average_curve_between_diastolic_peaks(data, signal='p_aortic_smooth', num_points=100)
        avg_time, avg_curve_distal = get_average_curve_between_diastolic_peaks(data, signal='p_distal_smooth', num_points=100)
        file_name = 'average_curve_plot_all.png'
    elif name == 'lower':
        avg_time, avg_curve_aortic = get_average_curve_between_diastolic_peaks(data, signal='p_aortic_smooth', num_points=100)
        avg_time, avg_curve_distal = get_average_curve_between_diastolic_peaks(data, signal='p_distal_smooth', num_points=100)
        file_name = 'average_curve_plot_lower.png'
    elif name == 'high':
        avg_time, avg_curve_aortic = get_average_curve_between_diastolic_peaks(data, signal='p_aortic_smooth', num_points=100)
        avg_time, avg_curve_distal = get_average_curve_between_diastolic_peaks(data, signal='p_distal_smooth', num_points=100)
        file_name = 'average_curve_plot_high.png'
    else:
        raise ValueError("Invalid name. Choose from 'all', 'lower', or 'high'.")
    
    # get all measurements
    iFR_mean, mid_systolic_ratio_mean, pdpa_mean = get_measurements(data)

    # add start and end time of aortic_ratio and diastolic_ratio to the plot as vertical lines, and add ifr, mid_systolic_ratio and pd_pa as text
    plt.figure(figsize=(10, 6))
    plt.plot(avg_time, avg_curve_aortic, label='p_aortic_smooth', color='blue')
    plt.plot(avg_time, avg_curve_distal, label='p_distal_smooth', color='green')
    plt.axvline(x=0.25, color='red', linestyle='--')
    plt.axvline(x=0.4, color='red', linestyle='--')
    plt.axvline(x=0.6, color='blue', linestyle='--')
    plt.axvline(x=0.97, color='blue', linestyle='--')
    plt.text(0.5, 0.9, f'iFR: {iFR_mean:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.85, f'mid_systolic_ratio: {mid_systolic_ratio_mean:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.8, f'pd/pa: {pdpa_mean:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.title(f'Average Curve between Diastolic Peaks ({name})')
    plt.legend()
    plt.savefig(file_name)  # Save plot to a file
    plt.close()  # Close the plot to free up memory

print(len(ifr_df))
ifr_df = preprocess_data(ifr_df)
print(len(ifr_df))
ifr_df = refind_peaks(ifr_df, signal='p_aortic_smooth')
ifr_df = find_saddle_point_with_trimmed_interval(ifr_df, signal='p_aortic_smooth')
ifr_df = calculate_ifr(ifr_df)
ifr_df = calculate_systolic_measures(ifr_df)

# create average curve between diastolic peaks for p_aortic_smooth and p_distal_smooth and plto it
avg_time, avg_curve_aortic = get_average_curve_between_diastolic_peaks(ifr_df, signal='p_aortic_smooth', num_points=100)
avg_time, avg_curve_distal = get_average_curve_between_diastolic_peaks(ifr_df, signal='p_distal_smooth', num_points=100)

low_df, high_df = split_df_by_pdpa(ifr_df)

avg_time_low, avg_curve_aortic_low = get_average_curve_between_diastolic_peaks(low_df, signal='p_aortic_smooth', num_points=100)
avg_time_low, avg_curve_distal_low = get_average_curve_between_diastolic_peaks(low_df, signal='p_distal_smooth', num_points=100)

avg_time_high, avg_curve_aortic_high = get_average_curve_between_diastolic_peaks(high_df, signal='p_aortic_smooth', num_points=100)
avg_time_high, avg_curve_distal_high = get_average_curve_between_diastolic_peaks(high_df, signal='p_distal_smooth', num_points=100)

plot_average_curve(ifr_df, name='all')
plot_average_curve(low_df, name='lower')
plot_average_curve(high_df, name='high')

# Save the DataFrame to a .csv file
ifr_df.to_csv('narco_119_pressure_dobu_iFR.csv', index=False)

# keep only lines with either iFR or mid_systolic_ratio not NaN
ifr_df = ifr_df[~(ifr_df['iFR'].isna() & ifr_df['mid_systolic_ratio'].isna())]

# plot iFR, mid_systolic_ratio and pd/pa over time
# Save the plot as a .png file
plt.figure(figsize=(10, 6))
plt.plot(ifr_df['time'], ifr_df['iFR'], label='iFR', color='blue')
plt.plot(ifr_df['time'], ifr_df['mid_systolic_ratio'], label='mid_systolic_ratio', color='green')
plt.plot(ifr_df['time'], ifr_df['pd/pa'], label='pd/pa', color='red')
# plot horizontal line at 0.8
plt.axhline(y=0.8, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('iFR, mid_systolic_ratio and pd/pa over Time')
plt.legend()
plt.savefig("ifr_plot.png")  # Save plot to a file
plt.close()  # Close the plot to free up memory

plt.figure(figsize=(10, 6))
plt.plot(avg_time, avg_curve_aortic, label='p_aortic_smooth', color='blue')
plt.plot(avg_time, avg_curve_distal, label='p_distal_smooth', color='green')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Average Curve between Diastolic Peaks')
plt.legend()
plt.savefig("average_curve_plot.png")  # Save plot to a file
plt.close()  # Close the plot to free up memory
