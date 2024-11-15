from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import butter, filtfilt

ifr_df = pd.read_csv('NARCO_10_eval/narco_10_pressure_rest_1.csv')
# ifr_df = pd.read_csv('NARCO_10_eval/narco_10_pressure_dobu.csv')
# ifr_df = pd.read_csv('NARCO_119_eval/narco_119_pressure_dobu.csv')
# ifr_df = ifr_df.head(500)

ifr_df['p_aortic_smooth'] = ifr_df['p_aortic'].rolling(window=10).mean()
ifr_df['p_distal_smooth'] = ifr_df['p_distal'].rolling(window=10).mean()

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
    - ifr (float): The calculated iFR value.
    """
    # Calculate the mean of the signal
    aortic_indices = ifr_df.index[ifr_df['peaks'] == 3]
    diastole_indices = ifr_df.index[ifr_df['peaks'] == 2]

    # if one of them is longer remove last index
    if len(aortic_indices) > len(diastole_indices):
        aortic_indices = aortic_indices[:-1]
    elif len(diastole_indices) > len(aortic_indices):
        diastole_indices = diastole_indices[:-1]
    
    print(len(aortic_indices), len(diastole_indices))

    # calculate iFR between 3 and 2 peaks by not considering first 20% and last 0.3% of the data
    for i in range(min(len(aortic_indices), len(diastole_indices))):
        aortic_idx = aortic_indices[i]
        diastole_idx = diastole_indices[i]
        
        # Get the data range between systole and diastole
        interval_aortic = ifr_df.loc[aortic_idx:diastole_idx, 'p_aortic_smooth']
        interval_distal = ifr_df.loc[aortic_idx:diastole_idx, 'p_distal_smooth']

        # get the integral of the interval by taking the sum of difference between p_aortic_smooth and p_distal_smooth
        ifr_df.at[diastole_idx, 'diastolic_integral'] = np.sum(interval_aortic - interval_distal)

        # remove 5 percent at each end
        interval_aortic = interval_aortic[int(len(interval_aortic)*0.25):int(len(interval_aortic)*0.90)]
        interval_distal = interval_distal[int(len(interval_distal)*0.25):int(len(interval_distal)*0.90)]

        # calculate additionally diastolic ratio as p_distal_smooth / p_aortic_smooth for every poitn in interval
        ifr_df.loc[aortic_idx:diastole_idx, 'diastolic_ratio'] = (interval_distal / interval_aortic).values

        ifr_df.at[aortic_idx, 'iFR'] = interval_distal.mean() / interval_aortic.mean()

    return ifr_df

def calculate_systolic_measures(ifr_df):
    diastole_indices = ifr_df.index[ifr_df['peaks'] == 2]
    aortic_indices = ifr_df.index[ifr_df['peaks'] == 3]

    # ignore first aortic_indices entry and last diastole_indices entry
    aortic_indices = aortic_indices[1:]
    diastole_indices = diastole_indices[:-1]

    for i in range(min(len(diastole_indices), len(aortic_indices))):
        diastole_idx = diastole_indices[i]
        aortic_idx = aortic_indices[i]
        
        # Get the data range between systole and diastole
        interval_aortic = ifr_df.loc[diastole_idx:aortic_idx, 'p_aortic_smooth']
        interval_distal = ifr_df.loc[diastole_idx:aortic_idx, 'p_distal_smooth']

        # get the integral of the interval by taking the sum of difference between p_aortic_smooth and p_distal_smooth
        ifr_df.at[diastole_idx, 'systolic_integral'] = np.sum(interval_aortic - interval_distal)

        # remove 5 percent at each end
        interval_aortic = interval_aortic[int(len(interval_aortic)*0.25):int(len(interval_aortic)*0.75)]
        interval_distal = interval_distal[int(len(interval_distal)*0.25):int(len(interval_distal)*0.75)]

        # calculate additionally diastolic ratio as p_distal_smooth / p_aortic_smooth for every poitn in interval
        ifr_df.loc[diastole_idx:aortic_idx, 'aortic_ratio'] = (interval_distal / interval_aortic).reindex(ifr_df.loc[diastole_idx:aortic_idx].index).values

        ifr_df.at[aortic_idx, 'mid_systolic_ratio'] = interval_distal.mean() / interval_aortic.mean()

    return ifr_df

ifr_df = refind_peaks(ifr_df, signal='p_aortic_smooth')
ifr_df = find_saddle_point_with_trimmed_interval(ifr_df, signal='p_aortic_smooth')
ifr_df = calculate_ifr(ifr_df)
ifr_df = calculate_systolic_measures(ifr_df)

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


# # plot with original peaks
# # Save the plot as a .png file
# plt.figure(figsize=(10, 6))
# plt.plot(ifr_df['time'], ifr_df['p_aortic_smooth'], label='p_aortic', color='blue')
# plt.plot(ifr_df['time'], ifr_df['p_distal_smooth'], label='p_distal', color='green')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 1], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 1], color='red', label='Systole')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 2], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 2], color='blue', label='Diastole')
# plt.xlabel('Time')
# plt.ylabel('Pressure')
# plt.title('Original p_aortic and p_distal over Time')
# plt.legend()
# plt.savefig("original_pressure_plot.png")  # Save plot to a file
# plt.close()  # Close the plot to free up memory

# # plot with original peaks
# # Save the plot as a .png file
# plt.figure(figsize=(10, 6))
# plt.plot(ifr_df['time'], ifr_df['p_aortic_smooth'], label='p_aortic', color='blue')
# plt.plot(ifr_df['time'], ifr_df['p_distal_smooth'], label='p_distal', color='green')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 1], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 1], color='red', label='Systole')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 2], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 2], color='blue', label='Diastole')
# plt.xlabel('Time')
# plt.ylabel('Pressure')
# plt.title('Original p_aortic and p_distal over Time')
# plt.legend()
# plt.savefig("redefined_pressure_plot.png")  # Save plot to a file
# plt.close()  # Close the plot to free up memory

# # plot with original peaks
# # Save the plot as a .png file
# plt.figure(figsize=(10, 6))
# plt.plot(ifr_df['time'], ifr_df['p_aortic_smooth'], label='p_aortic', color='blue')
# plt.plot(ifr_df['time'], ifr_df['p_distal_smooth'], label='p_distal', color='green')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 1], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 1], color='red', label='Systole')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 2], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 2], color='blue', label='Diastole')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 3], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 3], color='orange', label='Closure')
# plt.xlabel('Time')
# plt.ylabel('Pressure')
# plt.title('Original p_aortic and p_distal over Time')
# plt.legend()
# plt.savefig("saddle_pressure_plot.png")  # Save plot to a file
# plt.close()  # Close the plot to free up memory

# # plot iFR over time
# # Save the plot as a .png file
# plt.figure(figsize=(10, 6))
# plt.plot(ifr_df['time'], ifr_df['p_aortic_smooth'], label='p_aortic', color='blue')
# plt.plot(ifr_df['time'], ifr_df['p_distal_smooth'], label='p_distal', color='green')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 1], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 1], color='red', label='Systole')
# plt.scatter(ifr_df['time'][ifr_df['peaks'] == 2], ifr_df['p_aortic_smooth'][ifr_df['peaks'] == 2], color='blue', label='Diastole')
# # plot iFR * 100
# plt.scatter(ifr_df['time'][~ifr_df['iFR'].isna()], ifr_df['iFR'][~ifr_df['iFR'].isna()] * 100, label='iFR', color='orange')
# plt.xlabel('Time')
# plt.ylabel('Pressure')
# plt.title('Original p_aortic and p_distal over Time')
# plt.legend()
# plt.savefig("ifr_plot.png")  # Save plot to a file
# plt.close()  # Close the plot to free up memory
