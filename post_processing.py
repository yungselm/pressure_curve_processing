import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class PostProcessing:
    def __init__(self, output_path):
        self.ifr_df = pd.DataFrame()
        # initialize an empty dataframe with predefined columns
        self.result_df = pd.DataFrame(
            columns=[
                'patient_id',
                'iFR_mean_rest',
                'mid_systolic_ratio_mean_rest',
                'pdpa_mean_rest',
                'iFR_mean_ado',
                'mid_systolic_ratio_mean_ado',
                'pdpa_mean_ado',
                'iFR_mean_dobu',
                'mid_systolic_ratio_mean_dobu',
                'pdpa_mean_dobu',
            ]
        )
        self.output_dir = output_path
        # write empty dataframe to an excel file with the output path
        self.result_df.to_excel(self.output_dir, index=False)

    def __call__(self, *args, **kwds):
        pass


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
        avg_time, avg_curve_aortic = get_average_curve_between_diastolic_peaks(
            data, signal='p_aortic_smooth', num_points=100
        )
        avg_time, avg_curve_distal = get_average_curve_between_diastolic_peaks(
            data, signal='p_distal_smooth', num_points=100
        )
        file_name = 'average_curve_plot_all.png'
    elif name == 'lower':
        avg_time, avg_curve_aortic = get_average_curve_between_diastolic_peaks(
            data, signal='p_aortic_smooth', num_points=100
        )
        avg_time, avg_curve_distal = get_average_curve_between_diastolic_peaks(
            data, signal='p_distal_smooth', num_points=100
        )
        file_name = 'average_curve_plot_lower.png'
    elif name == 'high':
        avg_time, avg_curve_aortic = get_average_curve_between_diastolic_peaks(
            data, signal='p_aortic_smooth', num_points=100
        )
        avg_time, avg_curve_distal = get_average_curve_between_diastolic_peaks(
            data, signal='p_distal_smooth', num_points=100
        )
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
    plt.text(
        0.5,
        0.9,
        f'iFR: {iFR_mean:.2f}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.5,
        0.85,
        f'mid_systolic_ratio: {mid_systolic_ratio_mean:.2f}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.5,
        0.8,
        f'pd/pa: {pdpa_mean:.2f}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=plt.gca().transAxes,
    )
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.title(f'Average Curve between Diastolic Peaks ({name})')
    plt.legend()
    plt.savefig(file_name)  # Save plot to a file
    plt.close()  # Close the plot to free up memory
