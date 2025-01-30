import sys
import os
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import tqdm
from loguru import logger
from statsmodels.nonparametric.smoothers_lowess import lowess
from utils import patient_id_from_dir


class IvusProcessor:
    def __init__(self, rest_dir, stress_dir):
        self.rest_dir = rest_dir
        self.stress_dir = stress_dir
        self.name_dir = patient_id_from_dir(rest_dir)
        self.rest_dir_rearranged = None
        self.stress_dir_rearranged = None

        self.rest_df = self.create_empty_df(rest_dir, 'loess_data_rest.csv')
        self.stress_df = self.create_empty_df(stress_dir, 'loess_data_stress.csv')
        self.rest_df_glob = self.create_empty_df(rest_dir, 'loess_data_rest_glob.csv')
        self.stress_df_glob = self.create_empty_df(stress_dir, 'loess_data_stress_glob.csv')

        self.phase_order = 10
        self.global_order = 4
        self.estimate_distance_true = 0  # if one the distance is estimated by hr, more error prone but works for faulty metadata with late pullback start
        self.PULLBACK_SPEED = 1
        self.START_FRAME = 0
        self.FRAME_RATE = 30
        self.flag = 1
        self.current_dir = 'rest'

    def create_empty_df(self, directory, name):
        """Create an empty DataFrame with these columns: position, lumen_area, lumen_area_ci_lower,
        lumen_area_ci_upper, elliptic_ratio, elliptic_ratio_ci_lower, elliptic_ratio_ci_upper, phase, distance"""
        df = pd.DataFrame(
            columns=[
                'position',
                'lumen_area',
                'lumen_area_ci_lower',
                'lumen_area_ci_upper',
                'elliptic_ratio',
                'elliptic_ratio_ci_lower',
                'elliptic_ratio_ci_upper',
                'shortest_distance',
                'shortest_ci_lower',
                'shortest_ci_upper',
                'phase',
            ]
        )

        df.to_csv(os.path.join(os.path.dirname(directory), name), index=False)

    def get_order_and_estimate(self, ):
        try:
            order_list = pd.read_excel(
                "../data/00_polynomial_fit_all.xlsx"
            )
            if order_list.empty:
                raise ValueError("Excel file is empty or does not have the required data.")
            matching_row = order_list[order_list['narco_id'] == self.name_dir]
            if matching_row.empty:
                raise ValueError(f"No matching narco_id found for {self.name_dir}.")

            self.phase_order = matching_row['phase_order'].values[0]
            self.global_order = matching_row['global_order'].values[0]
            self.estimate_distance_true = matching_row['estimate_distance_true'].values[0]

        except Exception as e:
            logger.error(f"Error reading polynomial fit orders or estimate of distance: {e}. Using default values.")

    def get_global_variables(self):
        """
        Calculates the global variables for the lumen area and elliptic ratio.
        """
        # in the directory read file ending with _report
        if self.current_dir == 'rest':
            report_file = [f for f in os.listdir(self.rest_dir) if f.endswith('_report.txt')][0]
        else:
            report_file = [f for f in os.listdir(self.stress_dir) if f.endswith('_report.txt')][0]

        df = pd.read_csv(
            os.path.join(self.rest_dir if self.current_dir == 'rest' else self.stress_dir, report_file), sep='\t'
        )

        if df is None or df.empty or pd.isna(df['pullback_speed'].unique()[0]):
            raise ValueError("The dataframe is empty or not initialized. Please provide a valid dataframe.")

        self.PULLBACK_SPEED = df['pullback_speed'].unique()[0]
        self.START_FRAME = df['pullback_start_frame'].unique()[0]
        self.FRAME_RATE = df['frame_rate'].unique()[0]

    def estimate_distance(self, df):
        """
        Estimates the distance between consecutive frames during a pullback.
        """
        if 'frame' not in df or len(df) < 2:
            raise ValueError("DataFrame must contain a 'frame' column with at least two frames.")

        # Calculate the time interval between consecutive frames
        frame_diffs = np.diff(df['frame'].sort_values())
        time_intervals = frame_diffs / self.FRAME_RATE  # Time interval in seconds

        # Calculate distance for each interval based on pullback speed (mm/s)
        distances = time_intervals * self.PULLBACK_SPEED

        # Generate cumulative distance, starting at 0
        cumulative_distance = np.cumsum(np.insert(distances, 0, 0))

        return cumulative_distance

    def prep_data(self, df):
        """
        Prepare data by filtering and calculating distances.
        Uses the global variables for processing.
        """
        try:
            self.get_global_variables()
        except ValueError as e:
            print(e)
            print("Using default values for pullback speed, start frame and frame rate.")

        df_dia = df[df['phase'] == 'D'].copy()
        df_sys = df[df['phase'] == 'S'].copy()

        if self.estimate_distance_true == 1:
            df_dia['distance'] = self.estimate_distance(df_dia)
            df_sys['distance'] = self.estimate_distance(df_sys)
        else:
            df_dia['distance'] = df_dia['position'].max() - df_dia['position']
            df_sys['distance'] = df_sys['position'].max() - df_sys['position']

        df_dia['distance'] = df_dia['distance'].values[::-1]
        df_sys['distance'] = df_sys['distance'].values[::-1]
        df_dia = df_dia[::-1]
        df_sys = df_sys[::-1]

        return pd.concat([df_dia, df_sys])

    def loess_fit(self, x, y, frac=0.3, ci=True, phase='D', global_fit=False):
        # store name of y
        y_name = y.name
        #  Data
        x = x.to_numpy()
        y = y.to_numpy()

        # Remove NaN or infinite values
        valid_indices = np.isfinite(x) & np.isfinite(y)
        x = x[valid_indices]
        y = y[valid_indices]

        # Perform LOESS smoothing
        smoothed = lowess(y, x, frac=frac, return_sorted=True)
        x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]

        # Bootstrapping for Confidence Intervals
        n_boot = 100  # Number of bootstrap samples
        boot_preds = []

        if ci:
            for _ in tqdm.tqdm(
                    range(n_boot), desc=f"Bootstrapping for {self.name_dir.split(os.sep)[-1]} variable {y_name}"
            ):
                # Resample with replacement
                indices = np.random.choice(len(x), len(x), replace=True)
                x_boot, y_boot = x[indices], y[indices]

                # Fit LOESS to resampled data
                smoothed_boot = lowess(y_boot, x_boot, frac=frac, return_sorted=True)
                boot_preds.append(np.interp(x_smooth, smoothed_boot[:, 0], smoothed_boot[:, 1]))

            boot_preds = np.array(boot_preds)

            # Calculate confidence intervals (e.g., 95%)
            ci_lower = np.percentile(boot_preds, 2.5, axis=0)
            ci_upper = np.percentile(boot_preds, 97.5, axis=0)

        else:
            ci_lower = y_smooth
            ci_upper = y_smooth

        if self.flag == 1:
            x_new = np.arange(0, 20, 0.2)
            y_new = np.interp(x_new, x_smooth, y_smooth)
            ci_lower_new = np.interp(x_new, x_smooth, ci_lower)
            ci_upper_new = np.interp(x_new, x_smooth, ci_upper)
            # save output to a new dataframe in the same directory
            df_out = pd.DataFrame(
                {
                    'position': x_new,
                    'lumen_area': np.nan,
                    'lumen_area_ci_lower': np.nan,
                    'lumen_area_ci_upper': np.nan,
                    'elliptic_ratio': np.nan,
                    'elliptic_ratio_ci_lower': np.nan,
                    'elliptic_ratio_ci_upper': np.nan,
                    'shortest_distance': np.nan,
                    'shortest_ci_lower': np.nan,
                    'shortest_ci_upper': np.nan,
                    'phase': phase,
                }
            )
            if y_name == 'lumen_area':
                df_out['lumen_area'] = y_new
                df_out['lumen_area_ci_lower'] = ci_lower_new
                df_out['lumen_area_ci_upper'] = ci_upper_new
            elif y_name == 'mean_elliptic_ratio':
                df_out['elliptic_ratio'] = y_new
                df_out['elliptic_ratio_ci_lower'] = ci_lower_new
                df_out['elliptic_ratio_ci_upper'] = ci_upper_new
            elif y_name == 'shortest_distance':
                df_out['shortest_distance'] = y_new
                df_out['shortest_ci_lower'] = ci_lower_new
                df_out['shortest_ci_upper'] = ci_upper_new
            else:
                logger.info("Unknown variable name")

            if self.current_dir == 'rest' and not global_fit:
                with open(os.path.join(os.path.dirname(self.rest_dir), 'loess_data_rest.csv'), 'a', newline='') as f:
                    df_out.to_csv(f, header=False, index=False)

            elif self.current_dir == 'stress' and not global_fit:
                with open(
                        os.path.join(os.path.dirname(self.stress_dir), 'loess_data_stress.csv'), 'a', newline=''
                ) as f:
                    df_out.to_csv(f, header=False, index=False)
            elif self.current_dir == 'rest' and global_fit:
                with open(
                        os.path.join(os.path.dirname(self.rest_dir), 'loess_data_rest_glob.csv'), 'a', newline=''
                ) as f:
                    df_out.to_csv(f, header=False, index=False)
            elif self.current_dir == 'stress' and global_fit:
                with open(
                        os.path.join(os.path.dirname(self.stress_dir), 'loess_data_stress_glob.csv'), 'a', newline=''
                ) as f:
                    df_out.to_csv(f, header=False, index=False)
            else:
                logger.info("Current directory is not set to rest or stress")

        return y_smooth, ci_lower, ci_upper

    def fit_curves_sys_dia(self, df):
        """Fits a polynomial curve to the lumen area and elliptic ratio."""
        if 'phase' not in df.columns:
            raise KeyError("The 'phase' column is missing from the DataFrame.")

        # Ensure distance is sorted within each phase
        fitted_results = (
            df.groupby('phase')
            .apply(
                lambda group: pd.DataFrame(
                    {
                        'fitted_lumen_area': self.loess_fit(
                            group['distance'],
                            group['lumen_area'],
                            phase=group['phase'].iloc[0],
                        )[0],
                        'area_ci_lower': self.loess_fit(
                            group['distance'],
                            group['lumen_area'],
                            phase=group['phase'].iloc[0],
                        )[1],
                        'area_ci_upper': self.loess_fit(
                            group['distance'],
                            group['lumen_area'],
                            phase=group['phase'].iloc[0],
                        )[2],
                    },
                    index=group.index,
                )
            )
            .reset_index(drop=True)
        )

        df = df.reset_index(drop=True)
        df = pd.concat([df, fitted_results], axis=1)

        # Ensure sorted and consistent distance for mean elliptic ratio
        df['mean_elliptic_ratio'] = df.groupby('distance')['elliptic_ratio'].transform('mean')
        df = df.sort_values('distance').reset_index(drop=True)  # Sort overall for consistency and reset index
        fitted_elliptic_results = pd.DataFrame(
            {
                'fitted_elliptic_ratio': self.loess_fit(df['distance'], df['mean_elliptic_ratio'])[0],
                'elliptic_ci_lower': self.loess_fit(df['distance'], df['mean_elliptic_ratio'])[1],
                'elliptic_ci_upper': self.loess_fit(df['distance'], df['mean_elliptic_ratio'])[2],
            }
        ).reset_index(drop=True)

        df = pd.concat([df, fitted_elliptic_results], axis=1)

        # Apply similar fixes for shortest distance
        fitted_shortest_results = (
            df.groupby('phase')
            .apply(
                lambda group: pd.DataFrame(
                    {
                        'fitted_shortest_distance': self.loess_fit(
                            group.sort_values('distance')['distance'],
                            group.sort_values('distance')['shortest_distance'],
                            phase=group['phase'].iloc[0],
                        )[0],
                        'shortest_ci_lower': self.loess_fit(
                            group.sort_values('distance')['distance'],
                            group.sort_values('distance')['shortest_distance'],
                            phase=group['phase'].iloc[0],
                        )[1],
                        'shortest_ci_upper': self.loess_fit(
                            group.sort_values('distance')['distance'],
                            group.sort_values('distance')['shortest_distance'],
                            phase=group['phase'].iloc[0],
                        )[2],
                    },
                    index=group.index,
                )
            )
            .reset_index(drop=True)
        )

        df = pd.concat([df, fitted_shortest_results], axis=1)
        return df.sort_values(by='distance').reset_index(drop=True)  # Final sort and reset index

    def fit_curve_global(self, df):
        """Fits a polynomial curve to the lumen area and elliptic ratio."""
        df['fitted_lumen_area_glob'], df['area_ci_lower_glob'], df['area_ci_upper_glob'] = self.loess_fit(
            df['distance'], df['lumen_area'], phase=df['phase'].iloc[0], global_fit=True
        )
        df['mean_elliptic_ratio_glob'] = df['elliptic_ratio'].mean()
        df['fitted_elliptic_ratio_glob'], df['elliptic_ci_lower_glob'], df['elliptic_ci_upper_glob'] = self.loess_fit(
            df['distance'], df['mean_elliptic_ratio'], phase=df['phase'].iloc[0], global_fit=True
        )
        df['fitted_shortest_distance_glob'], df['shortest_ci_lower_glob'], df['shortest_ci_upper_glob'] = (
            self.loess_fit(df['distance'], df['shortest_distance'], phase=df['phase'].iloc[0], global_fit=True)
        )
        df = df.sort_values(by='distance')
        return df

    def ensure_directory_exists(self, path):
        """Ensure the directory for the given path exists."""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_data(self, df_rest, df_stress, variable='lumen_area'):
        """Plot systole and diastole data for rest and stress."""
        # Determine the common x-axis limits based on the dataframe with more x-values
        if variable == 'lumen_area':
            norm = 'lumen_area'
            fitted = 'fitted_lumen_area'
            ylabel = 'Lumen Area (mm²) and Elliptic Ratio'
            title = 'Lumen Area vs Distance'
        elif variable == 'shortest_distance':
            norm = 'shortest_distance'
            fitted = 'fitted_shortest_distance'
            ylabel = 'Shortest Distance (mm) and Elliptic Ratio'
            title = 'Shortest Distance vs Distance'
        else:
            raise ValueError(f"Variable {variable} not supported.")

        x_min = min([df_rest['distance'].min(), df_stress['distance'].min()])
        x_max = max([df_rest['distance'].max(), df_stress['distance'].max()])

        # Determine the common y-axis limits based on the dataframe with more y-values
        y_min_lumen = min(df_rest[norm].min(), df_stress[norm].min())
        y_max_lumen = max(df_rest[norm].max(), df_stress[norm].max())
        y_min_elliptic = min(df_rest['mean_elliptic_ratio'].min(), df_stress['mean_elliptic_ratio'].min())
        y_max_elliptic = max(df_rest['mean_elliptic_ratio'].max(), df_stress['mean_elliptic_ratio'].max())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.4})

        # Plot lumen area for rest
        for phase, group in df_rest.groupby('phase'):
            ax1.plot(group['distance'], group[fitted], label=f'Rest - {phase}')
            ax1.scatter(group['distance'], group[norm], alpha=0.3)
            ax1.fill_between(group['distance'], group['area_ci_lower'], group['area_ci_upper'], color='blue', alpha=0.2)

        # Highlight the smallest lumen area for rest
        min_lumen_area_rest = df_rest[norm].min()
        min_lumen_area_frame_rest = df_rest.loc[df_rest[norm].idxmin(), 'frame']
        min_lumen_area_distance_rest = df_rest.loc[df_rest[norm].idxmin(), 'distance']

        ax1.scatter(min_lumen_area_distance_rest, min_lumen_area_rest, color='#0055ff', zorder=5)
        ax1.text(
            min_lumen_area_distance_rest,
            min_lumen_area_rest,
            f'{min_lumen_area_rest:.2f} ({min_lumen_area_frame_rest})',
            color='#0055ff',
        )

        # Overlay mean elliptic ratio for rest
        ax1.plot(
            df_rest['distance'], df_rest['fitted_elliptic_ratio'], color='darkblue', label='Mean Elliptic Ratio - Rest'
        )

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(min(y_min_lumen, y_min_elliptic), max(y_max_lumen, y_max_elliptic))
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)
        ax1.invert_xaxis()
        ax1.legend()

        # Add a second x-axis for frames
        ax1_frames = ax1.twiny()
        ax1_frames.set_xlim(ax1.get_xlim())
        ax1_frames.set_xticks(df_rest['distance'][::5])
        ax1_frames.set_xticklabels(df_rest['frame'][::5])
        ax1_frames.set_xlabel('Frames')
        ax1.axhline(y=1.5, color='r', linestyle='--')

        # Plot lumen area for stress
        for phase, group in df_stress.groupby('phase'):
            ax2.plot(group['distance'], group[fitted], label=f'Dobutamine - {phase}')
            ax2.scatter(group['distance'], group[norm], alpha=0.3)
            ax2.fill_between(group['distance'], group['area_ci_lower'], group['area_ci_upper'], color='blue', alpha=0.2)

        # Highlight the smallest lumen area for dobutamine
        min_lumen_area_dobu = df_stress[norm].min()
        min_lumen_area_frame_dobu = df_stress.loc[df_stress[norm].idxmin(), 'frame']
        min_lumen_area_distance_dobu = df_stress.loc[df_stress[norm].idxmin(), 'distance']

        ax2.scatter(min_lumen_area_distance_dobu, min_lumen_area_dobu, color='#0055ff', zorder=5)
        ax2.text(
            min_lumen_area_distance_dobu,
            min_lumen_area_dobu,
            f'{min_lumen_area_dobu:.2f} ({min_lumen_area_frame_dobu})',
            color='#0055ff',
        )

        # Overlay mean elliptic ratio for dobutamine
        ax2.plot(
            df_stress['distance'],
            df_stress['fitted_elliptic_ratio'],
            color='darkblue',
            label='Mean Elliptic Ratio - Dobutamine',
        )

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(min(y_min_lumen, y_min_elliptic), max(y_max_lumen, y_max_elliptic))
        ax2.set_xlabel('Distance (mm)')
        ax2.set_ylabel(ylabel)
        ax2.set_title(title)
        ax2.invert_xaxis()
        ax2.legend()

        # Add a second x-axis for frames
        ax2_frames = ax2.twiny()
        ax2_frames.set_xlim(ax2.get_xlim())
        ax2_frames.set_xticks(df_stress['distance'][::5])
        ax2_frames.set_xticklabels(df_stress['frame'][::5])
        ax2_frames.set_xlabel('Frames')
        ax2.axhline(y=1.5, color='r', linestyle='--')

        plot_path = os.path.join(os.path.dirname(self.rest_dir), 'comparison.png')
        self.ensure_directory_exists(plot_path)
        plt.savefig(plot_path)
        plt.close()

    def plot_global(self, df_rest, df_dobu, variable='lumen_area'):
        """Plot global data for rest and stress side by side."""
        if variable == 'lumen_area':
            norm = 'lumen_area'
            fitted = 'fitted_lumen_area_glob'
            ylabel = 'Lumen Area (mm²)'

        elif variable == 'shortest_distance':
            norm = 'shortest_distance'
            fitted = 'fitted_shortest_distance_glob'
            ylabel = 'Shortest Distance (mm)'

        # Plot global data for rest
        plt.plot(df_rest['distance'], df_rest[fitted], label='Rest')
        plt.plot(df_dobu['distance'], df_dobu[fitted], label='Dobutamine')
        plt.fill_between(
            df_rest['distance'], df_rest['area_ci_lower_glob'], df_rest['area_ci_upper_glob'], color='blue', alpha=0.2
        )
        plt.fill_between(
            df_dobu['distance'], df_dobu['area_ci_lower_glob'], df_dobu['area_ci_upper_glob'], color='blue', alpha=0.2
        )
        plt.scatter(df_rest['distance'], df_rest[norm], alpha=0.3)
        plt.scatter(df_dobu['distance'], df_dobu[norm], alpha=0.3)
        plt.title('Global Lumen Area vs Distance')
        plt.xlabel('Distance (mm)')
        plt.ylabel(ylabel)
        plt.gca().invert_xaxis()  # Correct method to invert x-axis
        plt.legend()

        plot_path = os.path.join(os.path.dirname(self.rest_dir), 'global_comparison.png')
        plot_path = plot_path.replace('\\', '/')
        self.ensure_directory_exists(plot_path)
        plt.savefig(plot_path)

    def process_directory(self, directory):
        """Process all txt files in a directory."""
        if 'combined_sorted_manual.csv' in os.listdir(directory):
            file = 'combined_sorted_manual.csv'
        else:
            file = 'combined_sorted.csv'
        df = pd.read_csv(os.path.join(directory, file), sep=',')
        df = self.prep_data(df)
        df = self.fit_curves_sys_dia(df)
        df = self.fit_curve_global(df)
        return df

    def run(self):
        """Main method to process data and generate plots."""
        print("Starting run method")
        self.get_order_and_estimate()

        self.current_dir = 'rest'
        logger.info(f"Processing rest directory of {self.name_dir}")
        rest_df = self.process_directory(self.rest_dir)
        self.current_dir = 'stress'
        logger.info(f"Processing stress directory {self.name_dir}")
        stress_df = self.process_directory(self.stress_dir)

        self.plot_data(rest_df, stress_df, variable='lumen_area')
        self.plot_global(rest_df, stress_df, variable='lumen_area')

        rest_df.to_csv(os.path.join(os.path.dirname(self.rest_dir), 'output_rest.csv'))
        stress_df.to_csv(os.path.join(os.path.dirname(self.stress_dir), 'output_stress.csv'))


# Usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rest_dir", default="../data/NARCO_122/rest")
    parser.add_argument("--stress_dir", default="../data/NARCO_122/stress")
    args = parser.parse_args()
    # processor = IvusProcessor(
    #     rest_dir=args.rest_dir,
    #     stress_dir=args.stress_dir,
    # )
    processor = IvusProcessor(
        rest_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_192\rest",
        stress_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_192\stress",
    )
    processor.run()
