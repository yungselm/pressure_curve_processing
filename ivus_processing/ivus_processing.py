import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm

pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.nonparametric.smoothers_lowess import lowess


from loguru import logger


class IvusProcessor:
    def __init__(self, rest_dir, stress_dir):
        self.rest_dir = rest_dir
        self.stress_dir = stress_dir
        self.name_dir = self.patient_id_from_dir(rest_dir)
        self.rest_dir_rearranged = None
        self.stress_dir_rearranged = None
        self.phase_order = 10
        self.global_order = 4
        self.estimate_distance_true = 0  # if one the distance is estimated by hr, more error prone but works for faulty metadata with late pullback start
        self.PULLBACK_SPEED = 1
        self.START_FRAME = 0
        self.FRAME_RATE = 30

    def patient_id_from_dir(self, directory):
        """Extract the patient ID from the directory name. by keeping only NARCO_XX"""
        # remove \rest or \stress from the directory
        directory = directory.replace("\\rest", "").replace("\\stress", "")
        return directory.split("/")[-1]

    def get_order_and_estimate(self):
        try:
            order_list = pd.read_excel(
                r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\00_polynomial_fit_all.xlsx"
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

    def get_global_variables(self, df):
        """
        Calculates the global variables for the lumen area and elliptic ratio.
        """
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
            self.get_global_variables(df)
        except ValueError as e:
            print(e)
            print("Using default values for pullback speed, start frame and frame rate.")

        df = df[df['phase'] != '-'].copy()
        # df = df[df['frame'] >= self.START_FRAME].copy()

        df_dia = df[df['phase'] == 'D'].copy()
        df_sys = df[df['phase'] == 'S'].copy()

        if self.estimate_distance_true == 1:
            df_dia['distance'] = self.estimate_distance(df_dia)
            df_sys['distance'] = self.estimate_distance(df_sys)
        else:
            df_dia['distance'] = (df_dia['position'] - df_dia['position'].max()) * -1
            df_sys['distance'] = (df_sys['position'] - df_sys['position'].max()) * -1

        # if sum of first half of df_dia['frame'] is bigger than sum of last half of df_dia['frame'], ascending order otherwise descending same for df_sys
        if df_dia['frame'].iloc[:len(df_dia) // 2].sum() > df_dia['frame'].iloc[len(df_dia) // 2:].sum():
            distance_dia = sorted(df_dia['distance'])
        else:
            distance_dia = sorted(df_dia['distance'], reverse=True)

        if df_sys['frame'].iloc[:len(df_sys) // 2].sum() > df_sys['frame'].iloc[len(df_sys) // 2:].sum():
            distance_sys = sorted(df_sys['distance'])
        else:
            distance_sys = sorted(df_sys['distance'], reverse=True)

        df_dia['distance'] = distance_dia
        df_sys['distance'] = distance_sys

        return pd.concat([df_dia, df_sys])


    @staticmethod
    def loess_fit(x, y, frac=0.35, ci=False):
        # Data
        x = x.to_numpy()
        y = y.to_numpy()

        # Perform LOESS smoothing
        smoothed = lowess(y, x, frac=frac, return_sorted=True)
        x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]

        # Bootstrapping for Confidence Intervals
        n_boot = 1000  # Number of bootstrap samples
        boot_preds = []

        if ci:
            for _ in tqdm.tqdm(range(n_boot), desc=f"Bootstrapping for {os.path.basename(__file__)}"):
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

        return y_smooth, ci_lower, ci_upper

    def fit_curves_sys_dia(self, df):
        """Fits a polynomial curve to the lumen area and elliptic ratio."""
        # Ensure distance is sorted within each phase
        fitted_results = df.groupby('phase').apply(
            lambda group: pd.DataFrame({
                'fitted_lumen_area': self.loess_fit(group.sort_values('distance')['distance'], group.sort_values('distance')['lumen_area'])[0],
                'area_ci_lower': self.loess_fit(group.sort_values('distance')['distance'], group.sort_values('distance')['lumen_area'])[1],
                'area_ci_upper': self.loess_fit(group.sort_values('distance')['distance'], group.sort_values('distance')['lumen_area'])[2]
            }, index=group.index)
        ).reset_index(drop=True)
        
        df = df.reset_index(drop=True)
        df = pd.concat([df, fitted_results], axis=1)
        
        # Ensure sorted and consistent distance for mean elliptic ratio
        df['mean_elliptic_ratio'] = df.groupby('distance')['elliptic_ratio'].transform('mean')
        df = df.sort_values('distance').reset_index(drop=True)  # Sort overall for consistency and reset index
        fitted_elliptic_results = pd.DataFrame({
            'fitted_elliptic_ratio': self.loess_fit(df['distance'], df['mean_elliptic_ratio'])[0],
            'elliptic_ci_lower': self.loess_fit(df['distance'], df['mean_elliptic_ratio'])[1],
            'elliptic_ci_upper': self.loess_fit(df['distance'], df['mean_elliptic_ratio'])[2]
        }).reset_index(drop=True)
        
        df = pd.concat([df, fitted_elliptic_results], axis=1)
        
        # Apply similar fixes for shortest distance
        fitted_shortest_results = df.groupby('phase').apply(
            lambda group: pd.DataFrame({
                'fitted_shortest_distance': self.loess_fit(group.sort_values('distance')['distance'], group.sort_values('distance')['shortest_distance'])[0],
                'shortest_ci_lower': self.loess_fit(group.sort_values('distance')['distance'], group.sort_values('distance')['shortest_distance'])[1],
                'shortest_ci_upper': self.loess_fit(group.sort_values('distance')['distance'], group.sort_values('distance')['shortest_distance'])[2]
            }, index=group.index)
        ).reset_index(drop=True)
        
        df = pd.concat([df, fitted_shortest_results], axis=1)
        return df.sort_values(by='distance').reset_index(drop=True)  # Final sort and reset index

    def fit_curve_global(self, df):
        """Fits a polynomial curve to the lumen area and elliptic ratio."""
        df['fitted_lumen_area_glob'], df['area_ci_lower_glob'], df['area_ci_upper_glob'] = self.loess_fit(df['distance'], df['lumen_area'])
        df['mean_elliptic_ratio_glob'] = df['elliptic_ratio'].mean()
        df['fitted_elliptic_ratio_glob'], df['elliptic_ci_lower_glob'], df['elliptic_ci_upper_glob'] = self.loess_fit(df['distance'], df['mean_elliptic_ratio'])
        df['fitted_shortest_distance_glob'], df['shortest_ci_lower_glob'], df['shortest_ci_upper_glob'] = self.loess_fit(df['distance'], df['shortest_distance'])
        df = df.sort_values(by='distance')
        return df

    def load_data(self, file_path, sep='\t'):
        """Load data from a file."""
        return pd.read_csv(file_path, sep=sep)

    def ensure_directory_exists(self, path):
        """Ensure the directory for the given path exists."""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_data_comparison(self, df_rest, df_dobu, df_rest_rearranged, df_dobu_rearranged, variable='lumen_area'):
        """Plot original and rearranged systole and diastole data for rest and stress side by side."""
        if variable == 'lumen_area':
            norm = 'lumen_area'
            fitted = 'fitted_lumen_area'
            ylabel = 'Lumen Area (mm²)'
        elif variable == 'shortest_distance':
            norm = 'shortest_distance'
            fitted = 'fitted_shortest_distance'
            ylabel = 'Shortest Distance (mm)'
        else:
            raise ValueError(f"Variable {variable} not supported.")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.4})

        # Plot original data for rest
        ax1 = axes[0, 0]
        for phase, group in df_rest.groupby('phase'):
            ax1.plot(group['distance'], group[fitted], label=f'Rest - {phase}')
            ax1.fill_between(group['distance'], group['area_ci_lower'], group['area_ci_upper'], color='blue', alpha=0.2)
            ax1.scatter(group['distance'], group[norm], alpha=0.3)
        ax1.set_title('Original Rest Data')
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel(ylabel)
        ax1.invert_xaxis()
        ax1.legend()

        # Plot rearranged data for rest
        ax2 = axes[0, 1]
        for phase, group in df_rest_rearranged.groupby('phase'):
            ax2.plot(group['distance'], group[fitted], label=f'Rest Rearranged - {phase}')
            ax2.fill_between(group['distance'], group['area_ci_lower'], group['area_ci_upper'], color='blue', alpha=0.2)
            ax2.scatter(group['distance'], group[norm], alpha=0.3)
        ax2.set_title('Rearranged Rest Data')
        ax2.set_xlabel('Distance (mm)')
        ax2.set_ylabel(ylabel)
        ax2.invert_xaxis()
        ax2.legend()

        # Plot original data for dobutamine
        ax3 = axes[1, 0]
        for phase, group in df_dobu.groupby('phase'):
            ax3.plot(group['distance'], group[fitted], label=f'Dobutamine - {phase}')
            ax3.scatter(group['distance'], group[norm], alpha=0.3)
            ax3.fill_between(group['distance'], group['area_ci_lower'], group['area_ci_upper'], color='blue', alpha=0.2)
        ax3.set_title('Original Dobutamine Data')
        ax3.set_xlabel('Distance (mm)')
        ax3.set_ylabel(ylabel)
        ax3.invert_xaxis()
        ax3.legend()

        # Plot rearranged data for dobutamine
        ax4 = axes[1, 1]
        for phase, group in df_dobu_rearranged.groupby('phase'):
            ax4.plot(group['distance'], group[fitted], label=f'Dobutamine Rearranged - {phase}')
            ax4.scatter(group['distance'], group[norm], alpha=0.3)
            ax4.fill_between(group['distance'], group['area_ci_lower'], group['area_ci_upper'], color='blue', alpha=0.2)
        ax4.set_title('Rearranged Dobutamine Data')
        ax4.set_xlabel('Distance (mm)')
        ax4.set_ylabel(ylabel)
        ax4.invert_xaxis()
        ax4.legend()

        plot_path = os.path.join(os.path.dirname(self.rest_dir), 'comparison.png')
        plot_path = plot_path.replace('\\', '/')
        self.ensure_directory_exists(plot_path)
        plt.savefig(plot_path)

    def plot_global_comparison(self, df_rest, df_dobu, df_rest_rearranged, df_dobu_rearranged, variable='lumen_area'):
        """Plot global data for rest and stress side by side."""
        if variable == 'lumen_area':
            norm = 'lumen_area'
            fitted = 'fitted_lumen_area_glob'
            ylabel = 'Lumen Area (mm²)'

        elif variable == 'shortest_distance':
            norm = 'shortest_distance'
            fitted = 'fitted_shortest_distance_glob'
            ylabel = 'Shortest Distance (mm)'

        fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.4})

        # Plot global data for rest
        ax1 = axes[0]
        ax1.plot(df_rest['distance'], df_rest[fitted], label='Rest')
        ax1.plot(df_dobu['distance'], df_dobu[fitted], label='Dobutamine')
        ax1.fill_between(df_rest['distance'], df_rest['area_ci_lower_glob'], df_rest['area_ci_upper_glob'], color='blue', alpha=0.2)
        ax1.fill_between(df_dobu['distance'], df_dobu['area_ci_lower_glob'], df_dobu['area_ci_upper_glob'], color='blue', alpha=0.2)
        ax1.scatter(df_rest['distance'], df_rest[norm], alpha=0.3)
        ax1.scatter(df_dobu['distance'], df_dobu[norm], alpha=0.3)
        ax1.set_title('Global Lumen Area vs Distance')
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel(ylabel)
        ax1.invert_xaxis()
        ax1.legend()

        # Plot global data for dobutamine
        ax2 = axes[1]
        ax2.plot(df_rest_rearranged['distance'], df_rest_rearranged[fitted], label='Rest Rearranged')
        ax2.plot(df_dobu_rearranged['distance'], df_dobu_rearranged[fitted], label='Dobutamine Rearranged')
        ax2.fill_between(df_rest_rearranged['distance'], df_rest_rearranged['area_ci_lower_glob'], df_rest_rearranged['area_ci_upper_glob'], color='blue', alpha=0.2)
        ax2.fill_between(df_dobu_rearranged['distance'], df_dobu_rearranged['area_ci_lower_glob'], df_dobu_rearranged['area_ci_upper_glob'], color='blue', alpha=0.2)
        ax2.scatter(df_rest_rearranged['distance'], df_rest_rearranged[norm], alpha=0.3)
        ax2.scatter(df_dobu_rearranged['distance'], df_dobu_rearranged[norm], alpha=0.3)
        ax2.set_title('Global Lumen Area vs Distance (Rearranged)')
        ax2.set_xlabel('Distance (mm)')
        ax2.set_ylabel(ylabel)
        ax2.invert_xaxis()
        ax2.legend()

        plot_path = os.path.join(os.path.dirname(self.rest_dir), 'global_comparison.png')
        plot_path = plot_path.replace('\\', '/')
        self.ensure_directory_exists(plot_path)
        plt.savefig(plot_path)

    def process_directory(self, directory):
        """Process all txt files in a directory."""
        if 'combined_sorted_manual.csv' in os.listdir(directory):
            files = [f for f in os.listdir(directory) if f.endswith('_report.txt') or f == 'combined_sorted_manual.csv']
        else:
            files = [f for f in os.listdir(directory) if f.endswith('_report.txt') or f.endswith('combined_sorted.csv')]
        # sort so that _report.txt files come first, so that pullbackspeed etc. from report.txt can be used for combined_sorted.csv
        files = sorted(files, key=lambda x: x.endswith('_report.txt'), reverse=True)
        dfs = {}
        for file in files:
            if file.endswith('.txt'):
                flag = 1
                df = self.load_data(os.path.join(directory, file), sep='\t')
            elif file.endswith('.csv'):
                flag = 0
                df = self.load_data(os.path.join(directory, file), sep=',')
            df = self.prep_data(df)
            df = self.fit_curves_sys_dia(df)
            df = self.fit_curve_global(df)
            dfs[file] = df
        return dfs
    
    def run(self):
        """Main method to process data and generate plots."""
        print("Starting run method")
        self.get_order_and_estimate()

        rest_data = self.process_directory(self.rest_dir)
        stress_data = self.process_directory(self.stress_dir)

        try:
            rest_df = next(df for filename, df in rest_data.items() if filename.endswith('_report.txt'))
            rest_df_rearranged = next(
                df
                for filename, df in rest_data.items()
                if filename.endswith('combined_sorted.csv') or filename == 'combined_sorted_manual.csv'
            )
            stress_df = next(df for filename, df in stress_data.items() if filename.endswith('_report.txt'))
            stress_df_rearranged = next(
                df
                for filename, df in stress_data.items()
                if filename.endswith('combined_sorted.csv') or filename == 'combined_sorted_manual.csv'
            )
        except StopIteration:
            logger.error("Required files not found in the directories.")
            return
        
        # make a check, if first half of fitted_lumen_area is bigger than second half, reverse all columns beginning with fitted while keeping other columns
        if (rest_df['lumen_area'] - rest_df['fitted_lumen_area']).abs().sum() > (rest_df['lumen_area'] - rest_df['fitted_lumen_area'][::-1]).abs().sum():
            for col in rest_df.columns:
                if col.startswith('fitted'):
                    rest_df[col] = rest_df[col].values[::-1]
        
        if (stress_df['lumen_area'] - stress_df['fitted_lumen_area']).abs().sum() > (stress_df['lumen_area'] - stress_df['fitted_lumen_area'][::-1]).abs().sum():
            for col in stress_df.columns:
                if col.startswith('fitted'):
                    stress_df[col] = stress_df[col].values[::-1]

        # reset indices
        rest_df = rest_df.reset_index(drop=True)
        stress_df = stress_df.reset_index(drop=True)

        self.plot_data_comparison(rest_df, stress_df, rest_df_rearranged, stress_df_rearranged, variable='lumen_area')
        self.plot_global_comparison(rest_df, stress_df, rest_df_rearranged, stress_df_rearranged, variable='lumen_area')

        rest_df.to_csv(os.path.join(self.rest_dir, 'output.csv'))
        stress_df.to_csv(os.path.join(self.stress_dir, 'output.csv'))
        rest_df_rearranged.to_csv(os.path.join(self.rest_dir, 'output_rearranged.csv'))
        stress_df_rearranged.to_csv(os.path.join(self.stress_dir, 'output_rearranged.csv'))

# Usage
processor = IvusProcessor(
    rest_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_24\rest",
    stress_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_24\stress")
processor.run()
