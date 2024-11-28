import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

pd.options.mode.chained_assignment = None  # default='warn'

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
        self.estimate_distance_true = 0 # if one the distance is estimated by hr, more error prone but works for faulty metadata with late pullback start
        self.PULLBACK_SPEED = 1
        self.START_FRAME = 0
        self.FRAME_RATE = 30

    def patient_id_from_dir(self, directory):
        """Extract the patient ID from the directory name. by keeping only NARCO_XX"""
        # remove \rest or \stress from the directory
        directory = directory.replace("\\rest", "").replace("\\stress", "")
        return directory.split("/")[-1]

    def get_order_and_estimate(self):
        # read them from the excel file at "C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\00_polynomial_fit_all.xlsx"
        try:
            order_list = pd.read_excel(r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\00_polynomial_fit_all.xlsx")
            self.phase_order = order_list[order_list['narco_id'] == self.name_dir]['phase_order'].values[0]
            self.global_order = order_list[order_list['narco_id'] == self.name_dir]['global_order'].values[0]
            self.estimate_distance_true = order_list[order_list['narco_id'] == self.name_dir]['estimate_distance_true'].values[0]

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
            logger.error(f'{e}. Using default values for pullback speed, start frame and frame rate.')

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
        if df_dia['frame'].iloc[: len(df_dia) // 2].sum() > df_dia['frame'].iloc[len(df_dia) // 2 :].sum():
            distance_dia = sorted(df_dia['distance'])
        else:
            distance_dia = sorted(df_dia['distance'], reverse=True)

        if df_sys['frame'].iloc[: len(df_sys) // 2].sum() > df_sys['frame'].iloc[len(df_sys) // 2 :].sum():
            distance_sys = sorted(df_sys['distance'])
        else:
            distance_sys = sorted(df_sys['distance'], reverse=True)

        df_dia['distance'] = distance_dia
        df_sys['distance'] = distance_sys

        return pd.concat([df_dia, df_sys])

    @staticmethod
    def polynomial_fit(x, y, degree=10):
        # Remove NaN and infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < degree + 1:
            raise ValueError("Not enough valid data points to fit the polynomial.")

        try:
            p = np.polyfit(x_clean, y_clean, degree)
            return np.polyval(p, x)
        except np.linalg.LinAlgError as e:
            logger.error(f"Error in polynomial fitting: {e}")
            return np.full_like(x, np.nan)

    def fit_curves_sys_dia(self, df):
        """Fits a polynomial curve to the lumen area and elliptic ratio."""
        df['fitted_lumen_area'] = df.groupby('phase', group_keys=False).apply(
            lambda group: pd.Series(
                self.polynomial_fit(group['distance'], group['lumen_area'], self.phase_order), index=group.index
            ),
            include_groups=False,
        )
        df['mean_elliptic_ratio'] = df.groupby('distance')['elliptic_ratio'].transform('mean')
        df['fitted_elliptic_ratio'] = self.polynomial_fit(df['distance'], df['mean_elliptic_ratio'], self.phase_order)
        df['fitted_shortest_distance'] = df.groupby('phase', group_keys=False).apply(
            lambda group: pd.Series(
                self.polynomial_fit(group['distance'], group['shortest_distance'], self.phase_order), index=group.index
            ),
            include_groups=False,
        )
        df = df.sort_values(by='distance')
        return df

    def fit_curve_global(self, df):
        """Fits a polynomial curve to the lumen area and elliptic ratio."""
        df['fitted_lumen_area_glob'] = self.polynomial_fit(df['distance'], df['lumen_area'], self.global_order)
        df['mean_elliptic_ratio_glob'] = df['elliptic_ratio'].mean()
        df['fitted_elliptic_ratio_glob'] = self.polynomial_fit(df['distance'], df['mean_elliptic_ratio'], self.global_order)
        df['fitted_shortest_distance_glob'] = self.polynomial_fit(df['distance'], df['shortest_distance'], self.global_order)
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

    def plot_data(self, df_rest, df_dobu, variable='lumen_area'):
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

        x_min = min(df_rest['distance'].min(), df_dobu['distance'].min())
        x_max = max(df_rest['distance'].max(), df_dobu['distance'].max())

        # Determine the common y-axis limits based on the dataframe with more y-values
        y_min_lumen = min(df_rest[norm].min(), df_dobu[norm].min())
        y_max_lumen = max(df_rest[norm].max(), df_dobu[norm].max())
        y_min_elliptic = min(df_rest['mean_elliptic_ratio'].min(), df_dobu['mean_elliptic_ratio'].min())
        y_max_elliptic = max(df_rest['mean_elliptic_ratio'].max(), df_dobu['mean_elliptic_ratio'].max())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.4})

        # Plot lumen area for rest
        for phase, group in df_rest.groupby('phase'):
            ax1.plot(group['distance'], group[fitted], label=f'Rest - {phase}')
            ax1.scatter(group['distance'], group[norm], alpha=0.3)

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

        # Plot lumen area for dobutamine
        for phase, group in df_dobu.groupby('phase'):
            ax2.plot(group['distance'], group[fitted], label=f'Dobutamine - {phase}')
            ax2.scatter(group['distance'], group[norm], alpha=0.3)

        # Highlight the smallest lumen area for dobutamine
        min_lumen_area_dobu = df_dobu[norm].min()
        min_lumen_area_frame_dobu = df_dobu.loc[df_dobu[norm].idxmin(), 'frame']
        min_lumen_area_distance_dobu = df_dobu.loc[df_dobu[norm].idxmin(), 'distance']

        ax2.scatter(min_lumen_area_distance_dobu, min_lumen_area_dobu, color='#0055ff', zorder=5)
        ax2.text(
            min_lumen_area_distance_dobu,
            min_lumen_area_dobu,
            f'{min_lumen_area_dobu:.2f} ({min_lumen_area_frame_dobu})',
            color='#0055ff',
        )

        # Overlay mean elliptic ratio for dobutamine
        ax2.plot(
            df_dobu['distance'],
            df_dobu['fitted_elliptic_ratio'],
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
        ax2_frames.set_xticks(df_dobu['distance'][::5])
        ax2_frames.set_xticklabels(df_dobu['frame'][::5])
        ax2_frames.set_xlabel('Frames')
        ax2.axhline(y=1.5, color='r', linestyle='--')

        plot_path = os.path.join(os.path.dirname(self.rest_dir), 'comparison.png')
        self.ensure_directory_exists(plot_path)
        plt.savefig(plot_path)

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
        ax4.set_title('Rearranged Dobutamine Data')
        ax4.set_xlabel('Distance (mm)')
        ax4.set_ylabel(ylabel)
        ax4.invert_xaxis()
        ax4.legend()

        plot_path = os.path.join(os.path.dirname(self.rest_dir), 'comparison.png')
        plot_path = plot_path.replace('\\', '/')
        self.ensure_directory_exists(plot_path)
        plt.savefig(plot_path)

    def plot_global(self, df_rest, df_dobu):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df_rest['distance'], df_rest['fitted_lumen_area_glob'], label='Rest')
        ax.plot(df_dobu['distance'], df_dobu['fitted_lumen_area_glob'], label='Dobutamine')
        # plot all points where phase is not '-'
        ax.scatter(df_rest['distance'], df_rest['lumen_area'], alpha=0.3)
        ax.scatter(df_dobu['distance'], df_dobu['lumen_area'], alpha=0.3)

        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Lumen Area (mm²)')
        ax.set_title('Global Lumen Area vs Distance')
        ax.invert_xaxis()
        ax.legend()

        plot_path = os.path.join(os.path.dirname(self.rest_dir), 'global.png')
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
                df = self.load_data(os.path.join(directory, file), sep='\t')
            elif file.endswith('.csv'):
                df = self.load_data(os.path.join(directory, file), sep=',')
            df = self.prep_data(df)
            df = self.fit_curves_sys_dia(df)
            df = self.fit_curve_global(df)
            dfs[file] = df
        return dfs

    def run(self):
        """Main method to process data and generate plots."""
        self.get_order_and_estimate()

        rest_data = self.process_directory(self.rest_dir)
        stress_data = self.process_directory(self.stress_dir)

        try:
            rest_df = next(df for filename, df in rest_data.items() if filename.endswith('_report.txt'))
            rest_df_rearranged = next(df for filename, df in rest_data.items() if filename.endswith('combined_sorted.csv') or filename == 'combined_sorted_manual.csv')
            stress_df = next(df for filename, df in stress_data.items() if filename.endswith('_report.txt'))
            stress_df_rearranged = next(df for filename, df in stress_data.items() if filename.endswith('combined_sorted.csv') or filename == 'combined_sorted_manual.csv')
        except StopIteration:
            logger.error("Required files not found in the directories.")
            return

        self.plot_data_comparison(rest_df, stress_df, rest_df_rearranged, stress_df_rearranged, variable='lumen_area')
        self.plot_global_comparison(rest_df, stress_df, rest_df_rearranged, stress_df_rearranged, variable='lumen_area')

        rest_df.to_csv(os.path.join(self.rest_dir, 'output.csv'))
        stress_df.to_csv(os.path.join(self.stress_dir, 'output.csv'))
        rest_df_rearranged.to_csv(os.path.join(self.rest_dir, 'output_rearranged.csv'))
        stress_df_rearranged.to_csv(os.path.join(self.stress_dir, 'output_rearranged.csv'))


if __name__ == "__main__":
    processor = IvusProcessor(
        rest_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_24\rest",
        stress_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_24\stress",
    )
    processor.run()
