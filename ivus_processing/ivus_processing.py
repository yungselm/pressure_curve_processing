import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
pd.options.mode.chained_assignment = None  # default='warn'


class IvusProcessor:
    def __init__(self, rest_dir, stress_dir):
        self.rest_dir = rest_dir
        self.stress_dir = stress_dir
        self.rest_dir_rearranged = None
        self.stress_dir_rearranged = None
        self.PULLBACK_SPEED = 1
        self.START_FRAME = 0
        self.FRAME_RATE = 30

    def get_global_variables(self, df):
        """
        Calculates the global variables for the lumen area and elliptic ratio.
        """
        if df is None or df.empty or pd.isna(df['pullback_speed'].unique()[0]):
            raise ValueError("The dataframe is empty or not initialized. Please provide a valid dataframe.")

        self.PULLBACK_SPEED = df['pullback_speed'].unique()[0]
        self.START_FRAME = df['pullback_start_frame'].unique()[0]
        self.FRAME_RATE = df['frame_rate'].unique()[0]

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
        df = df[df['frame'] >= self.START_FRAME].copy()

        df_dia = df[df['phase'] == 'D'].copy()
        df_sys = df[df['phase'] == 'S'].copy()

        df_dia['distance'] = (df_dia['frame'].max() - df_dia['frame']) / self.FRAME_RATE * self.PULLBACK_SPEED
        df_sys['distance'] = (df_sys['frame'].max() - df_sys['frame']) / self.FRAME_RATE * self.PULLBACK_SPEED

        df_dia_distance = df_dia['distance'].values
        df_dia_distance = np.sort(df_dia_distance)[::-1]
        df_dia['distance'] = df_dia_distance

        df_sys_distance = df_sys['distance'].values
        df_sys_distance = np.sort(df_sys_distance)[::-1]
        df_sys['distance'] = df_sys_distance

        return pd.concat([df_dia, df_sys])

    @staticmethod
    def polynomial_fit(x, y, degree=10):
        p = np.polyfit(x, y, degree)
        return np.polyval(p, x)

    def fit_curves_sys_dia(self, df, degree=10):
            """Fits a polynomial curve to the lumen area and elliptic ratio."""
            df['fitted_lumen_area'] = df.groupby('phase', group_keys=False).apply(
                lambda group: pd.Series(self.polynomial_fit(group['distance'], group['lumen_area'], degree), index=group.index)
            )
            df['mean_elliptic_ratio'] = df.groupby('distance')['elliptic_ratio'].transform('mean')
            df['fitted_elliptic_ratio'] = self.polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
            df = df.sort_values(by='distance')
            return df

    def fit_curve_global(self, df, degree=4):
        """Fits a polynomial curve to the lumen area and elliptic ratio."""
        df['fitted_lumen_area_glob'] = self.polynomial_fit(df['distance'], df['lumen_area'], degree)
        df['mean_elliptic_ratio_glob'] = df['elliptic_ratio'].mean()
        df['fitted_elliptic_ratio_glob'] = self.polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
        df = df.sort_values(by='distance')
        return df

    def load_data(self, file_path, sep='\t'):
        """Load data from a file."""
        return pd.read_csv(file_path, sep=sep)

    def plot_data(self, df_rest, df_dobu):
        """Plot systole and diastole data for rest and stress."""
        # Determine the common x-axis limits based on the dataframe with more x-values
        x_min = min(df_rest['distance'].min(), df_dobu['distance'].min())
        x_max = max(df_rest['distance'].max(), df_dobu['distance'].max())

        # Determine the common y-axis limits based on the dataframe with more y-values
        y_min_lumen = min(df_rest['lumen_area'].min(), df_dobu['lumen_area'].min())
        y_max_lumen = max(df_rest['lumen_area'].max(), df_dobu['lumen_area'].max())
        y_min_elliptic = min(df_rest['mean_elliptic_ratio'].min(), df_dobu['mean_elliptic_ratio'].min())
        y_max_elliptic = max(df_rest['mean_elliptic_ratio'].max(), df_dobu['mean_elliptic_ratio'].max())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.4})

        # Plot lumen area for rest
        for phase, group in df_rest.groupby('phase'):
            ax1.plot(group['distance'], group['fitted_lumen_area'], label=f'Rest - {phase}')
            ax1.scatter(group['distance'], group['lumen_area'], alpha=0.3)

        # Highlight the smallest lumen area for rest
        min_lumen_area_rest = df_rest['lumen_area'].min()
        min_lumen_area_frame_rest = df_rest.loc[df_rest['lumen_area'].idxmin(), 'frame']
        min_lumen_area_distance_rest = df_rest.loc[df_rest['lumen_area'].idxmin(), 'distance']

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
        ax1.set_ylabel('Lumen Area (mm²) and Elliptic Ratio')
        ax1.set_title('Lumen Area vs Distance (Rest)')
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
            ax2.plot(group['distance'], group['fitted_lumen_area'], label=f'Dobutamine - {phase}')
            ax2.scatter(group['distance'], group['lumen_area'], alpha=0.3)

        # Highlight the smallest lumen area for dobutamine
        min_lumen_area_dobu = df_dobu['lumen_area'].min()
        min_lumen_area_frame_dobu = df_dobu.loc[df_dobu['lumen_area'].idxmin(), 'frame']
        min_lumen_area_distance_dobu = df_dobu.loc[df_dobu['lumen_area'].idxmin(), 'distance']

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
        ax2.set_ylabel('Lumen Area (mm²) and Elliptic Ratio')
        ax2.set_title('Lumen Area vs Distance (Dobutamine)')
        ax2.invert_xaxis()
        ax2.legend()

        # Add a second x-axis for frames
        ax2_frames = ax2.twiny()
        ax2_frames.set_xlim(ax2.get_xlim())
        ax2_frames.set_xticks(df_dobu['distance'][::5])
        ax2_frames.set_xticklabels(df_dobu['frame'][::5])
        ax2_frames.set_xlabel('Frames')
        ax2.axhline(y=1.5, color='r', linestyle='--')

        plt.show()  # Ensure plots are displayed

    def plot_data_comparison(self, df_rest, df_dobu, df_rest_rearranged, df_dobu_rearranged):
        """Plot original and rearranged systole and diastole data for rest and stress side by side."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.4})

        # Plot original data for rest
        ax1 = axes[0, 0]
        for phase, group in df_rest.groupby('phase'):
            ax1.plot(group['distance'], group['fitted_lumen_area'], label=f'Rest - {phase}')
            ax1.scatter(group['distance'], group['lumen_area'], alpha=0.3)
        ax1.set_title('Original Rest Data')
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel('Lumen Area (mm²)')
        ax1.invert_xaxis()
        ax1.legend()

        # Plot rearranged data for rest
        ax2 = axes[0, 1]
        for phase, group in df_rest_rearranged.groupby('phase'):
            ax2.plot(group['distance'], group['fitted_lumen_area'], label=f'Rest Rearranged - {phase}')
            ax2.scatter(group['distance'], group['lumen_area'], alpha=0.3)
        ax2.set_title('Rearranged Rest Data')
        ax2.set_xlabel('Distance (mm)')
        ax2.set_ylabel('Lumen Area (mm²)')
        ax2.invert_xaxis()
        ax2.legend()

        # Plot original data for dobutamine
        ax3 = axes[1, 0]
        for phase, group in df_dobu.groupby('phase'):
            ax3.plot(group['distance'], group['fitted_lumen_area'], label=f'Dobutamine - {phase}')
            ax3.scatter(group['distance'], group['lumen_area'], alpha=0.3)
        ax3.set_title('Original Dobutamine Data')
        ax3.set_xlabel('Distance (mm)')
        ax3.set_ylabel('Lumen Area (mm²)')
        ax3.invert_xaxis()
        ax3.legend()

        # Plot rearranged data for dobutamine
        ax4 = axes[1, 1]
        for phase, group in df_dobu_rearranged.groupby('phase'):
            ax4.plot(group['distance'], group['fitted_lumen_area'], label=f'Dobutamine Rearranged - {phase}')
            ax4.scatter(group['distance'], group['lumen_area'], alpha=0.3)
        ax4.set_title('Rearranged Dobutamine Data')
        ax4.set_xlabel('Distance (mm)')
        ax4.set_ylabel('Lumen Area (mm²)')
        ax4.invert_xaxis()
        ax4.legend()

        plt.show()

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

        plt.show()  # Ensure plots are displayed

    def plot_global_comparison(self, df_rest, df_dobu, df_rest_rearranged, df_dobu_rearranged):
        """Plot global data for rest and stress side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.4})

        # Plot global data for rest
        ax1 = axes[0]
        ax1.plot(df_rest['distance'], df_rest['fitted_lumen_area_glob'], label='Rest')
        ax1.plot(df_dobu['distance'], df_dobu['fitted_lumen_area_glob'], label='Rest Rearranged')
        ax1.scatter(df_rest['distance'], df_rest['lumen_area'], alpha=0.3)
        ax1.scatter(df_dobu['distance'], df_dobu['lumen_area'], alpha=0.3)
        ax1.set_title('Global Lumen Area vs Distance (Rest)')
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel('Lumen Area (mm²)')
        ax1.invert_xaxis()
        ax1.legend()

        # Plot global data for dobutamine
        ax2 = axes[1]
        ax2.plot(df_rest_rearranged['distance'], df_rest_rearranged['fitted_lumen_area_glob'], label='Dobutamine')
        ax2.plot(df_dobu_rearranged['distance'], df_dobu_rearranged['fitted_lumen_area_glob'], label='Dobutamine Rearranged')
        ax2.scatter(df_rest_rearranged['distance'], df_rest_rearranged['lumen_area'], alpha=0.3)
        ax2.scatter(df_dobu_rearranged['distance'], df_dobu_rearranged['lumen_area'], alpha=0.3)
        ax2.set_title('Global Lumen Area vs Distance (Dobutamine)')
        ax2.set_xlabel('Distance (mm)')
        ax2.set_ylabel('Lumen Area (mm²)')
        ax2.invert_xaxis()
        ax2.legend()

        plt.show()

    def process_directory(self, directory):
        """Process all txt files in a directory."""
        print(f"Processing directory: {directory}")
        files = [f for f in os.listdir(directory) if f.endswith('_report.txt') or f.endswith('combined_sorted.csv')]
        # sort so that _report.txt files come first, so that pullbackspeed etc. from report.txt can be used for combined_sorted.csv
        files = sorted(files, key=lambda x: x.endswith('_report.txt'), reverse=True)
        dfs = {}
        for file in files:
            print(f"Processing file: {file}")
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
        print("Starting run method")
        rest_data = self.process_directory(self.rest_dir)
        stress_data = self.process_directory(self.stress_dir)

        rest_df = next(df for filename, df in rest_data.items() if filename.endswith('_report.txt'))
        rest_df_rearranged = next(df for filename, df in rest_data.items() if filename.endswith('combined_sorted.csv'))
        stress_df = next(df for filename, df in stress_data.items() if filename.endswith('_report.txt'))
        stress_df_rearranged = next(df for filename, df in stress_data.items() if filename.endswith('combined_sorted.csv'))

        self.plot_data_comparison(rest_df, stress_df, rest_df_rearranged, stress_df_rearranged)
        self.plot_global_comparison(rest_df, stress_df, rest_df_rearranged, stress_df_rearranged)


# Usage
processor = IvusProcessor(
    rest_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\rest",
    stress_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\stress")
processor.run()


# df_rest = pd.read_csv(
#     r'C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\rest\PD2EZDBF_report.txt', sep='\t'
# )
# df_dobu = pd.read_csv(
#     r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\stress\PD616KK1_report.txt", sep='\t'
# )
# df_rest_rearranged = pd.read_csv(
#     r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\rest\combined_sorted.csv"
#     )
# df_dobu_rearranged = pd.read_csv(
#     r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\stress\combined_sorted.csv"
#     )
# # df_rest = pd.read_csv(
# #     'C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_119_rest.txt', sep='\t'
# # )
# # df_dobu = pd.read_csv(
# #     'C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_119_stress.txt', sep='\t'
# # )

# PULLBACK_SPEED = 1  # mm/s
# START_FRAME = 0
# FRAME_RATE = 30  # frames per second


# def prep_data(df, start_frame, frame_rate, pullback_speed):
#     """Registers systole and diastole to each other based on the frame number and calculates the distance in mm based on frame rate and pullback speed."""
#     df = df[df['phase'] != '-'].copy()
#     df = df[df['frame'] >= start_frame].copy()

#     df_dia = df[df['phase'] == 'D'].copy()
#     df_sys = df[df['phase'] == 'S'].copy()

#     df_dia.loc[:, 'distance'] = (df_dia['frame'].max() - df_dia['frame']) / frame_rate * pullback_speed
#     df_sys.loc[:, 'distance'] = (df_sys['frame'].max() - df_sys['frame']) / frame_rate * pullback_speed

#     df_dia_distance = df_dia['distance'].values
#     df_dia_distance = np.sort(df_dia_distance)[::-1]  # Ensure descending order
#     df_dia['distance'] = df_dia_distance

#     df_sys_distance = df_sys['distance'].values
#     df_sys_distance = np.sort(df_sys_distance)[::-1]  # Ensure descending order
#     df_sys['distance'] = df_sys_distance

#     df = pd.concat([df_dia, df_sys])
#     return df


# def fit_curves_sys_dia(df, degree=10):
#     """Fits a polynomial curve to the lumen area and elliptic ratio."""
#     df['fitted_lumen_area'] = df.groupby('phase', group_keys=False).apply(
#         lambda group: pd.Series(polynomial_fit(group['distance'], group['lumen_area'], degree), index=group.index)
#     )
#     df['mean_elliptic_ratio'] = df.groupby('distance')['elliptic_ratio'].transform('mean')
#     df['fitted_elliptic_ratio'] = polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
#     df = df.sort_values(by='distance')

#     return df


# def fit_curve_global(df, degree=4):
#     """Fits a polynomial curve to the lumen area and elliptic ratio."""
#     df['fitted_lumen_area_glob'] = polynomial_fit(df['distance'], df['lumen_area'], degree)
#     df['mean_elliptic_ratio_glob'] = df['elliptic_ratio'].mean()
#     df['fitted_elliptic_ratio_glob'] = polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
#     df = df.sort_values(by='distance')

#     return df


# def polynomial_fit(x, y, degree=10):
#     p = np.polyfit(x, y, degree)
#     return np.polyval(p, x)


# def plot_data(df_rest, df_dobu):
#     # Determine the common x-axis limits based on the dataframe with more x-values
#     x_min = min(df_rest['distance'].min(), df_dobu['distance'].min())
#     x_max = max(df_rest['distance'].max(), df_dobu['distance'].max())

#     # Determine the common y-axis limits based on the dataframe with more y-values
#     y_min_lumen = min(df_rest['lumen_area'].min(), df_dobu['lumen_area'].min())
#     y_max_lumen = max(df_rest['lumen_area'].max(), df_dobu['lumen_area'].max())
#     y_min_elliptic = min(df_rest['mean_elliptic_ratio'].min(), df_dobu['mean_elliptic_ratio'].min())
#     y_max_elliptic = max(df_rest['mean_elliptic_ratio'].max(), df_dobu['mean_elliptic_ratio'].max())

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.4})

#     # Plot lumen area for rest
#     for phase, group in df_rest.groupby('phase'):
#         ax1.plot(group['distance'], group['fitted_lumen_area'], label=f'Rest - {phase}')
#         ax1.scatter(group['distance'], group['lumen_area'], alpha=0.3)

#     # Highlight the smallest lumen area for rest
#     min_lumen_area_rest = df_rest['lumen_area'].min()
#     min_lumen_area_frame_rest = df_rest.loc[df_rest['lumen_area'].idxmin(), 'frame']
#     min_lumen_area_distance_rest = df_rest.loc[df_rest['lumen_area'].idxmin(), 'distance']

#     ax1.scatter(min_lumen_area_distance_rest, min_lumen_area_rest, color='#0055ff', zorder=5)
#     ax1.text(
#         min_lumen_area_distance_rest,
#         min_lumen_area_rest,
#         f'{min_lumen_area_rest:.2f} ({min_lumen_area_frame_rest})',
#         color='#0055ff',
#     )

#     # Overlay mean elliptic ratio for rest
#     ax1.plot(
#         df_rest['distance'], df_rest['fitted_elliptic_ratio'], color='darkblue', label='Mean Elliptic Ratio - Rest'
#     )

#     ax1.set_xlim(x_min, x_max)
#     ax1.set_ylim(min(y_min_lumen, y_min_elliptic), max(y_max_lumen, y_max_elliptic))
#     ax1.set_xlabel('Distance (mm)')
#     ax1.set_ylabel('Lumen Area (mm²) and Elliptic Ratio')
#     ax1.set_title('Lumen Area vs Distance (Rest)')
#     ax1.invert_xaxis()
#     ax1.legend()

#     # Add a second x-axis for frames
#     ax1_frames = ax1.twiny()
#     ax1_frames.set_xlim(ax1.get_xlim())
#     ax1_frames.set_xticks(df_rest['distance'][::5])
#     ax1_frames.set_xticklabels(df_rest['frame'][::5])
#     ax1_frames.set_xlabel('Frames')
#     ax1.axhline(y=1.5, color='r', linestyle='--')

#     # Plot lumen area for dobutamine
#     for phase, group in df_dobu.groupby('phase'):
#         ax2.plot(group['distance'], group['fitted_lumen_area'], label=f'Dobutamine - {phase}')
#         ax2.scatter(group['distance'], group['lumen_area'], alpha=0.3)

#     # Highlight the smallest lumen area for dobutamine
#     min_lumen_area_dobu = df_dobu['lumen_area'].min()
#     min_lumen_area_frame_dobu = df_dobu.loc[df_dobu['lumen_area'].idxmin(), 'frame']
#     min_lumen_area_distance_dobu = df_dobu.loc[df_dobu['lumen_area'].idxmin(), 'distance']

#     ax2.scatter(min_lumen_area_distance_dobu, min_lumen_area_dobu, color='#0055ff', zorder=5)
#     ax2.text(
#         min_lumen_area_distance_dobu,
#         min_lumen_area_dobu,
#         f'{min_lumen_area_dobu:.2f} ({min_lumen_area_frame_dobu})',
#         color='#0055ff',
#     )

#     # Overlay mean elliptic ratio for dobutamine
#     ax2.plot(
#         df_dobu['distance'],
#         df_dobu['fitted_elliptic_ratio'],
#         color='darkblue',
#         label='Mean Elliptic Ratio - Dobutamine',
#     )

#     ax2.set_xlim(x_min, x_max)
#     ax2.set_ylim(min(y_min_lumen, y_min_elliptic), max(y_max_lumen, y_max_elliptic))
#     ax2.set_xlabel('Distance (mm)')
#     ax2.set_ylabel('Lumen Area (mm²) and Elliptic Ratio')
#     ax2.set_title('Lumen Area vs Distance (Dobutamine)')
#     ax2.invert_xaxis()
#     ax2.legend()

#     # Add a second x-axis for frames
#     ax2_frames = ax2.twiny()
#     ax2_frames.set_xlim(ax2.get_xlim())
#     ax2_frames.set_xticks(df_dobu['distance'][::5])
#     ax2_frames.set_xticklabels(df_dobu['frame'][::5])
#     ax2_frames.set_xlabel('Frames')
#     ax2.axhline(y=1.5, color='r', linestyle='--')

#     plt.show()

# def plot_data_comparison(df_rest, df_dobu, df_rest_rearranged, df_dobu_rearranged):
#     """Plot original and rearranged systole and diastole data for rest and stress side by side."""
#     print("Plotting data comparison for rest and stress")

#     fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.4})

#     # Plot original data for rest
#     ax1 = axes[0, 0]
#     for phase, group in df_rest.groupby('phase'):
#         ax1.plot(group['distance'], group['fitted_lumen_area'], label=f'Rest - {phase}')
#         ax1.scatter(group['distance'], group['lumen_area'], alpha=0.3)
#     ax1.set_title('Original Rest Data')
#     ax1.set_xlabel('Distance (mm)')
#     ax1.set_ylabel('Lumen Area (mm²)')
#     ax1.invert_xaxis()
#     ax1.legend()

#     # Plot rearranged data for rest
#     ax2 = axes[0, 1]
#     for phase, group in df_rest_rearranged.groupby('phase'):
#         ax2.plot(group['distance'], group['fitted_lumen_area'], label=f'Rest Rearranged - {phase}')
#         ax2.scatter(group['distance'], group['lumen_area'], alpha=0.3)
#     ax2.set_title('Rearranged Rest Data')
#     ax2.set_xlabel('Distance (mm)')
#     ax2.set_ylabel('Lumen Area (mm²)')
#     ax2.invert_xaxis()
#     ax2.legend()

#     # Plot original data for dobutamine
#     ax3 = axes[1, 0]
#     for phase, group in df_dobu.groupby('phase'):
#         ax3.plot(group['distance'], group['fitted_lumen_area'], label=f'Dobutamine - {phase}')
#         ax3.scatter(group['distance'], group['lumen_area'], alpha=0.3)
#     ax3.set_title('Original Dobutamine Data')
#     ax3.set_xlabel('Distance (mm)')
#     ax3.set_ylabel('Lumen Area (mm²)')
#     ax3.invert_xaxis()
#     ax3.legend()

#     # Plot rearranged data for dobutamine
#     ax4 = axes[1, 1]
#     for phase, group in df_dobu_rearranged.groupby('phase'):
#         ax4.plot(group['distance'], group['fitted_lumen_area'], label=f'Dobutamine Rearranged - {phase}')
#         ax4.scatter(group['distance'], group['lumen_area'], alpha=0.3)
#     ax4.set_title('Rearranged Dobutamine Data')
#     ax4.set_xlabel('Distance (mm)')
#     ax4.set_ylabel('Lumen Area (mm²)')
#     ax4.invert_xaxis()
#     ax4.legend()

#     plt.show()

# def plot_global(df_rest, df_dobu):
#     """Plots fitted_lumen_area_global for rest and dobutamine in the same plot."""
#     fig, ax = plt.subplots(figsize=(10, 5))

#     ax.plot(df_rest['distance'], df_rest['fitted_lumen_area_glob'], label='Rest')
#     ax.plot(df_dobu['distance'], df_dobu['fitted_lumen_area_glob'], label='Dobutamine')
#     # plot all points where phase is not '-'
#     ax.scatter(df_rest['distance'], df_rest['lumen_area'], alpha=0.3)
#     ax.scatter(df_dobu['distance'], df_dobu['lumen_area'], alpha=0.3)

#     ax.set_xlabel('Distance (mm)')
#     ax.set_ylabel('Lumen Area (mm²)')
#     ax.set_title('Global Lumen Area vs Distance')
#     ax.invert_xaxis()
#     ax.legend()

#     plt.show()

# def create_df_from_dir(path):
#     """Creates a dataframe from a directory containing txt files."""
#     df = pd.DataFrame(columns=['filename', 'phase_order', 'global_order'])
#     for file in os.listdir(path):
#         if file.endswith('.txt'):
#             df = pd.concat([df, pd.DataFrame([{'filename': file, 'phase_order': np.nan, 'global_order': np.nan}])], ignore_index=True)
#     df.to_csv(os.path.join(path, 'input_parameters.csv'), index=False)

# df_rest = prep_data(df_rest, START_FRAME, FRAME_RATE, PULLBACK_SPEED)
# df_dobu = prep_data(df_dobu, START_FRAME, FRAME_RATE, PULLBACK_SPEED)
# df_rest_rearranged = prep_data(df_rest_rearranged, START_FRAME, FRAME_RATE, PULLBACK_SPEED)
# df_dobu_rearranged = prep_data(df_dobu_rearranged, START_FRAME, FRAME_RATE, PULLBACK_SPEED)

# print(df_rest.head())
# print(df_rest_rearranged.head())

# df_rest = fit_curves_sys_dia(df_rest)
# df_dobu = fit_curves_sys_dia(df_dobu)
# df_rest_rearranged = fit_curves_sys_dia(df_rest_rearranged)
# df_dobu_rearranged = fit_curves_sys_dia(df_dobu_rearranged)

# print(df_rest.head())
# print(df_rest_rearranged.head())

# df_rest = fit_curve_global(df_rest)
# df_dobu = fit_curve_global(df_dobu)
# df_rest_rearranged = fit_curve_global(df_rest_rearranged)
# df_dobu_rearranged = fit_curve_global(df_dobu_rearranged)

# plot_data_comparison(df_rest, df_dobu, df_rest_rearranged, df_dobu_rearranged)
# plot_data(df_rest_rearranged, df_dobu_rearranged)
# plot_global(df_rest, df_dobu)

# create_df_from_dir('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports')
