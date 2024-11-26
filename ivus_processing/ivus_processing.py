import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
pd.options.mode.chained_assignment = None  # default='warn'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PressureCurveProcessor:
    def __init__(self, rest_dir, stress_dir):
        self.rest_dir = rest_dir
        self.stress_dir = stress_dir
        self.PULLBACK_SPEED = None
        self.START_FRAME = None
        self.FRAME_RATE = None

    def get_global_variables(self, df):
        """
        Calculates the global variables for the lumen area and elliptic ratio.
        """
        if df is None or df.empty:
            raise ValueError("The dataframe is empty or not initialized. Please provide a valid dataframe.")

        self.PULLBACK_SPEED = df['pullback_speed'].unique()[0]
        self.START_FRAME = df['pullback_start_frame'].unique()[0]
        self.FRAME_RATE = df['frame_rate'].unique()[0]

    def prep_data(self, df):
        """
        Prepare data by filtering and calculating distances.
        Uses the global variables for processing.
        """
        if self.PULLBACK_SPEED is None or self.START_FRAME is None or self.FRAME_RATE is None:
            self.get_global_variables(df)

        df = df[df['phase'] != '-'].copy()
        df = df[df['frame'] >= self.START_FRAME].copy()

        df_dia = df[df['phase'] == 'D'].copy()
        df_sys = df[df['phase'] == 'S'].copy()

        df_dia['distance'] = (df_dia['frame'].max() - df_dia['frame']) / self.FRAME_RATE * self.PULLBACK_SPEED
        df_sys['distance'] = (df_sys['frame'].max() - df_sys['frame']) / self.FRAME_RATE * self.PULLBACK_SPEED

        return pd.concat([df_dia, df_sys])

    @staticmethod
    def polynomial_fit(x, y, degree=10):
        p = np.polyfit(x, y, degree)
        return np.polyval(p, x)

    def fit_curves_sys_dia(self, df, degree=10):
        """Fit polynomial curves for systole and diastole phases."""
        df['fitted_lumen_area'] = df.groupby('phase', group_keys=False).apply(
            lambda group: pd.Series(self.polynomial_fit(group['distance'], group['lumen_area'], degree), index=group.index),
            include_groups=False
        )
        df['mean_elliptic_ratio'] = df.groupby('distance')['elliptic_ratio'].transform('mean')
        df['fitted_elliptic_ratio'] = self.polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
        return df.sort_values(by='distance')

    def fit_curve_global(self, df, degree=4):
        """Fit polynomial curves for global data."""
        df['fitted_lumen_area_glob'] = self.polynomial_fit(df['distance'], df['lumen_area'], degree)
        df['mean_elliptic_ratio_glob'] = df['elliptic_ratio'].mean()
        df['fitted_elliptic_ratio_glob'] = self.polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
        return df.sort_values(by='distance')

    def load_data(self, file_path, sep='\t'):
        """Load data from a file."""
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path, sep=sep)

    def process_directory(self, directory):
        """Process all txt files in a directory."""
        print(f"Processing directory: {directory}")
        files = [f for f in os.listdir(directory) if f.endswith('_report.txt')]
        dfs = {}
        for file in files:
            print(f"Processing file: {file}")
            df = self.load_data(os.path.join(directory, file))
            df = self.prep_data(df)
            df = self.fit_curves_sys_dia(df)
            df = self.fit_curve_global(df)
            dfs[file] = df
        return dfs

    def plot_data(self, df_rest, df_dobu):
        """Plot systole and diastole data for rest and stress."""
        print("Plotting data for rest and stress")
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

    def plot_global(self, df_rest, df_dobu):
        print("Plotting global data")
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

    def run(self):
        """Main method to process data and generate plots."""
        print("Starting run method")
        rest_data = self.process_directory(self.rest_dir)
        stress_data = self.process_directory(self.stress_dir)

        print("Rest data files:", rest_data.keys())
        print("Stress data files:", stress_data.keys())

        for rest_file, df_rest in rest_data.items():
            # Find the corresponding stress file
            stress_file = rest_file.replace('PD2EZDBF', 'PD616KK1')
            print(f"Matching {rest_file} with {stress_file}")
            if stress_file in stress_data:
                df_stress = stress_data[stress_file]
                print(f"Plotting data for {rest_file} and {stress_file}")
                self.plot_data(df_rest, df_stress)
                self.plot_global(df_rest, df_stress)
                plt.show()
        print('Hello world!')

# Usage
processor = PressureCurveProcessor(
    rest_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\rest",
    stress_dir=r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_234\stress")
processor.run()



# class IVUSProcessing:
#     def __init__(self, path):
#         self.path = path
#         self.df_rest = None
#         self.df_stress = None
#         self.PULLBACK_SPEED_rest = 1  # mm/s
#         self.START_FRAME_rest = 0
#         self.FRAME_RATE_rest = 30  # frames per second
#         self.PULLBACK_SPEED_stress = 1  # mm/s
#         self.START_FRAME_stress = 0
#         self.FRAME_RATE_stress = 30  # frames per second

#     def __call__(self):
#         self.get_global_variables()
#         self.df_rest = self.prep_data(self.df_rest, self.START_FRAME, self.FRAME_RATE, self.PULLBACK_SPEED)

#     def read_info(self, path):
#         """Read IVUS information from a text file."""
#         info = None
#         for filename in os.listdir(path):
#             if '_report' in filename or '_rest' in filename or '_stress' in filename:
#                 info = pd.read_csv(os.path.join(path, filename), sep='\t')
#                 break  # Exit the loop once the report file is found

#         if info is None:
#             raise FileNotFoundError("No report file found in the specified directory.")

#         return info
    
#     def get_global_variables(self):
#         """Calculates the global variables for the lumen area and elliptic ratio."""
#         if self.df_rest is None:
#             raise ValueError("No dataframes have been created yet. Please run the prep_data method first.")

#         self.PULLBACK_SPEED = self.df_rest['pullback_speed'].unique()[0]
#         self.START_FRAME = self.df_rest['pullback_start_frame'].unique()[0]
#         self.FRAME_RATE = self.df_rest['frame_rate'].unique()[0]
    
#     def prep_data(self, df, start_frame, frame_rate, pullback_speed):
#         """Registers systole and diastole to each other based on the frame number and calculates the distance in mm based on frame rate and pullback speed."""
#         df = df[df['phase'] != '-'].copy()
#         df = df[df['frame'] >= start_frame].copy()

#         df_dia = df[df['phase'] == 'D'].copy()
#         df_sys = df[df['phase'] == 'S'].copy()

#         df_dia.loc[:, 'distance'] = (df_dia['frame'].max() - df_dia['frame']) / frame_rate * pullback_speed
#         df_sys.loc[:, 'distance'] = (df_sys['frame'].max() - df_sys['frame']) / frame_rate * pullback_speed

#         df = pd.concat([df_dia, df_sys])

#         return df


# if __name__ == '__main__':
#     ivus = IVUSProcessing('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files\NARCO_234')
#     ivus()


# df_rest = pd.read_csv(
#     'C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_234_rest.txt', sep='\t'
# )
# df_dobu = pd.read_csv(
#     'C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_234_stress.txt', sep='\t'
# )
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

# df_rest = fit_curves_sys_dia(df_rest)
# df_dobu = fit_curves_sys_dia(df_dobu)

# df_rest = fit_curve_global(df_rest)
# df_dobu = fit_curve_global(df_dobu)

# plot_data(df_rest, df_dobu)
# plot_global(df_rest, df_dobu)

# create_df_from_dir('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports')
