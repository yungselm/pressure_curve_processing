import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None

from scipy.ndimage import gaussian_filter1d

df_rest = pd.read_csv(
    'C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_251_rest.txt', sep='\t'
)
df_dobu = pd.read_csv(
    'C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_251_stress.txt', sep='\t'
)

PULLBACK_SPEED = 1  # mm/s
START_FRAME = 0
FRAME_RATE = 30  # frames per second


def prep_data(df, start_frame, frame_rate, pullback_speed):
    """Registers systole and diastole to each other based on the frame number and calculates the distance in mm based on frame rate and pullback speed."""
    df = df[df['phase'] != '-'].copy()
    df = df[df['frame'] >= start_frame].copy()

    df_dia = df[df['phase'] == 'D'].copy()
    df_sys = df[df['phase'] == 'S'].copy()

    df_dia.loc[:, 'distance'] = (df_dia['frame'].max() - df_dia['frame']) / frame_rate * pullback_speed
    df_sys.loc[:, 'distance'] = (df_sys['frame'].max() - df_sys['frame']) / frame_rate * pullback_speed

    df = pd.concat([df_dia, df_sys])
    return df


def fit_curves_sys_dia(df, degree=10):
    """Fits a polynomial curve to the lumen area and elliptic ratio."""
    df['fitted_lumen_area'] = df.groupby('phase', group_keys=False).apply(
        lambda group: pd.Series(polynomial_fit(group['distance'], group['lumen_area'], degree), index=group.index)
    )
    df['mean_elliptic_ratio'] = df.groupby('distance')['elliptic_ratio'].transform('mean')
    df['fitted_elliptic_ratio'] = polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
    df = df.sort_values(by='distance')

    return df


def fit_curve_global(df, degree=4):
    """Fits a polynomial curve to the lumen area and elliptic ratio."""
    df['fitted_lumen_area_glob'] = polynomial_fit(df['distance'], df['lumen_area'], degree)
    df['mean_elliptic_ratio_glob'] = df['elliptic_ratio'].mean()
    df['fitted_elliptic_ratio_glob'] = polynomial_fit(df['distance'], df['mean_elliptic_ratio'], degree)
    df = df.sort_values(by='distance')

    return df


def polynomial_fit(x, y, degree=10):
    p = np.polyfit(x, y, degree)
    return np.polyval(p, x)


def plot_data(df_rest, df_dobu):
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

    plt.show()

def plot_global(df_rest, df_dobu):
    """Plots fitted_lumen_area_global for rest and dobutamine in the same plot."""
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

    plt.show()

df_rest = prep_data(df_rest, START_FRAME, FRAME_RATE, PULLBACK_SPEED)
df_dobu = prep_data(df_dobu, START_FRAME, FRAME_RATE, PULLBACK_SPEED)

df_rest = fit_curves_sys_dia(df_rest)
df_dobu = fit_curves_sys_dia(df_dobu)

df_rest = fit_curve_global(df_rest)
df_dobu = fit_curve_global(df_dobu)

plot_data(df_rest, df_dobu)
plot_global(df_rest, df_dobu)
