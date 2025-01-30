import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Plotting the dynamics of vessel changes for each patient
states = ['rest_dia', 'rest_sys', 'dia_stress', 'sys_stress']
state_labels = ['Diastole Rest', 'Systole Rest', 'Diastole Stress', 'Systole Stress']

def create_var_per_pressure(df, var_name: str) -> pd.DataFrame:
    rest_name_dia = var_name + '_dia_rest'
    rest_name_sys = var_name + '_sys_rest'
    adenosine_name_dia = var_name + '_dia_adenosine'
    adenosine_name_sys = var_name + '_sys_adenosine'
    stress_name_dia = var_name + '_dia_stress'
    stress_name_sys = var_name + '_sys_stress'

    new_name_rest_dia = var_name + '_per_mmHg_rest_dia'
    new_name_rest_sys = var_name + '_per_mmHg_rest_sys'
    new_name_dia_adenosine = var_name + '_per_mmHg_dia_adenosine'
    new_name_sys_adenosine = var_name + '_per_mmHg_sys_adenosine'
    new_name_dia_stress = var_name + '_per_mmHg_dia_stress'
    new_name_sys_stress = var_name + '_per_mmHg_sys_stress'

    df[new_name_rest_dia] = df[rest_name_dia] / df['mean_diastolic_pressure_rest']
    df[new_name_rest_sys] = df[rest_name_sys] / df['mean_systolic_pressure_rest']
    df[new_name_dia_adenosine] = df[adenosine_name_dia] / df['mean_diastolic_pressure_ado']
    df[new_name_sys_adenosine] = df[adenosine_name_sys] / df['mean_systolic_pressure_ado']
    df[new_name_dia_stress] = df[stress_name_dia] / df['mean_diastolic_pressure_dobu']
    df[new_name_sys_stress] = df[stress_name_sys] / df['mean_systolic_pressure_dobu']

    return df

# this prepares data for the changes of ostial and minimal lumen area, for full intramural length look at other script
df_pressure_ffr = pd.read_excel('output/results.xlsx')
df_ivus_full_analysis = pd.read_excel('ivus_ostial_data.xlsx')

# in column NARCO ID change every value to narco_id + cell value
df_ivus_full_analysis['NARCO ID'] = 'narco_' + df_ivus_full_analysis['NARCO ID'].astype(str)  
df_ivus_full_analysis.rename(columns={'NARCO ID': 'patient_id'}, inplace=True)

df_all = pd.merge(df_pressure_ffr, df_ivus_full_analysis, on='patient_id', how='inner')

df_all = create_var_per_pressure(df_all, 'ostial_area')
df_all = create_var_per_pressure(df_all, 'ostial_shortest')

output_dir_patient_plots = 'output/patient_plots/'
if not os.path.exists(output_dir_patient_plots):
    os.makedirs(output_dir_patient_plots)

# Loop through each patient
for patient_id in df_all['patient_id'].unique():
    df_patient = df_all[df_all['patient_id'] == patient_id]

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Plot 1: Scatter plots for pressure and ostial_shortest_per_mmHg
    # Rest data points
    rest_sys = ax1.scatter(df_patient['mean_systolic_pressure_rest'], df_patient['ostial_shortest_per_mmHg_rest_sys'], color='red', label='Rest Sys')
    rest_dia = ax1.scatter(df_patient['mean_diastolic_pressure_rest'], df_patient['ostial_shortest_per_mmHg_rest_dia'], color='blue', label='Rest Dia')
    # Adenosine data points
    ax1.scatter(df_patient['mean_systolic_pressure_ado'], df_patient['ostial_shortest_per_mmHg_sys_adenosine'], color='orange', label='Adenosine Sys')
    ax1.scatter(df_patient['mean_diastolic_pressure_ado'], df_patient['ostial_shortest_per_mmHg_dia_adenosine'], color='green', label='Adenosine Dia')
    # Stress data points
    stress_sys = ax1.scatter(df_patient['mean_systolic_pressure_dobu'], df_patient['ostial_shortest_per_mmHg_sys_stress'], color='darkred', label='Stress Sys')
    stress_dia = ax1.scatter(df_patient['mean_diastolic_pressure_dobu'], df_patient['ostial_shortest_per_mmHg_dia_stress'], color='darkblue', label='Stress Dia')

    # Add connections between rest sys and dia
    ax1.plot([df_patient['mean_systolic_pressure_rest'].values[0], df_patient['mean_diastolic_pressure_rest'].values[0]],
             [df_patient['ostial_shortest_per_mmHg_rest_sys'].values[0], df_patient['ostial_shortest_per_mmHg_rest_dia'].values[0]],
             color='black', linestyle='--', alpha=0.5, label='Rest Sys-Dia Connection')

    # Add connections between stress sys and dia
    ax1.plot([df_patient['mean_systolic_pressure_dobu'].values[0], df_patient['mean_diastolic_pressure_dobu'].values[0]],
             [df_patient['ostial_shortest_per_mmHg_sys_stress'].values[0], df_patient['ostial_shortest_per_mmHg_dia_stress'].values[0]],
             color='black', linestyle='-', alpha=0.5, label='Stress Sys-Dia Connection')

    # Customize ax1
    ax1.set_title(f'Patient ID: {patient_id} - Pressure vs ostial_shortest_per_mmHg')
    ax1.set_xlabel('Pressure')
    ax1.set_ylabel('ostial_shortest_per_mmHg')
    ax1.set_ylim(0, 0.05)
    ax1.set_yticks(np.arange(0, 0.06, 0.005))
    ax1.set_xlim(40, 170)
    ax1.set_xticks(np.arange(40, 170, 10))
    ax1.legend()

    # Plot 2: Line and scatter plots for iFR and pdpa
    try:
        # Extract pressure values, including adenosine if available
        pressure_values = [
            df_patient['iFR_mean_rest'].values[0],
            df_patient['pdpa_mean_rest'].values[0],
            df_patient['iFR_mean_ado'].values[0] if not pd.isna(df_patient['iFR_mean_ado'].values[0]) else None,  # Adenosine iFR (if available)
            df_patient['pdpa_mean_ado'].values[0] if not pd.isna(df_patient['pdpa_mean_ado'].values[0]) else None,  # Adenosine pdpa (if available)
            df_patient['iFR_mean_dobu'].values[0],
            df_patient['pdpa_mean_dobu'].values[0]
        ]
        
        # Plot lines connecting the states (skip adenosine if not available)
        x_labels = ['iFR_mean_rest', 'iFR_mean_dobu']
        y_values = [pressure_values[0], pressure_values[4]]
        if pressure_values[2] is not None:  # If adenosine iFR is available
            x_labels.insert(1, 'iFR_mean_ado')
            y_values.insert(1, pressure_values[2])
        ax2.plot(x_labels, y_values, color='grey', linestyle='--', label='iFR Progression')

        x_labels = ['pdpa_mean_rest', 'pdpa_mean_dobu']
        y_values = [pressure_values[1], pressure_values[5]]
        if pressure_values[3] is not None:  # If adenosine pdpa is available
            x_labels.insert(1, 'pdpa_mean_ado')
            y_values.insert(1, pressure_values[3])
        ax2.plot(x_labels, y_values, color='grey', linestyle='-', label='pdpa Progression')

        # Plot scatter points (skip adenosine if not available)
        scatter_labels = ['iFR_mean_rest', 'pdpa_mean_rest']
        scatter_values = [pressure_values[0], pressure_values[1]]
        scatter_colors = ['blue', 'red']
        if pressure_values[2] is not None:  # If adenosine iFR is available
            scatter_labels.append('iFR_mean_ado')
            scatter_values.append(pressure_values[2])
            scatter_colors.append('green')
        if pressure_values[3] is not None:  # If adenosine pdpa is available
            scatter_labels.append('pdpa_mean_ado')
            scatter_values.append(pressure_values[3])
            scatter_colors.append('darkgreen')
        scatter_labels.extend(['iFR_mean_dobu', 'pdpa_mean_dobu'])
        scatter_values.extend([pressure_values[4], pressure_values[5]])
        scatter_colors.extend(['darkblue', 'darkred'])

        ax2.scatter(scatter_labels, scatter_values, marker='o', color=scatter_colors, label='Data Points')

    except KeyError as e:
        print(f"Missing column for patient {patient_id}: {e}")
        continue  # Skip this patient if a required column is missing

    # Customize the plot
    ax2.set_ylim(0.3, 1.0)
    ax2.set_yticks(np.arange(0.3, 1.05, 0.05))
    ax2.hlines(0.8, xmin=-1, xmax=6, colors='r', linestyles='dotted', label='Threshold 0.8')
    ax2.set_title(f'Patient ID: {patient_id} - iFR and pdpa Progression')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.legend()

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_patient_plots, f'{patient_id}_combined_plot.png'))
    plt.close()

plt.scatter(df_all['mean_systolic_pressure_rest'], df_all['ostial_shortest_per_mmHg_rest_sys'], color='red')
plt.scatter(df_all['mean_diastolic_pressure_rest'], df_all['ostial_shortest_per_mmHg_rest_dia'], color='blue')
plt.scatter(df_all['mean_systolic_pressure_ado'], df_all['ostial_shortest_per_mmHg_sys_adenosine'], color='orange')
plt.scatter(df_all['mean_diastolic_pressure_ado'], df_all['ostial_shortest_per_mmHg_dia_adenosine'], color='green')
plt.scatter(df_all['mean_systolic_pressure_dobu'], df_all['ostial_shortest_per_mmHg_sys_stress'], color='darkred')
plt.scatter(df_all['mean_diastolic_pressure_dobu'], df_all['ostial_shortest_per_mmHg_dia_stress'], color='darkblue')
plt.title('All patients')
plt.xlabel('Pressure')
plt.ylabel('ostial_shortest_per_mmHg')
plt.show()

plt.scatter(df_all['mean_systolic_pressure_rest'], df_all['ostial_area_per_mmHg_rest_sys'], color='red')
plt.scatter(df_all['mean_diastolic_pressure_rest'], df_all['ostial_area_per_mmHg_rest_dia'], color='blue')
plt.scatter(df_all['mean_systolic_pressure_ado'], df_all['ostial_area_per_mmHg_sys_adenosine'], color='orange')
plt.scatter(df_all['mean_diastolic_pressure_ado'], df_all['ostial_area_per_mmHg_dia_adenosine'], color='green')
plt.scatter(df_all['mean_systolic_pressure_dobu'], df_all['ostial_area_per_mmHg_sys_stress'], color='darkred')
plt.scatter(df_all['mean_diastolic_pressure_dobu'], df_all['ostial_area_per_mmHg_dia_stress'], color='darkblue')
plt.title('All patients')
plt.xlabel('Pressure')
plt.ylabel('ostial_area_per_mmHg')
plt.show()

for patient_id in df_all['patient_id'].unique():
    df_patient = df_all[df_all['patient_id'] == patient_id]
    
    # Extract the relevant data for the patient
    ostial_area_per_mmHg = [
        df_patient['ostial_area_dia_rest'].values[0],
        df_patient['ostial_area_sys_rest'].values[0],
        df_patient['ostial_area_dia_stress'].values[0],
        df_patient['ostial_area_sys_stress'].values[0]
    ]
    
    # Plot the line connecting the states
    plt.plot(state_labels, ostial_area_per_mmHg, marker='o', label=f'Patient {patient_id}')

plt.title('Dynamics of Vessel Changes Across Different States')
plt.xlabel('State')
plt.ylabel('Ostial Area per mmHg')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# repeat for normalized ostial area
df_all['ostial_area_compression_rest_dia'] = 0
df_all['ostial_area_compression_rest_sys'] = (1 - df_all['ostial_area_sys_rest'] / df_all['ostial_area_dia_rest']) * 100
df_all['ostial_area_compression_dia_stress'] = (1 - df_all['ostial_area_dia_stress'] / df_all['ostial_area_dia_rest']) * 100
df_all['ostial_area_compression_sys_stress'] = (1 - df_all['ostial_area_sys_stress'] / df_all['ostial_area_dia_rest']) * 100

for patient_id in df_all['patient_id'].unique():
    df_patient = df_all[df_all['patient_id'] == patient_id]
    
    # Extract the relevant data for the patient
    ostial_area_per_mmHg = [
        df_patient['ostial_area_compression_rest_dia'].values[0],
        df_patient['ostial_area_compression_rest_sys'].values[0],
        df_patient['ostial_area_compression_dia_stress'].values[0],
        df_patient['ostial_area_compression_sys_stress'].values[0]
    ]
    
    # Plot the line connecting the states
    plt.plot(state_labels, ostial_area_per_mmHg, marker='o', label=f'Patient {patient_id}')

plt.title('Dynamics of Normalized Vessel Changes Across Different States')
plt.xlabel('State')
plt.ylabel('Percent compression ostial area')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# df_all['ostial_area_change_per_mmHg_rest_dia'] = 0
# df_all['ostial_area_change_per_mmHg_rest_sys'] = (df_all['ostial_area_dia_rest'] - df_all['ostial_area_sys_rest']) / (df_all['mean_diastolic_pressure_rest'] - df_all['mean_systolic_pressure_rest'])
# df_all['ostial_area_change_per_mmHg_dia_stress'] = (df_all['ostial_area_dia_stress'] - df_all['ostial_area_dia_rest']) / (df_all['mean_diastolic_pressure_dobu'] - df_all['mean_diastolic_pressure_rest'])
# df_all['ostial_area_change_per_mmHg_sys_stress'] = (df_all['ostial_area_sys_stress'] - df_all['ostial_area_dia_rest']) / (df_all['mean_systolic_pressure_dobu'] - df_all['mean_diastolic_pressure_rest'])

# for patient_id in df_all['patient_id'].unique():
#     df_patient = df_all[df_all['patient_id'] == patient_id]
    
#     # Extract the relevant data for the patient
#     ostial_area_per_mmHg = [
#         df_patient['ostial_area_change_per_mmHg_rest_dia'].values[0],
#         df_patient['ostial_area_change_per_mmHg_rest_sys'].values[0],
#         df_patient['ostial_area_change_per_mmHg_dia_stress'].values[0],
#         df_patient['ostial_area_change_per_mmHg_sys_stress'].values[0]
#     ]
    
#     # Plot the line connecting the states
#     plt.plot(state_labels, ostial_area_per_mmHg, marker='o', label=f'Patient {patient_id}')

# plt.title('Dynamics of Vessel Changes Across Different States')
# plt.xlabel('State')
# plt.ylabel('Ostial Area Change per mmHg')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()