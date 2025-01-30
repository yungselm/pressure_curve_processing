import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# # for every individual patient id plot 'ostial_shortest_per_mmHg_rest_dia','ostial_shortest_per_mmHg_rest_sys','ostial_shortest_per_mmHg_dia_adenosine','ostial_shortest_per_mmHg_sys_adenosine','ostial_shortest_per_mmHg_dia_stress','ostial_shortest_per_mmHg_sys_stress'
# for patient_id in df_all['patient_id'].unique():
#     df_patient = df_all[df_all['patient_id'] == patient_id]
#     plt.scatter(df_patient['mean_systolic_pressure_rest'], df_patient['ostial_shortest_per_mmHg_rest_sys'], color='red')
#     plt.scatter(df_patient['mean_diastolic_pressure_rest'], df_patient['ostial_shortest_per_mmHg_rest_dia'], color='blue')
#     plt.scatter(df_patient['mean_systolic_pressure_ado'], df_patient['ostial_shortest_per_mmHg_sys_adenosine'], color='orange')
#     plt.scatter(df_patient['mean_diastolic_pressure_ado'], df_patient['ostial_shortest_per_mmHg_dia_adenosine'], color='green')
#     plt.scatter(df_patient['mean_systolic_pressure_dobu'], df_patient['ostial_shortest_per_mmHg_sys_stress'], color='darkred')
#     plt.scatter(df_patient['mean_diastolic_pressure_dobu'], df_patient['ostial_shortest_per_mmHg_dia_stress'], color='darkblue')
#     plt.title(f'Patient ID: {patient_id}')
#     plt.xlabel('Pressure')
#     plt.ylabel('ostial_shortest_per_mmHg')
#     plt.show()

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

# Plotting the dynamics of vessel changes for each patient
states = ['rest_dia', 'rest_sys', 'dia_stress', 'sys_stress']
state_labels = ['Diastole Rest', 'Systole Rest', 'Diastole Stress', 'Systole Stress']

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