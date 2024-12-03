import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import patient_id_from_dir


class IVUSDataPrep:
    def __init__(self, rest_dir, output_path):
        self.working_dir = os.path.abspath(os.path.dirname(rest_dir))
        self.output_path = output_path
        self.output_file = None
        self.output_rest = None
        self.output_stress = None
        self.temp_df = pd.DataFrame()
        self.name_dir = patient_id_from_dir(self.working_dir).lower()

    def __call__(self):
        self.output_file = pd.read_excel(os.path.join(self.output_path, "results.xlsx"))
        self.output_rest = pd.read_csv(os.path.join(self.working_dir, 'output_rest.csv'))
        self.output_stress = pd.read_csv(os.path.join(self.working_dir, 'output_stress.csv'))
        self.prep_data('lumen_area')
        self.prep_data('shortest_distance')
        self.save_data()

    def prep_data(self, metric = 'lumen_area'):
        rest = self.output_rest
        stress = self.output_stress

        # Ensure the columns are numeric
        rest[metric] = pd.to_numeric(rest[metric], errors='coerce')
        rest[f'fitted_{metric}'] = pd.to_numeric(rest[f'fitted_{metric}'], errors='coerce')
        stress[metric] = pd.to_numeric(stress[metric], errors='coerce')
        stress[f'fitted_{metric}'] = pd.to_numeric(stress[f'fitted_{metric}'], errors='coerce')

        self.temp_df['patient_id'] = [self.name_dir]
        self.temp_df[f'rest_phasic_compression_ostium_{metric}'] = rest[rest['phase'] ==  'S'][metric].iloc[0] - rest[rest['phase'] ==  'D'][metric].iloc[0]
        self.temp_df[f'rest_phasic_compression_mla_{metric}'] = rest[rest['phase'] ==  'S'][metric].min() - rest[rest['phase'] ==  'D'][metric].min()
        self.temp_df[f'rest_phasic_compression_ostium_fitted_{metric}'] = rest[rest['phase'] == 'S'][f'fitted_{metric}'].iloc[0] - rest[rest['phase'] == 'D'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'rest_phasic_compression_mla_fitted_{metric}'] = rest[rest['phase'] ==  'S'][f'fitted_{metric}'].min() - rest[rest['phase'] ==  'D'][f'fitted_{metric}'].min()
        
        self.temp_df[f'stress_phasic_compression_ostium_{metric}'] = stress[stress['phase'] == 'S'][metric].iloc[0] - stress[stress['phase'] ==  'D'][metric].iloc[0]
        self.temp_df[f'stress_phasic_compression_mla_{metric}'] = stress[stress['phase'] == 'S'][metric].min() - stress[stress['phase'] ==  'D'][metric].min()
        self.temp_df[f'stress_phasic_compression_ostium_fitted_{metric}'] = stress[stress['phase'] == 'S'][f'fitted_{metric}'].iloc[0] - stress[stress['phase'] == 'D'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'stress_phasic_compression_mla_fitted_{metric}'] = stress[stress['phase'] == 'S'][f'fitted_{metric}'].min() - stress[stress['phase'] ==  'D'][f'fitted_{metric}'].min()

        # lateral compression
        self.temp_df[f'global_lateral_compression_ostium_{metric}'] = stress[f'fitted_{metric}_glob'].iloc[0] - rest[f'fitted_{metric}_glob'].iloc[0]
        self.temp_df[f'global_lateral_compression_mla_{metric}'] = stress[f'fitted_{metric}_glob'].min() - rest[f'fitted_{metric}_glob'].min()
        self.temp_df[f'lateral_compression_ostium_diastole_{metric}'] = stress[stress['phase'] == 'D'][metric].iloc[0] - rest[rest['phase'] == 'D'][metric].iloc[0]
        self.temp_df[f'lateral_compression_ostium_systole_{metric}'] = stress[stress['phase'] == 'S'][metric].iloc[0] - rest[rest['phase'] == 'S'][metric].iloc[0]
        self.temp_df[f'lateral_compression_mla_diastole_{metric}'] = stress[stress['phase'] == 'D'][metric].min() - rest[rest['phase'] == 'D'][metric].min()
        self.temp_df[f'lateral_compression_mla_systole_{metric}'] = stress[stress['phase'] == 'S'][metric].min() - rest[rest['phase'] == 'S'][metric].min()
        self.temp_df[f'lateral_compression_ostium_diastole_fitted_{metric}'] = stress[stress['phase'] == 'D'][f'fitted_{metric}'].iloc[0] - rest[rest['phase'] == 'D'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'lateral_compression_ostium_systole_fitted_{metric}'] = stress[stress['phase'] == 'S'][f'fitted_{metric}'].iloc[0] - rest[rest['phase'] == 'S'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'lateral_compression_mla_diastole_fitted_{metric}'] = stress[stress['phase'] == 'D'][f'fitted_{metric}'].min() - rest[rest['phase'] == 'D'][f'fitted_{metric}'].min()
        self.temp_df[f'lateral_compression_mla_systole_fitted_{metric}'] = stress[stress['phase'] == 'S'][f'fitted_{metric}'].min() - rest[rest['phase'] == 'S'][f'fitted_{metric}'].min()

    def save_data(self):
        # Load the existing Excel file
        self.output_file = pd.read_excel(os.path.join(self.output_path, "results.xlsx"))

        # Ensure all columns in temp_df exist in the output_file
        for column in self.temp_df.columns:
            if column not in self.output_file.columns:
                self.output_file[column] = np.nan

        # Find the row corresponding to the patient ID
        patient_row = self.output_file[self.output_file['patient_id'] == self.name_dir]

        if not patient_row.empty:
            # Update the existing row with new data
            for column in self.temp_df.columns:
                self.output_file.loc[self.output_file['patient_id'] == self.name_dir, column] = self.temp_df[column].values[0]
        else:
            # Append the new data if patient ID is not found
            self.output_file = pd.concat([self.output_file, self.temp_df], ignore_index=True)

        # Save the updated DataFrame back to the Excel file
        self.output_file.to_excel(os.path.join(self.output_path, "results.xlsx"), index=False)
    

if __name__ == "__main__":
    rest_dir = r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_119\."
    output_path = r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\output"
    ivus_data_prep = IVUSDataPrep(rest_dir, output_path)
    ivus_data_prep()