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
        self.prep_data()
        self.save_data()

    def prep_data(self):
        rest = self.output_rest
        stress = self.output_stress

        # Ensure the columns are numeric
        rest['lumen_area'] = pd.to_numeric(rest['lumen_area'], errors='coerce')
        rest['fitted_lumen_area'] = pd.to_numeric(rest['fitted_lumen_area'], errors='coerce')
        stress['lumen_area'] = pd.to_numeric(stress['lumen_area'], errors='coerce')
        stress['fitted_lumen_area'] = pd.to_numeric(stress['fitted_lumen_area'], errors='coerce')

        self.temp_df['patient_id'] = [self.name_dir]
        self.temp_df['rest_phasic_compression_ostium'] = rest[rest['phase'] ==  'S']['lumen_area'].iloc[0] - rest[rest['phase'] ==  'D']['lumen_area'].iloc[0]
        self.temp_df['rest_phasic_compression_mla'] = rest[rest['phase'] ==  'S']['lumen_area'].min() - rest[rest['phase'] ==  'D']['lumen_area'].min()
        self.temp_df['rest_phasic_compression_ostium_fitted'] = rest[rest['phase'] == 'S']['fitted_lumen_area'].iloc[0] - rest[rest['phase'] == 'D']['fitted_lumen_area'].iloc[0]
        self.temp_df['rest_phasic_compression_mla_fitted'] = rest[rest['phase'] ==  'S']['fitted_lumen_area'].min() - rest[rest['phase'] ==  'D']['fitted_lumen_area'].min()
        
        self.temp_df['stress_phasic_compression_ostium'] = stress[stress['phase'] == 'S']['lumen_area'].iloc[0] - stress[stress['phase'] ==  'D']['lumen_area'].iloc[0]
        self.temp_df['stress_phasic_compression_mla'] = stress[stress['phase'] == 'S']['lumen_area'].min() - stress[stress['phase'] ==  'D']['lumen_area'].min()
        self.temp_df['stress_phasic_compression_ostium_fitted'] = stress[stress['phase'] == 'S']['fitted_lumen_area'].iloc[0] - stress[stress['phase'] == 'D']['fitted_lumen_area'].iloc[0]
        self.temp_df['stress_phasic_compression_mla_fitted'] = stress[stress['phase'] == 'S']['fitted_lumen_area'].min() - stress[stress['phase'] ==  'D']['fitted_lumen_area'].min()

        # lateral compression
        self.temp_df['global_lateral_compression_ostium'] = stress['fitted_lumen_area_glob'].iloc[0] - rest['fitted_lumen_area_glob'].iloc[0]
        self.temp_df['global_lateral_compression_mla'] = stress['fitted_lumen_area_glob'].min() - rest['fitted_lumen_area_glob'].min()
        self.temp_df['lateral_compression_ostium_diastole'] = stress[stress['phase'] == 'D']['lumen_area'].iloc[0] - rest[rest['phase'] == 'D']['lumen_area'].iloc[0]
        self.temp_df['lateral_compression_ostium_systole'] = stress[stress['phase'] == 'S']['lumen_area'].iloc[0] - rest[rest['phase'] == 'S']['lumen_area'].iloc[0]
        self.temp_df['lateral_compression_mla_diastole'] = stress[stress['phase'] == 'D']['lumen_area'].min() - rest[rest['phase'] == 'D']['lumen_area'].min()
        self.temp_df['lateral_compression_mla_systole'] = stress[stress['phase'] == 'S']['lumen_area'].min() - rest[rest['phase'] == 'S']['lumen_area'].min()
        self.temp_df['lateral_compression_ostium_diastole_fitted'] = stress[stress['phase'] == 'D']['fitted_lumen_area'].iloc[0] - rest[rest['phase'] == 'D']['fitted_lumen_area'].iloc[0]
        self.temp_df['lateral_compression_ostium_systole_fitted'] = stress[stress['phase'] == 'S']['fitted_lumen_area'].iloc[0] - rest[rest['phase'] == 'S']['fitted_lumen_area'].iloc[0]
        self.temp_df['lateral_compression_mla_diastole_fitted'] = stress[stress['phase'] == 'D']['fitted_lumen_area'].min() - rest[rest['phase'] == 'D']['fitted_lumen_area'].min()
        self.temp_df['lateral_compression_mla_systole_fitted'] = stress[stress['phase'] == 'S']['fitted_lumen_area'].min() - rest[rest['phase'] == 'S']['fitted_lumen_area'].min()

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
            self.output_file = self.output_file.append(self.temp_df, ignore_index=True)

        # Save the updated DataFrame back to the Excel file
        self.output_file.to_excel(os.path.join(self.output_path, "results.xlsx"), index=False)
    

if __name__ == "__main__":
    rest_dir = r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\NARCO_119\."
    output_path = r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\output"
    ivus_data_prep = IVUSDataPrep(rest_dir, output_path)
    ivus_data_prep()