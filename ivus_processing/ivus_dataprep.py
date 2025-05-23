import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils import patient_id_from_dir


class IVUSDataPrep:
    def __init__(self, rest_dir, output_path):
        self.working_dir = os.path.abspath(os.path.dirname(rest_dir))
        self.output_path = output_path
        self.output_file = None
        self.output_rest = None
        self.output_stress = None
        self.temp_df = pd.DataFrame()
        self.name_dir = self.patient_id_from_dir(self.working_dir).lower()

    def __call__(self):
        self.output_file = pd.read_excel(os.path.join(self.output_path, "results.xlsx"))
        self.output_rest = pd.read_csv(os.path.join(self.working_dir, 'output_rest.csv'))
        self.output_stress = pd.read_csv(os.path.join(self.working_dir, 'output_stress.csv'))
        self.prep_data('lumen_area')
        self.prep_data('shortest_distance')
        self.save_data()

    def patient_id_from_dir(self, directory):  # Added self as first parameter
        """Extract the patient ID from the directory name. by keeping only NARCO_XX"""
        # Remove \rest or \stress from the directory
        directory = directory.replace("rest", "").replace("stress", "")
        # Split the directory by os separator and get the last part
        patient_id = os.path.basename(directory)
        # Check if the patient_id starts with 'NARCO_' and return it
        if patient_id.startswith("NARCO_"):
            return patient_id
        else:
            return None

    def prep_data(self, metric = 'lumen_area'):
        rest = self.output_rest
        stress = self.output_stress

        # Ensure the columns are numeric
        rest[metric] = pd.to_numeric(rest[metric], errors='coerce')
        rest[f'fitted_{metric}'] = pd.to_numeric(rest[f'fitted_{metric}'], errors='coerce')
        stress[metric] = pd.to_numeric(stress[metric], errors='coerce')
        stress[f'fitted_{metric}'] = pd.to_numeric(stress[f'fitted_{metric}'], errors='coerce')

        self.temp_df['patient_id'] = [self.name_dir]
        self.temp_df[f'ostial_{metric}_rest_systole'] = rest[rest['phase'] ==  'S'][metric].iloc[0]
        self.temp_df[f'ostial_{metric}_rest_diastole'] = rest[rest['phase'] ==  'D'][metric].iloc[0]
        self.temp_df[f'mla_{metric}_rest_systole'] = rest[rest['phase'] ==  'S'][metric].min()
        self.temp_df[f'mla_{metric}_rest_diastole'] = rest[rest['phase'] ==  'D'][metric].min()
        self.temp_df[f'ostial_{metric}_rest_systole_fitted'] = rest[rest['phase'] == 'S'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'ostial_{metric}_rest_diastole_fitted'] = rest[rest['phase'] == 'D'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'mla_{metric}_rest_systole_fitted'] = rest[rest['phase'] ==  'S'][f'fitted_{metric}'].min()
        self.temp_df[f'mla_{metric}_rest_diastole_fitted'] = rest[rest['phase'] ==  'D'][f'fitted_{metric}'].min()
        self.temp_df[f'global_ostial_{metric}_rest'] = rest[rest['phase'] == 'S'][f'fitted_{metric}_glob'].iloc[0]
        self.temp_df[f'global_mla_{metric}_rest'] = rest[rest['phase'] == 'S'][f'fitted_{metric}_glob'].min()

        self.temp_df[f'ostial_{metric}_stress_systole'] = stress[stress['phase'] == 'S'][metric].iloc[0]
        self.temp_df[f'ostial_{metric}_stress_diastole'] = stress[stress['phase'] ==  'D'][metric].iloc[0]
        self.temp_df[f'mla_{metric}_stress_systole'] = stress[stress['phase'] ==  'S'][metric].min()
        self.temp_df[f'mla_{metric}_stress_diastole'] = stress[stress['phase'] ==  'D'][metric].min()
        self.temp_df[f'ostial_{metric}_stress_systole_fitted'] = stress[stress['phase'] == 'S'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'ostial_{metric}_stress_diastole_fitted'] = stress[stress['phase'] == 'D'][f'fitted_{metric}'].iloc[0]
        self.temp_df[f'mla_{metric}_stress_systole_fitted'] = stress[stress['phase'] ==  'S'][f'fitted_{metric}'].min()
        self.temp_df[f'mla_{metric}_stress_diastole_fitted'] = stress[stress['phase'] ==  'D'][f'fitted_{metric}'].min()
        self.temp_df[f'global_ostial_{metric}_stress'] = stress[stress['phase'] == 'S'][f'fitted_{metric}_glob'].iloc[0]
        self.temp_df[f'global_mla_{metric}_stress'] = stress[stress['phase'] == 'S'][f'fitted_{metric}_glob'].min()

        self.temp_df[f'ostial_wallthickness_rest_systole'] = rest[rest['phase'] == 'S']['measurement_1'].iloc[0]
        self.temp_df[f'ostial_wallthickness_rest_diastole'] = rest[rest['phase'] == 'D']['measurement_1'].iloc[0]
        self.temp_df[f'mla_wallthickness_rest_systole'] = rest[rest['phase'] == 'S']['measurement_1'].min()
        self.temp_df[f'mla_wallthickness_rest_diastole'] = rest[rest['phase'] == 'D']['measurement_1'].min()
        self.temp_df[f'ostial_wallthickness_stress_systole'] = stress[stress['phase'] == 'S']['measurement_1'].iloc[0]
        self.temp_df[f'ostial_wallthickness_stress_diastole'] = stress[stress['phase'] == 'D']['measurement_1'].iloc[0]
        self.temp_df[f'mla_wallthickness_stress_systole'] = stress[stress['phase'] == 'S']['measurement_1'].min()
        self.temp_df[f'mla_wallthickness_stress_diastole'] = stress[stress['phase'] == 'D']['measurement_1'].min()

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
    rest_dir = r"D:\00_coding\pressure_curve_processing\ivus\NARCO_210\."
    output_path = r"D:\00_coding\pressure_curve_processing\output"
    ivus_data_prep = IVUSDataPrep(rest_dir, output_path)
    ivus_data_prep()