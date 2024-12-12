from os.path import split


# def patient_id_from_dir(directory):
#     """Extract the patient ID from the directory name. by keeping only NARCO_XX"""
#     # remove \rest or \stress from the directory
#     directory = directory.replace("rest", "").replace("stress", "")
#     # split the directory by os separator and get the last part
#     patient_id = os.path.basename(directory)
#     # check if the patient_id starts with 'NARCO_' and return it
#     if patient_id.startswith("NARCO_"):
#         return patient_id
#     else:
#         return None
def patient_id_from_dir(directory: str):
    return split(split(directory)[0])[-1]
