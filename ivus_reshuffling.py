import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

diastolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/Rest/PDBHSCIO_diastolic.npy')
systolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/Rest/PDBHSCIO_systolic.npy')

diastolic_info = pd.read_csv('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/Rest/PDBHSCIO_report.txt', sep='\t')
diastolic_info = diastolic_info[diastolic_info['phase'] == 'D']

# diastolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/PD6IBR6T_diastolic.npy')
# systolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/PD6IBR6T_systolic.npy')

# diastolic_info = pd.read_csv('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/PD6IBR6T_report.txt', sep='\t')
# diastolic_info = diastolic_info[diastolic_info['phase'] == 'D']

def crop_image(image, x1, x2, y1, y2):
    return image[x1:x2, y1:y2]

# Create a new array for the cropped images
cropped_diastolic_frames = np.zeros((diastolic_frames.shape[0], 412, 412))

# Crop all images to the same size in diastolic_frames, by removing 50 from every side
for idx, frame in enumerate(diastolic_frames):
    cropped_diastolic_frames[idx] = crop_image(frame, 50, 462, 50, 462)

# Update diastolic_frames to the cropped version
diastolic_frames = cropped_diastolic_frames

# diastolic_frames is a (39, 512, 512) array. I want to get every slice of the 39 seperately and give them a index, then the last slide can stay, and the others should be reshuffled. based on which is correlated the most to the last slice
def correlation(slice1, slice2):
    corr = np.corrcoef(slice1.ravel(), slice2.ravel())[0, 1]
    return corr

last_slice = diastolic_frames[-1]

indices_to_test = np.array(range(len(diastolic_frames) - 1))
indices_to_test = list(indices_to_test)

# Initialize the new sorted indices list with the index of the last slice
sorted_indices = [len(diastolic_frames) - 1]

# Loop until all slices are sorted
while indices_to_test:
    corrs = []
    for idx in indices_to_test:
        corrs.append(correlation(diastolic_frames[idx], last_slice))
    
    max_corr_index = np.argmax(corrs)
    last_slice = diastolic_frames[max_corr_index]
    sorted_indices.append(indices_to_test[max_corr_index])
    indices_to_test.pop(max_corr_index)

# Create the sorted array using the sorted indices
sorted_diastolic_frames = diastolic_frames[sorted_indices]
# switch order of the sorted_diastolic_frames
sorted_diastolic_frames = np.flip(sorted_diastolic_frames, axis=0)

# Reverse the sorted indices to match the flipped sorted_diastolic_frames
sorted_indices = sorted_indices[::-1]

# get a reordered list of the diastolic_info
sorted_diastolic_info = diastolic_info.iloc[sorted_indices].reset_index(drop=True)

# plot comparison of lumen_area from the diastolic_info versus the sorted_diastolic_info
plt.figure(figsize=(10, 5))
plt.plot(diastolic_info['lumen_area'].values, label='Original')
plt.plot(sorted_diastolic_info['lumen_area'].values, label='Sorted')
plt.legend()
plt.xlabel('Frame Index')
plt.ylabel('Lumen Area')
plt.title('Comparison of Lumen Area Before and After Sorting')
plt.show()

# plot every image in sorted_diastolic_frames
num_frames = len(sorted_diastolic_frames)
cols = 8
rows = (num_frames + cols - 1) // cols

fig, ax = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
for idx, frame in enumerate(sorted_diastolic_frames):
    ax[idx // cols, idx % cols].imshow(frame, cmap='gray')
    ax[idx // cols, idx % cols].set_title(f'Frame {idx}')
    ax[idx // cols, idx % cols].axis('off')

# Hide any unused subplots
for i in range(num_frames, rows * cols):
    ax[i // cols, i % cols].axis('off')

plt.show()

# plot every image in diastolic_frames
num_frames = len(diastolic_frames)
cols = 8
rows = (num_frames + cols - 1) // cols

fig, ax = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
for idx, frame in enumerate(diastolic_frames):
    ax[idx // cols, idx % cols].imshow(frame, cmap='gray')
    ax[idx // cols, idx % cols].set_title(f'Frame {idx}')
    ax[idx // cols, idx % cols].axis('off')

# Hide any unused subplots
for i in range(num_frames, rows * cols):
    ax[i // cols, i % cols].axis('off')

plt.show()