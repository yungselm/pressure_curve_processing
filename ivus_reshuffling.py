import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist
import cv2

# diastolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/NARCO_234/stress/PD616KK1_diastolic.npy')
# systolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/NARCO_234/stress/PD616KK1_systolic.npy')

# diastolic_info = pd.read_csv('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_234_stress.txt', sep='\t')
# diastolic_info = diastolic_info[diastolic_info['phase'] == 'D']

diastolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/NARCO_234/rest/PD2EZDBF_diastolic.npy')
systolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/NARCO_234/rest/PD2EZDBF_systolic.npy')

diastolic_info = pd.read_csv('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/000_Reports/NARCO_234_rest.txt', sep='\t')
diastolic_info = diastolic_info[diastolic_info['phase'] == 'D']

# diastolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/PD6IBR6T_diastolic.npy')
# systolic_frames = np.load('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/PD6IBR6T_systolic.npy')

# diastolic_info = pd.read_csv('C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/test_files/PD6IBR6T_report.txt', sep='\t')
# diastolic_info = diastolic_info[diastolic_info['phase'] == 'D']

def crop_image(image, x1, x2, y1, y2):
    return image[x1:x2, y1:y2]

# Create a new array for the cropped images
cropped_diastolic_frames = np.zeros((diastolic_frames.shape[0], 462, 462))

# Crop all images to the same size in diastolic_frames, by removing 50 from every side
for idx, frame in enumerate(diastolic_frames):
    cropped_diastolic_frames[idx] = crop_image(frame, 50, 512, 25, 487)

# Update diastolic_frames to the cropped version
diastolic_frames = cropped_diastolic_frames


def rotate_image(image, angle):
    """Rotate an image by a specified angle."""
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

def max_correlation(slice1, slice2, rotation_step=10):
    """Compute the maximum correlation between two frames with rotation adjustment."""
    max_corr = -np.inf
    best_angle = 0
    for angle in range(0, 360, rotation_step):
        rotated_slice2 = rotate_image(slice2, angle)
        corr = np.corrcoef(slice1.ravel(), rotated_slice2.ravel())[0, 1]
        if corr > max_corr:
            max_corr = corr
            best_angle = angle
    return max_corr, best_angle

def compute_correlation_matrix(frames):
    """Compute a pairwise correlation matrix for the frames."""
    n_frames = frames.shape[0]
    corr_matrix = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            corr = np.corrcoef(frames[i].ravel(), frames[j].ravel())[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    return corr_matrix

def greedy_path(corr_matrix):
    """Find a greedy path through the frames based on the correlation matrix."""
    n_frames = corr_matrix.shape[0]
    visited = [False] * n_frames
    path = [n_frames - 1]  # Start with the last frame (ostium fixed)
    visited[-1] = True

    for _ in range(n_frames - 1):
        last = path[-1]
        remaining = [(idx, corr_matrix[last, idx]) for idx in range(n_frames) if not visited[idx]]
        next_frame = max(remaining, key=lambda x: x[1])[0]  # Choose the most correlated
        path.append(next_frame)
        visited[next_frame] = True

    return path

def compute_correlation_matrix_with_rotation(frames, rotation_step=10):
    """Compute a pairwise correlation matrix for the frames with rotation adjustment."""
    n_frames = frames.shape[0]
    corr_matrix = np.zeros((n_frames, n_frames))
    rotation_matrix = np.zeros((n_frames, n_frames))  # Optional: track best rotation angles

    # Total number of comparisons
    total_comparisons = n_frames * (n_frames - 1) // 2

    with tqdm(total=total_comparisons, desc="Computing Correlation Matrix") as pbar:
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                corr, angle = max_correlation(frames[i], frames[j], rotation_step)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                rotation_matrix[i, j] = angle  # Track best rotation angle
                rotation_matrix[j, i] = -angle  # Opposite for the reverse comparison
                pbar.update(1)  # Increment progress bar

    return corr_matrix, rotation_matrix

# Compute the updated correlation matrix
correlation_matrix, rotation_matrix = compute_correlation_matrix_with_rotation(diastolic_frames)

# Use the greedy path function as before
sorted_indices = greedy_path(correlation_matrix)

# Use sorted indices to reorder frames and info
sorted_diastolic_frames = diastolic_frames[sorted_indices]
sorted_diastolic_info = diastolic_info.iloc[sorted_indices].reset_index(drop=True)

# flip the sorted_diastolic_frames
sorted_diastolic_frames = sorted_diastolic_frames[::-1]
sorted_diastolic_info = sorted_diastolic_info[::-1]

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

last_frame_index = len(diastolic_frames) - 1
best_angles = [rotation_matrix[last_frame_index, i] for i in range(len(diastolic_frames))]
plt.plot(best_angles, label='Best Rotation Angles')
plt.xlabel('Frame Index')
plt.ylabel('Angle (degrees)')
plt.title('Best Rotation Angles for Last Frame')
plt.legend()
plt.show()
