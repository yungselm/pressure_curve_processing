import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import TextBox, CheckButtons
from tqdm import tqdm


class Reshuffling:
    def __init__(self, path, plot=False):
        self.path = path
        self.diastolic_frames, self.systolic_frames = self.load_frames(path)
        self.diastolic_info, self.systolic_info = self.read_info(path)
        self.sorted_diastolic_indices = None
        self.sorted_systolic_indices = None
        self.plot_true = plot

    def __call__(self):
        x1, x2, y1, y2 = 50, 512, 25, 487
        self.diastolic_frames = self.diastolic_frames[:, x1:x2, y1:y2]
        self.systolic_frames = self.systolic_frames[:, x1:x2, y1:y2]
        self.refresh_plot(self.diastolic_frames, "diastole",
                          save=os.path.join(self.path, "pre_sorted_diastolic_frames.png"))
        self.refresh_plot(self.systolic_frames, "systole",
                          save=os.path.join(self.path, "pre_sorted_systolic_frames.png"))

        (
            self.sorted_diastolic_indices,
            self.sorted_diastolic_frames,
            self.sorted_diastolic_info,
            self.diastolic_correlation_matrix,
            self.diastolic_rotation_matrix,
        ) = self.reshuffle(self.diastolic_frames, self.diastolic_info, 'diastolic')
        (
            self.sorted_systolic_indices,
            self.sorted_systolic_frames,
            self.sorted_systolic_info,
            self.systolic_correlation_matrix,
            self.systolic_rotation_matrix,
        ) = self.reshuffle(self.systolic_frames, self.systolic_info, 'systolic')
        self.rearrange_info_and_save(name="pre_combined")
        if self.plot_true:
            self.sorted_diastolic_frames, self.sorted_diastolic_indices = self.plot_images(self.sorted_diastolic_frames,  title='Diastolic Frames')
            self.sorted_systolic_frames, self.sorted_systolic_indices = self.plot_images(self.sorted_systolic_frames, title='Systolic Frames')
            # Update sorted frames and info based on rearranged indices
            columns_to_rearrange = [col for col in self.diastolic_info.columns if col != 'position']
            self.sorted_diastolic_info[columns_to_rearrange] = self.sorted_diastolic_info[columns_to_rearrange].values[
                self.sorted_diastolic_indices]
            self.sorted_systolic_info[columns_to_rearrange] = self.sorted_systolic_info[columns_to_rearrange].values[
                self.sorted_systolic_indices]
            # self.sorted_diastolic_info = self.sorted_diastolic_info[::-1].reset_index(drop=True)
            # self.sorted_systolic_info = self.sorted_systolic_info[::-1].reset_index(drop=True)

            self.refresh_plot(self.sorted_diastolic_frames, "diastole", save=os.path.join(self.path, "sorted_diastolic_frames.png"))
            self.refresh_plot(self.sorted_systolic_frames, "systole", save=os.path.join(self.path, "sorted_systolic_frames.png"))

            self.plot_correlation_matrix(self.diastolic_correlation_matrix, title='diastolic_correlation_matrix')
            self.plot_correlation_matrix(self.systolic_correlation_matrix, title='systolic_correlation_matrix')
            self.rearrange_info_and_save()
            self.plot_comparison(self.diastolic_info, self.sorted_diastolic_info, suffix="_diastole")
            self.plot_comparison(self.systolic_info, self.sorted_systolic_info, suffix="_systole")
        else:
            self.rearrange_info_and_save()

        return self.sorted_diastolic_frames, self.sorted_systolic_frames

    @staticmethod
    def load_frames(path):
        """Load IVUS frames from a directory."""
        diastolic_frames = None
        systolic_frames = None
        for filename in os.listdir(path):
            if filename.endswith('_diastolic.npy'):
                diastolic_frames = np.load(os.path.join(path, filename))
            elif filename.endswith('_systolic.npy'):
                systolic_frames = np.load(os.path.join(path, filename))
            else:
                continue

        if diastolic_frames is None or systolic_frames is None:
            raise FileNotFoundError("Diastolic or systolic frames not found in the specified directory.")

        return diastolic_frames, systolic_frames

    @staticmethod
    def read_info(path):
        """Read IVUS information from a text file."""
        info = None
        for filename in os.listdir(path):
            if '_report' in filename:
                info = pd.read_csv(os.path.join(path, filename), sep='\t')
                break  # Exit the loop once the report file is found

        if info is None:
            raise FileNotFoundError("No report file found in the specified directory.")

        info_dia = info[info['phase'] == 'D'].reset_index(drop=True)
        info_sys = info[info['phase'] == 'S'].reset_index(drop=True)

        return info_dia, info_sys

    # def crop_images(self, frames, x1, x2, y1, y2):
    #     """Crop all images in a set of frames to the same size."""
    #     # cropped_frames = np.zeros((frames.shape[0], x2 - x1, y2 - y1))
    #     # for idx, frame in enumerate(frames):
    #     #     cropped_frames[idx] = frame[x1:x2, y1:y2]
    #     cropped_frames = frames[:, x1:x2, y1:y2]
    #     return cropped_frames

    def rotate_image(self, image, angle):
        """Rotate an image by a specified angle."""
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_image

    def max_correlation(self, slice1, slice2, rotation_step=10):
        """Compute the maximum correlation between two frames with rotation adjustment."""
        max_corr = -np.inf
        best_angle = 0
        for angle in range(0, 360, rotation_step):
            rotated_slice2 = self.rotate_image(slice2, angle)
            corr = np.corrcoef(slice1.ravel(), rotated_slice2.ravel())[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_angle = angle
        return max_corr, best_angle

    def compute_correlation_matrix(self, frames):
        """Compute a pairwise correlation matrix for the frames."""
        n_frames = frames.shape[0]
        corr_matrix = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                corr = np.corrcoef(frames[i].ravel(), frames[j].ravel())[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        return corr_matrix

    @staticmethod
    def greedy_path(corr_matrix):
        """Find a greedy path through the frames based on the correlation matrix."""
        n_frames = corr_matrix.shape[0]
        if n_frames == 0:
            raise ValueError("Correlation matrix is empty.")

        visited = [False] * n_frames
        path = [n_frames - 1]  # Start with the last frame (ostium fixed)
        visited[-1] = True

        for _ in range(n_frames - 1):
            last = path[-1]
            remaining = [(idx, corr_matrix[last, idx]) for idx in range(n_frames) if not visited[idx]]
            next_frame = max(remaining, key=lambda x: x[1])[0]
            path.append(next_frame)
            visited[next_frame] = True

        return path

    def compute_correlation_matrix_with_rotation(self, frames, rotation_step=10):
        """Compute a pairwise correlation matrix for the frames with rotation adjustment."""
        n_frames = frames.shape[0]
        if n_frames == 0:
            raise ValueError("No frames to process.")

        corr_matrix = np.zeros((n_frames, n_frames))
        rotation_matrix = np.zeros((n_frames, n_frames))

        total_comparisons = n_frames * (n_frames - 1) // 2

        with tqdm(total=total_comparisons, desc="Computing Correlation Matrix") as pbar:
            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    corr, angle = self.max_correlation(frames[i], frames[j], rotation_step)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    rotation_matrix[i, j] = angle
                    rotation_matrix[j, i] = -angle
                    pbar.update(1)

        return corr_matrix, rotation_matrix

    def reshuffle(self, frames, info, phase):
        corr_matrix_file = os.path.join(self.path, f'{phase}_correlation_matrix.npy')
        rotation_matrix_file = os.path.join(self.path, f'{phase}_rotation_matrix.npy')

        if os.path.exists(corr_matrix_file) and os.path.exists(rotation_matrix_file):
            correlation_matrix = np.load(corr_matrix_file)
            rotation_matrix = np.load(rotation_matrix_file)
        else:
            correlation_matrix, rotation_matrix = self.compute_correlation_matrix_with_rotation(frames)
            np.save(corr_matrix_file, correlation_matrix)
            np.save(rotation_matrix_file, rotation_matrix)

        sorted_indices = self.greedy_path(correlation_matrix)
        sorted_frames = frames[sorted_indices]

        # Rearrange all columns except 'position'
        columns_to_rearrange = [col for col in info.columns if col != 'position']
        sorted_info = info.copy()
        sorted_info[columns_to_rearrange] = info[columns_to_rearrange].values[sorted_indices]

        sorted_frames = sorted_frames[::-1]
        sorted_info = sorted_info[::-1].reset_index(drop=True)
        # print(sorted_info )
        return sorted_indices, sorted_frames, sorted_info, correlation_matrix, rotation_matrix

    def plot_comparison(self, info, sorted_info, suffix: str):
        plt.figure(figsize=(10, 5))
        plt.plot(info['lumen_area'].values, label='Original')
        plt.plot(sorted_info['lumen_area'].values, label='Sorted')
        plt.legend()
        plt.xlabel('Frame Index')
        plt.ylabel('Lumen Area')
        plt.title('Comparison of Lumen Area Before and After Sorting')
        plt.savefig(os.path.join(self.path, f'lumen_area_comparison{suffix}.png'))

    @staticmethod
    def refresh_plot(sorted_frames, title, frame_to_move: list[int]=None, end_position: list[int]=None, finished: list[bool]=None,
                     save: str = None):
        """Refresh the plot with the current frame order."""
        plt.close('all')  # Close any previous plots

        num_frames = len(sorted_frames)
        cols = 8
        rows = (num_frames + cols - 1) // cols
        fig, ax = plt.subplots(rows, cols, figsize=(20, rows * 2.5))

        for idx, frame in enumerate(sorted_frames):
            ax[idx // cols, idx % cols].imshow(frame, cmap='gray')
            ax[idx // cols, idx % cols].set_title(f'Frame {idx}')
            ax[idx // cols, idx % cols].axis('off')

        for i in range(num_frames, rows * cols):
            ax[i // cols, i % cols].axis('off')

        plt.subplots_adjust(bottom=0.2, top=0.90, hspace=0.3)  # Adjust space for widgets and subplots
        plt.suptitle(title)
        if save:
            plt.savefig(save)
            return
        # Create TextBox for frame to move
        axbox1 = plt.axes([0.1, 0.05, 0.2, 0.05])
        text_box1 = TextBox(axbox1, 'Frame to Move', initial="")

        def update_frame_to_move(text):
            try:
                frame_to_move[0] = int(text)
            except ValueError:
                print("Invalid input for 'Frame to Move'.")

        text_box1.on_submit(update_frame_to_move)

        # Create TextBox for end position
        axbox2 = plt.axes([0.4, 0.05, 0.2, 0.05])
        text_box2 = TextBox(axbox2, 'End Position', initial="")

        def update_end_position(text):
            try:
                end_position[0] = int(text)
            except ValueError:
                print("Invalid input for 'End Position'.")

        text_box2.on_submit(update_end_position)

        # Add CheckButtons for "Plot Finished?"
        axcheck = plt.axes([0.75, 0.05, 0.2, 0.05])
        check_button = CheckButtons(axcheck, ["Plot Finished?"], [finished[0]], )

        def finish_plot(label):
            if label == "Plot Finished?":
                finished[0] = not finished[0]

        check_button.on_clicked(finish_plot)
        if not save:
            plt.show()

    def plot_images(self, frames, title):
        """
        Plot images interactively with text boxes for rearranging frames
        and a checkbox to indicate when plotting is finished.
        """
        sorted_frames = list(frames)  # Convert to a list for easy manipulation
        finished = [False]  # Use a mutable object to track completion state
        # if title == 'Diastolic Frames':
        #     rearranged_indices = self.sorted_diastolic_indices  # Initialize rearranged indices
        # elif title == "Systolic Frames":
        #     rearranged_indices = self.sorted_systolic_indices
        # else:
        #     raise ValueError(f"title: {title} is not correct")
        # Frame indices for reordering
        rearranged_indices = list(range(len(sorted_frames)))
        frame_to_move = [0]
        end_position = [0]

        # Main interactive loop
        while not finished[0]:
            self.refresh_plot(sorted_frames, title, frame_to_move, end_position, finished)

            # Attempt to move the frame if indices are valid
            if 0 <= frame_to_move[0] <= len(sorted_frames) and 0 <= end_position[0] <= len(sorted_frames):
                print(f'rearranged_indices before move: {rearranged_indices}')
                source_index = frame_to_move[0]
                target_index = end_position[0]
                frame = sorted_frames.pop(source_index)
                target_index = (target_index - 1) if target_index > source_index else target_index
                sorted_frames.insert(target_index, frame)

                # Update rearranged_indices accordingly
                index = rearranged_indices.pop(source_index)
                rearranged_indices.insert(target_index, index)
                print(f'rearranged_indices after move: {rearranged_indices}')
            else:
                print("Invalid indices. No changes made.")

        # Update the class attribute with the final rearranged indices
        # if title == 'Diastolic Frames':
        #     self.sorted_diastolic_indices = rearranged_indices
        # elif title == "Systolic Frames":
        #     self.sorted_systolic_indices = rearranged_indices
        # else:
        #     raise ValueError("title is not correct")

        # Return final sorted frames
        print("Final frame order determined.")
        return np.array(sorted_frames), rearranged_indices

    def plot_correlation_matrix(self, correlation_matrix, title='Correlation Matrix'):
        plt.figure(figsize=(8, 8))
        plt.imshow(correlation_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Frame Index')
        plt.ylabel('Frame Index')
        plt.savefig(os.path.join(self.path, f'{title}.png'))

    def rearrange_info_and_save(self, name: str = "combined"):
        # Rearrange self.diastolic_info based on the sorted_diastolic_indices
        # sorted_diastolic_info = self.diastolic_info.copy()
        # sorted_systolic_info = self.systolic_info.copy()
        #
        # # # Rearrange all columns except 'position'
        # columns_to_rearrange = [col for col in self.diastolic_info.columns if col != 'position']
        # self.sorted_diastolic_info[columns_to_rearrange] = self.diastolic_info[columns_to_rearrange].values[
        #     self.sorted_diastolic_indices]
        # self.sorted_systolic_info[columns_to_rearrange] = self.systolic_info[columns_to_rearrange].values[
        #     self.sorted_systolic_indices]

        # Set new index
        # sorted_diastolic_info = sorted_diastolic_info.reset_index(drop=True)
        # sorted_systolic_info = sorted_systolic_info.reset_index(drop=True)

        combined_info_sorted = pd.concat([self.sorted_diastolic_info, self.sorted_systolic_info], axis=0)
        combined_info_original = pd.concat([self.diastolic_info, self.systolic_info], axis=0)
        # Save the rearranged information to a new file
        if self.plot_true:
            combined_info_sorted.to_csv(os.path.join(self.path, f'{name}_sorted_manual.csv'), index=False)
            combined_info_original.to_csv(os.path.join(self.path, f'{name}_original_manual.csv'), index=False)
        else:
            combined_info_sorted.to_csv(os.path.join(self.path, f'{name}_sorted.csv'), index=False)
            combined_info_original.to_csv(os.path.join(self.path, f'{name}_original.csv'), index=False)

        print("Combined info saved successfully.")


if __name__ == '__main__':
    # reshuffeling = Reshuffeling(r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\Rest")
    reshuffeling = Reshuffling(
        "../data/NARCO_122/rest", plot=True
    )
    diastolic_frames_, systolic_frames_ = reshuffeling()
