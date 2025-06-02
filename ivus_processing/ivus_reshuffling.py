import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import TextBox, CheckButtons
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff

class Reshuffling:
    def __init__(self, path, resolution=0.01755671203136,  plot=False):
        self.path = path
        self.csv_path = None
        self.resolution = resolution
        self.plot_true = plot
        self.diastolic_frames, self.systolic_frames = self.load_frames(path)
        self.diastolic_frames_og = self.diastolic_frames.copy()
        self.systolic_frames_og = self.systolic_frames.copy()
        self.diastolic_info, self.systolic_info = self.read_info(path)
        self.dia_contours, self.dia_refpts = self.load_contours('diastolic')
        self.sys_contours, self.sys_refpts = self.load_contours('systolic')
        self.dia_contours_og = self.dia_contours.copy()
        self.sys_contours_og = self.sys_contours.copy()
        self.sorted_diastolic_indices = []
        self.sorted_systolic_indices = []

    def __call__(self):
        x_min, x_max = 25, 487
        y_min, y_max = 25, 487 # cropping images to get rid of frame logo and remove unnecessary info
        # embed contours and ref point on diastolic_frames
        self.diastolic_frames = self.embed_contours(
            self.diastolic_frames,
            self.dia_contours,
            color=(255, 255, 0),  # yellow in RGB
            thickness=3           # increase “intensity” by drawing a fatter line
        )
        self.systolic_frames = self.embed_contours(
            self.systolic_frames,
            self.sys_contours,
            color=(255, 255, 0),
            thickness=3
        )
        self.diastolic_frames = self.diastolic_frames[:, y_min:y_max, x_min:x_max]
        self.systolic_frames = self.systolic_frames[:, y_min:y_max, x_min:x_max]

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
        ) = self.reshuffle(self.diastolic_frames, self.dia_contours, self.diastolic_info, 'diastolic')
        (
            self.sorted_systolic_indices,
            self.sorted_systolic_frames,
            self.sorted_systolic_info,
            self.systolic_correlation_matrix,
            self.systolic_rotation_matrix,
        ) = self.reshuffle(self.systolic_frames, self.sys_contours, self.systolic_info, 'systolic')

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

    def embed_contours(
        self,
        frames: np.ndarray,
        contours: dict[int, np.ndarray],
        color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Burn each contour into its frame as a colored polyline:
          - frames: NxHxW uint8 (grayscale) array
          - contours: dict mapping frame idx -> (Mx2) array of [x,y] points
          - color: (R, G, B), e.g. yellow = (255,255,0)
          - thickness: line thickness in pixels
        Returns NxHxWx3 uint8 array.
        """
        N, H, W = frames.shape
        # convert to color
        frames_color = np.stack([frames]*3, axis=-1)

        for idx, pts in contours.items():
            # round to ints
            xy = np.round(pts).astype(int)
            x, y = xy[:, 0], xy[:, 1]
            # flip vertically
            y = H - 1 - y
            # clamp
            x = np.clip(x, 0, W - 1)
            y = np.clip(y, 0, H - 1)

            # reshape for cv2.polylines: (M,1,2) in (x,y) order
            poly = np.stack([x, y], axis=1).reshape(-1, 1, 2)
            # draw closed contour
            cv2.polylines(
                frames_color[idx],
                [poly],
                isClosed=True,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA
            )

        return frames_color

    def load_contours(self, phase):
        """
        Given phase='diastolic' or 'systolic', looks for a subfolder
        "<phase>_csv_files" containing:
          - {phase}_contours.csv
          - {phase}_reference_points.csv
        Returns two dicts mapping frame_index -> ndarray of shape (N,2)
        """
        # 1) locate the *_csv_files folder
        csv_dirs = [
            d for d in os.listdir(self.path)
            if d.endswith("_csv_files") and os.path.isdir(os.path.join(self.path, d))
        ]
        if not csv_dirs:
            raise FileNotFoundError(f"No '*_csv_files' folder found under {self.path}")
        # if there are multiple, you could pick the one containing phase in its name
        csv_dir = None
        for d in csv_dirs:
            if phase in d:
                csv_dir = d
                break
        if csv_dir is None:
            # fallback to the first one
            csv_dir = csv_dirs[0]

        full_csv_dir = os.path.join(self.path, csv_dir)
        self.csv_path = full_csv_dir
        contour_file = os.path.join(full_csv_dir, f"{phase}_contours.csv")
        refpts_file  = os.path.join(full_csv_dir, f"{phase}_reference_points.csv")
        def _read(fname):
            # Force a 2D array even if there's only one row
            raw = np.loadtxt(fname, delimiter='\t')
            if raw.ndim == 0:
                # single value → no valid rows
                return {}
            if raw.ndim == 1:
                # exactly one row: turn shape (4,) → (1,4)
                raw = raw[None, :]

            mm_to_px = 1.0 / self.resolution

            out: dict[int, list[list[float]]] = {}
            for idx, x_mm, y_mm, _z in raw:
                i = int(idx)
                x_px, y_px = x_mm * mm_to_px, y_mm * mm_to_px
                out.setdefault(i, []).append([x_px, y_px])

            # convert lists → arrays
            return {k: np.array(v) for k, v in out.items()}
        
        if f"{phase}_contours.npz" in os.listdir(full_csv_dir):
            # load and convert to normal dict
            npzfile = np.load(os.path.join(full_csv_dir, f"{phase}_contours.npz"))
            contours = {
                int(k.split('_',1)[1]): v
                for k, v in npzfile.items()
            }
            # read reference points as before
            ref_points = _read(refpts_file)
            ref_points = {k: v for k, v in ref_points.items() if k in contours}

            return contours, ref_points

        else:
            contours = _read(contour_file)
            # Map original indices to new consecutive indices 0..N-1
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(contours.keys()))}
            # Remap contours to new indices
            contours = {old_to_new[k]: v for k, v in contours.items()}

            ref_points = _read(refpts_file)
            # Remap ref_points to new indices, only if the index exists in contours
            ref_points = {old_to_new[k]: v for k, v in ref_points.items() if k in old_to_new}

            return contours, ref_points

    @staticmethod
    def load_frames(path) -> tuple[np.ndarray, np.ndarray]:
        """Load IVUS frames from a directory."""
        diastolic_frames = None
        systolic_frames = None
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if filename == 'pre_combined_diastolic_sorted.npy' and os.path.isfile(full_path):
                diastolic_frames = np.load(full_path)
            elif filename == 'pre_combined_systolic_sorted.npy' and os.path.isfile(full_path):
                systolic_frames = np.load(full_path)
            elif filename.endswith('_diastolic.npy') and os.path.isfile(full_path) and diastolic_frames is None:
                diastolic_frames = np.load(full_path)
            elif filename.endswith('_systolic.npy') and os.path.isfile(full_path) and systolic_frames is None:
                systolic_frames = np.load(full_path)
            else:
                continue

        if diastolic_frames is None or systolic_frames is None:
            raise FileNotFoundError("Diastolic or systolic frames not found in the specified directory.")

        return diastolic_frames, systolic_frames

    @staticmethod
    def read_info(path) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def rotate_image(self, image, angle):
        """Rotate an image by a specified angle."""
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_image

    def _normalize_array(self, arr: np.array):
        mean_arr = np.mean(arr)
        std_arr = np.std(arr)
        return (arr - mean_arr) / std_arr

    def max_correlation(self, slice1, slice2, rotation_step=10):
        """Compute the maximum correlation between two frames with rotation adjustment."""
        if slice1.ndim == 3 and slice1.shape[2] == 3:
            slice1 = slice1[..., 0]
        if slice2.ndim == 3 and slice2.shape[2] == 3:
            slice2 = slice2[..., 0]
        slice1 = self._normalize_array(slice1)
        weight_map = self.weigth_map_gaussian()
        weighted_slice1 = slice1 * weight_map
        
        max_corr = -np.inf
        best_angle = 0

        for angle in range(0, 360, rotation_step):
            rotated_slice2 = self.rotate_image(slice2, angle)
            rotated_slice2 = self._normalize_array(rotated_slice2)

            weighted_slice2 = rotated_slice2 * weight_map
            corr = np.corrcoef(weighted_slice1.ravel(), weighted_slice2.ravel())[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_angle = angle
        return max_corr, best_angle
    
    @staticmethod
    def weigth_map_gaussian(sigma=175.0):
        height, width = 462, 462

        x = np.linspace(0, height - 1, height)
        y = np.linspace(0, width - 1, width)
        x, y = np.meshgrid(x, y)

        x0 = (width - 1) / 2
        y0 = (height - 1) / 2

        gaussian_2d = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        # Normalize so the peak is exactly 1
        gaussian_2d /= gaussian_2d.max()
        # set catheter region to 0
        gaussian_2d[203:259, 203:259] = 0

        return gaussian_2d

    # def compute_correlation_matrix(self, frames):
    #     """Compute a pairwise correlation matrix for the frames."""
    #     n_frames = frames.shape[0]
    #     corr_matrix = np.zeros((n_frames, n_frames))
    #     for i in range(n_frames):
    #         for j in range(i + 1, n_frames):
    #             corr = np.corrcoef(frames[i].ravel(), frames[j].ravel())[0, 1]
    #             corr_matrix[i, j] = corr
    #             corr_matrix[j, i] = corr
    #     return corr_matrix

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

    def compute_correlation_matrix_with_rotation(self, frames, contours, rotation_step=10):
        """Compute a pairwise correlation matrix for the frames with rotation adjustment."""
        n_frames = frames.shape[0]
        if n_frames == 0:
            raise ValueError("No frames to process.")

        corr_matrix = np.zeros((n_frames, n_frames))
        rotation_matrix = np.zeros((n_frames, n_frames))

        total_comparisons = n_frames * (n_frames - 1) // 2

        alpha = 0.3

        keys = sorted(contours.keys())
        first_idx = keys[0]
        last_idx = keys[-1]
        D_max, _, _ = directed_hausdorff(contours[first_idx], contours[last_idx])

        with tqdm(total=total_comparisons, desc="Computing Correlation Matrix") as pbar:
            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    corr, angle = self.max_correlation(frames[i], frames[j], rotation_step)
                    dist, angle_hausdorff = self.min_hausdorff(contours[i], contours[j], rotation_step)

                    # 1) normalize both metrics to [0,1]
                    s_corr = (corr + 1) / 2
                    d_norm = min(dist, D_max) / D_max
                    s_haus = 1 - d_norm

                    # 2) weighted combination
                    combined_score = alpha * s_corr + (1 - alpha) * s_haus

                    # 3) store combined_score in your matrix instead of raw corr
                    corr_matrix[i, j] = combined_score
                    corr_matrix[j, i] = combined_score

                    # still keep rotation info if you want
                    rotation_matrix[i, j] = angle
                    rotation_matrix[j, i] = -angle

                    pbar.update(1)

        return corr_matrix, rotation_matrix

    def min_hausdorff(self, contour1, contour2, rotation_step=10):
        min_hausdorff = np.inf
        best_angle = 0
        for angle in range(0, 360, rotation_step):
            rotated_contour2 = self.rotate_contour(contour2, angle)
            hausdorff, _, _ = directed_hausdorff(contour1, rotated_contour2)
            if hausdorff < min_hausdorff:
                min_hausdorff = hausdorff
                best_angle = angle
        return min_hausdorff, best_angle

    def rotate_contour(self, contour, angle, image_shape=(512, 512)):
        """
        Rotate a contour (Nx2 or Nx3 array: [x, y] or [x, y, z]) by a specified angle (degrees)
        around the center of the image.

        Args:
            contour: np.ndarray of shape (N, 2) or (N, 3) or (N, 4) (idx, x, y, z or just x, y)
            angle: float, rotation angle in degrees (counterclockwise)
            image_shape: tuple, (height, width) of the image

        Returns:
            rotated_contour: np.ndarray of same shape as input
        """
        # Extract x, y coordinates
        if contour.shape[1] == 4:
            # (idx, x, y, z)
            x = contour[:, 1]
            y = contour[:, 2]
            rest = contour[:, [0, 3]]
        elif contour.shape[1] == 3:
            # (x, y, z)
            x = contour[:, 0]
            y = contour[:, 1]
            rest = contour[:, 2:]
        else:
            # (x, y)
            x = contour[:, 0]
            y = contour[:, 1]
            rest = None

        # Center of rotation
        h, w = image_shape
        center = np.array([w / 2, h / 2])

        # Convert angle to radians
        theta = np.deg2rad(angle)
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Shift to origin, rotate, shift back
        coords = np.stack([x, y], axis=1) - center
        rotated_coords = coords @ rot_mat.T + center

        # Reconstruct output
        if rest is not None:
            if rest.shape[1] == 2:
                # (idx, x, y, z)
                rotated = np.column_stack([rest[:, 0], rotated_coords, rest[:, 1]])
            else:
                # (x, y, z)
                rotated = np.column_stack([rotated_coords, rest])
        else:
            rotated = rotated_coords

        return rotated

    def reshuffle(self, frames, contours, info, phase):
        corr_matrix_file = os.path.join(self.path, f'{phase}_correlation__hausdorff_matrix.npy')
        rotation_matrix_file = os.path.join(self.path, f'{phase}_rotation__hausdorff_matrix.npy')

        if os.path.exists(corr_matrix_file) and os.path.exists(rotation_matrix_file):
            correlation_matrix = np.load(corr_matrix_file)
            rotation_matrix = np.load(rotation_matrix_file)
        else:
            correlation_matrix, rotation_matrix = self.compute_correlation_matrix_with_rotation(frames, contours)
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
        plt.close()

    def plot_images(self, frames, title):
        """
        Plot images interactively with text boxes for rearranging frames
        and a checkbox to indicate when plotting is finished.
        """
        sorted_frames = list(frames)  # Convert to a list for easy manipulation
        finished = [False]  # Use a mutable object to track completion state
       
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
                print(f"Invalid indices: Frame {frame_to_move[0]} or Position {end_position[0]} out of range.")

        # Return final sorted frames
        print("Final frame order determined.")
        return np.array(sorted_frames), rearranged_indices

    def plot_correlation_matrix(self, correlation_matrix, title='Correlation Matrix'):
        plt.close('all')
        plt.figure(figsize=(8, 8))
        plt.imshow(correlation_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Frame Index')
        plt.ylabel('Frame Index')
        plt.savefig(os.path.join(self.path, f'{title}.png'))
        plt.close()

    def rearrange_info_and_save(self, name: str = "combined"):
        combined_info_sorted = pd.concat([self.sorted_diastolic_info, self.sorted_systolic_info], axis=0)
        combined_info_original = pd.concat([self.diastolic_info, self.systolic_info], axis=0)
        # Save the rearranged information to a new file
        if self.plot_true:
            print(f"name is {name}")
            combined_info_sorted.to_csv(os.path.join(self.path, f'{name}_sorted_manual.csv'), index=False)
            combined_info_original.to_csv(os.path.join(self.path, f'{name}_original_manual.csv'), index=False)
            # adjust .npy file and .csv file to match the new order.
            print(f"REARRANGE: sorted diastolic indices -> {self.sorted_diastolic_indices}")
            print(f"REARRANGE: sorted systolic indices -> {self.sorted_systolic_indices}")
            print(f"REARRANGE: dia_contour_og -> {self.sorted_diastolic_frames.shape}")
            # Remove last dimension (color) and pad to (512, 512)
            def pad_to_512x512(frames):
                # frames: (N, H, W, 3) or (N, H, W)
                if frames.ndim == 4:
                    frames = frames[..., 0]  # take first channel (grayscale)
                N, H, W = frames.shape
                padded = np.zeros((N, 512, 512), dtype=frames.dtype)
                y_offset = (512 - H) // 2
                x_offset = (512 - W) // 2
                padded[:, y_offset:y_offset+H, x_offset:x_offset+W] = frames
                return padded

            sorted_diastolic_frames = pad_to_512x512(self.sorted_diastolic_frames)
            sorted_systolic_frames = pad_to_512x512(self.sorted_systolic_frames)
            np.save(os.path.join(self.path, f'{name}_diastolic_sorted.npy'), sorted_diastolic_frames)
            np.save(os.path.join(self.path, f'{name}_systolic_sorted.npy'), sorted_systolic_frames)
            
            ordered_items = [self.dia_contours_og[i] for i in self.sorted_diastolic_indices][::-1]
            self.dia_contour_og = {new_idx: item for new_idx, item in enumerate(ordered_items)}
            ordered_items = [self.sys_contours_og[i] for i in self.sorted_systolic_indices][::-1]
            self.sys_contour_og = {new_idx: item for new_idx, item in enumerate(ordered_items)}
            np.savez(os.path.join(self.csv_path, 'diastolic_contours.npz'), **{f'key_{k}': v for k, v in self.dia_contours_og.items()})
            np.savez(os.path.join(self.csv_path, 'systolic_contours.npz'), **{f'key_{k}': v for k, v in self.sys_contours_og.items()})
        else:
            combined_info_sorted.to_csv(os.path.join(self.path, f'{name}_sorted.csv'), index=False)
            combined_info_original.to_csv(os.path.join(self.path, f'{name}_original.csv'), index=False)
            # adjust .npy file and .csv file to match the new order.
            

        print("Combined info saved successfully.")


if __name__ == '__main__':
    # alternative resolution
    res = 0.01755671203136
    # res = 0.02145760878921
    reshuffeling = Reshuffling(
        r"D:\00_coding\pressure_curve_processing\ivus\NARCO_279\stress", resolution=res, plot=True
    )
    diastolic_frames_, systolic_frames_ = reshuffeling()
