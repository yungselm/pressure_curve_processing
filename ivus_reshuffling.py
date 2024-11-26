import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from ipywidgets import interact, VBox, HBox, Dropdown, Button
from IPython.display import display
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class Reshuffeling:
    def __init__(self, path):
        self.path = path
        self.diastolic_frames, self.systolic_frames = self.load_frames(path)
        self.diastolic_info, self.systolic_info = self.read_info(path)

    def __call__(self):
        self.diastolic_frames = self.crop_images(self.diastolic_frames, 50, 512, 25, 487)
        self.systolic_frames = self.crop_images(self.systolic_frames, 50, 512, 25, 487)
        self.sorted_diastolic_frames, self.sorted_diastolic_info, self.correlation_matrix, self.rotation_matrix = (
            self.reshuffle(self.diastolic_frames, self.diastolic_info)
        )
        self.plot_comparison(self.diastolic_info, self.sorted_diastolic_info)
        self.plot_images()
        self.plot_angles(self.diastolic_frames, self.rotation_matrix)

        return self.sorted_diastolic_frames, self.systolic_frames

    def load_frames(self, path):
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

    def read_info(self, path):
        """Read IVUS information from a text file."""
        info = None
        for filename in os.listdir(path):
            if '_report' in filename:
                info = pd.read_csv(os.path.join(path, filename), sep='\t')
                break  # Exit the loop once the report file is found

        if info is None:
            raise FileNotFoundError("No report file found in the specified directory.")

        info_dia = info[info['phase'] == 'D']
        info_sys = info[info['phase'] == 'S']

        return info_dia, info_sys

    def crop_images(self, frames, x1, x2, y1, y2):
        """Crop all images in a set of frames to the same size."""
        cropped_frames = np.zeros((frames.shape[0], x2 - x1, y2 - y1))
        for idx, frame in enumerate(frames):
            cropped_frames[idx] = frame[x1:x2, y1:y2]
        return cropped_frames

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

    def greedy_path(self, corr_matrix):
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

    def reshuffle(self, diastolic_frames, diastolic_info):
        correlation_matrix, rotation_matrix = self.compute_correlation_matrix_with_rotation(diastolic_frames)
        sorted_indices = self.greedy_path(correlation_matrix)
        sorted_diastolic_frames = diastolic_frames[sorted_indices]
        sorted_diastolic_info = diastolic_info.iloc[sorted_indices].reset_index(drop=True)
        sorted_diastolic_frames = sorted_diastolic_frames[::-1]
        sorted_diastolic_info = sorted_diastolic_info[::-1]

        return sorted_diastolic_frames, sorted_diastolic_info, correlation_matrix, rotation_matrix

    def plot_comparison(self, diastolic_info, sorted_diastolic_info):
        plt.figure(figsize=(10, 5))
        plt.plot(diastolic_info['lumen_area'].values, label='Original')
        plt.plot(sorted_diastolic_info['lumen_area'].values, label='Sorted')
        plt.legend()
        plt.xlabel('Frame Index')
        plt.ylabel('Lumen Area')
        plt.title('Comparison of Lumen Area Before and After Sorting')
        plt.show()

    def plot_images(self):
        frames = self.sorted_diastolic_frames
        num_frames = len(frames)
        cols = 8
        rows = (num_frames + cols - 1) // cols

        fig, ax = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
        for idx, frame in enumerate(frames):
            ax[idx // cols, idx % cols].imshow(frame, cmap='gray')
            ax[idx // cols, idx % cols].set_title(f'Frame {idx}')
            ax[idx // cols, idx % cols].axis('off')

        for i in range(num_frames, rows * cols):
            ax[i // cols, i % cols].axis('off')

        plt.show()

    def update_order(self, **kwargs):
        # Update the order based on user selection
        self.order = [kwargs[f'frame_{i}'] for i in range(self.num_frames)]
        self.plot_images()

    def interactive_plot(self):
        # Create dropdown widgets for each frame
        widgets = []
        for i in range(self.num_frames):
            widgets.append(Dropdown(options=list(range(self.num_frames)), 
                                    value=self.order[i], 
                                    description=f'Frame {i}:',
                                    continuous_update=False))
        
        # Create an interactive UI
        ui = VBox(widgets)
        interact_ui = interact(self.update_order, **{f'frame_{i}': w for i, w in enumerate(widgets)})
        display(ui, interact_ui)

    def plot_angles(self, diastolic_frames, rotation_matrix):
        last_frame_index = len(diastolic_frames) - 1
        best_angles = [rotation_matrix[last_frame_index, i] for i in range(len(diastolic_frames))]
        plt.plot(best_angles, label='Best Rotation Angles')
        plt.xlabel('Frame Index')
        plt.ylabel('Angle (degrees)')
        plt.title('Best Rotation Angles for Last Frame')
        plt.legend()
        plt.show()


reshuffeling = Reshuffeling(r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test_files\Stress")
diastolic_frames, systolic_frames = reshuffeling()
reshuffeling.plot_images()

# class DraggablePixmapItem(QGraphicsPixmapItem):
#     def __init__(self, pixmap, index, parent=None):
#         super().__init__(pixmap, parent)
#         self.index = index  # Store the index of the image
#         self.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
#         self.setFlag(QGraphicsPixmapItem.ItemSendsGeometryChanges, True)

#     def mouseReleaseEvent(self, event):
#         super().mouseReleaseEvent(event)
#         # Emit a custom signal or handle the drop logic here


# class ImageReorderingApp(QMainWindow):
#     def __init__(self, frames):
#         super().__init__()
#         self.frames = frames
#         self.order = list(range(len(frames)))  # Track current order of images

#         self.initUI()

#     def initUI(self):
#         self.setWindowTitle("Drag-and-Drop Image Reordering")
#         self.setGeometry(100, 100, 800, 600)

#         # Main widget and layout
#         central_widget = QWidget()
#         layout = QVBoxLayout(central_widget)
#         self.setCentralWidget(central_widget)

#         # Graphics view and scene for displaying images
#         self.view = QGraphicsView()
#         self.scene = QGraphicsScene()
#         self.view.setScene(self.scene)
#         layout.addWidget(self.view)

#         self.load_images()

#     def load_images(self):
#         """Load images into the QGraphicsScene."""
#         cols = 4  # Number of columns in the grid
#         spacing = 10  # Spacing between images
#         thumbnail_size = 100  # Size of each thumbnail

#         for idx, frame_idx in enumerate(self.order):
#             # Convert NumPy array to QPixmap
#             frame = self.frames[frame_idx]
#             height, width = frame.shape
#             bytes_per_line = width
#             q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
#             pixmap = QPixmap.fromImage(q_image).scaled(thumbnail_size, thumbnail_size, Qt.KeepAspectRatio)

#             # Create a draggable pixmap item
#             item = DraggablePixmapItem(pixmap, frame_idx)
#             row, col = divmod(idx, cols)
#             item.setPos(col * (thumbnail_size + spacing), row * (thumbnail_size + spacing))
#             self.scene.addItem(item)

#     def keyPressEvent(self, event):
#         """Re-render the grid in the new order when Enter is pressed."""
#         if event.key() == Qt.Key_Return:  # Rearrange images on Enter key
#             self.refresh_order()

#     def refresh_order(self):
#         """Re-render the images based on the current order."""
#         self.scene.clear()
#         self.load_images()


# def main():
#     app = QApplication(sys.argv)

#     # Use diastolic_frames from reshuffeling
#     frames = diastolic_frames

#     # Launch the app
#     main_window = ImageReorderingApp(frames)
#     main_window.show()
#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()

