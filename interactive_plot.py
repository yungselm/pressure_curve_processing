import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton
import threading

class InteractivePlotCleaning:
    def __init__(self, df):
        self.df = df
        self.time = df['time']
        self.p_aortic = df['p_aortic']
        self.p_distal = df['p_distal']
        self.peaks = df['peaks']
        self.vertical_lines = []  # Store vertical lines
        self.selected_line = None  # Track selected line for movement
        self.last_selected_line = None  # Track the last clicked line
        self.default_line_style = (0, (1, 3))  # Dashed style for lines
        self.line_colors = {1: 'red', 2: 'blue'}  # Colors for each peak type
        self.time_autosave = 15  # Time interval for autosave in seconds
        self.autosave_thread = None
        self.autosave_active = False
        self.autosave_event = threading.Event()  # Event to signal autosave thread to stop

    def __call__(self):
        self.main_plot()
        return self.df

    def main_plot(self):
        # Initialize Matplotlib figure
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Plot data
        self.ax.plot(self.time, self.p_aortic, label='P Aortic')
        self.ax.plot(self.time, self.p_distal, label='P Distal')
        self.ax.scatter(self.time[self.peaks == 1], self.p_aortic[self.peaks == 1], c='r', label='Systole')
        self.ax.scatter(self.time[self.peaks == 2], self.p_aortic[self.peaks == 2], c='b', label='Diastole')

        # Draw initial vertical lines for peaks
        for peak_type in [1, 2]:
            peak_times = self.time[self.peaks == peak_type]
            for t in peak_times:
                line = self.ax.axvline(x=t, color=self.line_colors[peak_type], linestyle=self.default_line_style)
                self.vertical_lines.append((line, peak_type))  # Store line and peak type
        
        diastole_ax = plt.axes([0.6, 0.05, 0.1, 0.075])  # Position for "Diastole" button
        diastole_button = Button(diastole_ax, 'Set Diastole', color='blue', hovercolor='lightblue')
        diastole_button.on_clicked(self.set_diastole)
        
        systole_ax = plt.axes([0.4, 0.05, 0.1, 0.075])  # Position for "Systole" button
        systole_button = Button(systole_ax, 'Set Systole', color='red', hovercolor='lightcoral')
        systole_button.on_clicked(self.set_systole)

        # Add delete button
        delete_ax = plt.axes([0.2, 0.05, 0.1, 0.075])  # Position for "Delete" button
        delete_button = Button(delete_ax, 'Delete Line')
        delete_button.on_clicked(self.remove_line)

        # Connect event handlers to figure
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Start autosave thread
        self.autosave_active = True
        self.autosave_thread = threading.Thread(target=self.autosave)
        self.autosave_thread.start()

        # Display the legend and plot
        self.ax.legend(loc='upper right')
        plt.show(block=True)  # Block execution until the plot window is closed

        # Stop autosave when plot window is closed
        self.autosave_event.set()  # Signal the autosave thread to stop
        if self.autosave_thread is not None:
            self.autosave_thread.join()  # Wait for the autosave thread to finish

        # Update the DataFrame after the plot window is closed
        self.update_df()

    # Event handlers
    def on_click(self, event):
        """Handle mouse click events."""
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
            return
        if event.button is MouseButton.LEFT and event.inaxes == self.ax and event.xdata is not None:
            # Check if clicking near an existing line
            for line, peak_type in self.vertical_lines:
                line_pos = line.get_xdata()  # Get line's position
                if line_pos is not None and abs(line_pos[0] - event.xdata) < 0.05:  # Tolerance for selection
                    self.selected_line = line
                    self.last_selected_line = line
                    print(f"Selected line at {self.selected_line.get_xdata()}")
                    line.set_linestyle('solid')  # Highlight selected line
                    self.fig.canvas.draw()
                    return

            # If no line is selected, create a new one
            new_line = self.ax.axvline(x=event.xdata, color='grey', linestyle=self.default_line_style)
            self.vertical_lines.append((new_line, None))  # New lines don't have a peak type yet
            self.last_selected_line = new_line
            print(f"Created new line at {new_line.get_xdata()}")
            self.fig.canvas.draw()

    def on_motion(self, event):
        """Handle mouse motion events (dragging)."""
        if self.selected_line and event.button is MouseButton.LEFT:
            self.selected_line.set_xdata([event.xdata])  # Update line position
            self.fig.canvas.draw()

    def on_release(self, event):
        """Handle mouse release events."""
        if self.selected_line:
            self.selected_line.set_linestyle(self.default_line_style)  # Reset style
            self.selected_line = None  # Deselect line
            self.fig.canvas.draw()

    def remove_line(self, event):
        """Remove the last selected line."""
        if self.last_selected_line:
            for i, (line, peak_type) in enumerate(self.vertical_lines):
                if line == self.last_selected_line:
                    line.remove()  # Remove line from plot
                    self.vertical_lines.pop(i)  # Remove from storage
                    break
            self.last_selected_line = None  # Reset last selected line
            self.fig.canvas.draw()

    def set_diastole(self, event):
        """Set the last selected line as diastole."""
        if self.last_selected_line:
            print(f"Setting line at {self.last_selected_line.get_xdata()} as diastole")
            self.last_selected_line.set_color('blue')
            for i, (line, peak_type) in enumerate(self.vertical_lines):
                if line == self.last_selected_line:
                    self.vertical_lines[i] = (line, 2)  # Update peak type to 2 (diastole)
                    break
            self.fig.canvas.draw()
        else:
            print("No line selected")

    def set_systole(self, event):
        """Set the last selected line as systole."""
        if self.last_selected_line:
            print(f"Setting line at {self.last_selected_line.get_xdata()} as systole")
            self.last_selected_line.set_color('red')
            for i, (line, peak_type) in enumerate(self.vertical_lines):
                if line == self.last_selected_line:
                    self.vertical_lines[i] = (line, 1)  # Update peak type to 1 (systole)
                    break
            self.fig.canvas.draw()
        else:
            print("No line selected")

    def update_df(self):
        """Update the DataFrame with the new peak positions."""
        # Clear existing peak columns
        self.df['peaks'] = 0  # Reset peaks column to 0
        # Add new peak positions
        for line, peak_type in self.vertical_lines:
            if peak_type is not None and line.get_xdata() is not None:
                t = line.get_xdata()[0]
                closest_idx = (self.df['time'] - t).abs().idxmin()  # Find the closest time index
                self.df.at[closest_idx, 'peaks'] = peak_type  # Update the peak type at the closest time index

    def autosave(self):
        """Autosave the DataFrame every self.time_autosave seconds."""
        while not self.autosave_event.is_set():
            self.update_df()  # Update the DataFrame with the latest peak positions
            self.df.to_csv("C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/NARCO_10_eval/narco_10_pressure_rest_1_autosave.csv", index=False)
            self.autosave_event.wait(self.time_autosave)


if __name__ == "__main__":
    file = "C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/NARCO_10_eval/narco_10_pressure_rest_1.csv"
    df = pd.read_csv(file)
    interactive_plot = InteractivePlotCleaning(df)
    updated_df = interactive_plot()
    updated_df.to_csv(file, index=False)
