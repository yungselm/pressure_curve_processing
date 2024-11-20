import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton

# Load the dataset
df = pd.read_csv("C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/NARCO_119_eval/narco_119_pressure_dobu.csv")

# Extract relevant data
time = df['time']
p_aortic = df['p_aortic']
p_distal = df['p_distal']
peaks = df['peaks']

# Initialize Matplotlib figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Plot data
ax.plot(time, p_aortic, label='P Aortic')
ax.plot(time, p_distal, label='P Distal')
ax.scatter(time[peaks == 1], p_aortic[peaks == 1], c='r', label='Peaks == 1')
ax.scatter(time[peaks == 2], p_aortic[peaks == 2], c='b', label='Peaks == 2')

# Variables to store interactive elements
vertical_lines = []  # Store vertical lines
selected_line = None  # Track selected line for movement
default_line_style = (0, (1, 3))  # Dashed style for lines
line_colors = {1: 'red', 2: 'blue'}  # Colors for each peak type

# Draw initial vertical lines for peaks
for peak_type in [1, 2]:
    peak_times = time[peaks == peak_type]
    for t in peak_times:
        line = ax.axvline(x=t, color=line_colors[peak_type], linestyle=default_line_style)
        vertical_lines.append((line, peak_type))  # Store line and peak type

# Event handlers
def on_click(event):
    """Handle mouse click events."""
    global selected_line
    if fig.canvas.cursor().shape() != 0:  # zooming or panning mode
        return
    if event.button is MouseButton.LEFT and event.inaxes == ax and event.xdata is not None:
        # Check if clicking near an existing line
        for line, peak_type in vertical_lines:
            line_pos = line.get_xdata()  # Get line's position
            if line_pos is not None and abs(line_pos[0] - event.xdata) < 0.1:  # Tolerance for selection
                selected_line = line
                line.set_linestyle('solid')  # Highlight selected line
                fig.canvas.draw()
                return

        # If no line is selected, create a new one
        new_line = ax.axvline(x=event.xdata, color='grey', linestyle=default_line_style)
        vertical_lines.append((new_line, None))  # New lines don't have a peak type yet
        fig.canvas.draw()

def on_motion(event):
    """Handle mouse motion events (dragging)."""
    global selected_line
    if selected_line and event.button is MouseButton.LEFT:
        selected_line.set_xdata([event.xdata])  # Update line position
        fig.canvas.draw()


def on_release(event):
    """Handle mouse release events."""
    global selected_line
    if selected_line:
        selected_line.set_linestyle(default_line_style)  # Reset style
        selected_line = None  # Deselect line
        fig.canvas.draw()


def remove_line(event):
    """Remove the currently selected line."""
    global selected_line
    if selected_line:
        for i, (line, peak_type) in enumerate(vertical_lines):
            if line == selected_line:
                line.remove()  # Remove line from plot
                del vertical_lines[i]  # Remove from storage
                break
        selected_line = None  # Reset selection
        fig.canvas.draw()

def set_diastole(event):
    """Set the selected line as diastole."""
    global selected_line
    if selected_line:
        selected_line.set_color('blue')
        for i, (line, peak_type) in enumerate(vertical_lines):
            if line == selected_line:
                vertical_lines[i] = (line, 2)  # Update peak type to 2 (diastole)
                break
        fig.canvas.draw()

def set_systole(event):
    """Set the selected line as systole."""
    global selected_line
    if selected_line:
        selected_line.set_color('red')
        for i, (line, peak_type) in enumerate(vertical_lines):
            if line == selected_line:
                vertical_lines[i] = (line, 1)  # Update peak type to 1 (systole)
                break
        fig.canvas.draw()


# Add buttons for interactivity
# remove_ax = plt.axes([0.8, 0.05, 0.1, 0.075])  # Position for "Remove" button
# remove_button = Button(remove_ax, 'Remove Line')
# remove_button.on_clicked(remove_line)
diastole_ax = plt.axes([0.6, 0.05, 0.1, 0.075])  # Position for "Diastole" button
diastole_button = Button(diastole_ax, 'Set Diastole')
diastole_button.on_clicked(set_diastole)
systole_ax = plt.axes([0.4, 0.05, 0.1, 0.075])  # Position for "Systole" button
systole_button = Button(systole_ax, 'Set Systole')
systole_button.on_clicked(set_systole)

# Connect event handlers to figure
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# Display the legend and plot
ax.legend(loc='upper right')
plt.show()
