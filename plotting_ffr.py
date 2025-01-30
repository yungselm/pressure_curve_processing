import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

# Load CSV file
df = pd.read_csv(r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test\NARCO_251_eval\narco_251_pressure_dobu.csv")
path = r"C:\WorkingData\Documents\2_Coding\Python\pressure_curve_processing\test\NARCO_251_eval\narco_251_pressure_ffrdobu.png"
print(df.head())
# Filter data within time range
df = df[(df['time'] > 40) & (df['time'] < 48)]

# Calculate mean values
mean_aortic = int(df['p_mean_aortic'].mean().round(0))
mean_distal = int(df['p_mean_distal'].mean().round(0))
ffr = df['pd/pa'].mean().round(2)

# Print initial rows for verification
print(df.head())

# Plot data
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

plt.plot(df['time'], df['p_distal'], color='#12F508', label='Distal Pressure')
plt.plot(df['time'], df['p_mean_distal'], color='#12F508')
plt.plot(df['time'], df['p_aortic'], color='#FF0000', label='Aortic Pressure')
plt.plot(df['time'], df['p_mean_aortic'], color='#FF0000')

# Add dotted horizontal lines on every y-tick
for y in plt.gca().get_yticks():
    plt.axhline(y=y, color='white', linestyle='--', linewidth=0.5)

# Add text box in the upper-right corner
# Using multiple `plt.text` calls for specific alignment and coloring
plt.text(
    0.80, 0.95,  # X, Y position (upper right corner relative to axes)
    "Pa", 
    fontsize=12, 
    color='#FF0000', 
    ha='left', 
    transform=plt.gca().transAxes
)
plt.text(
    0.98, 0.87,  # Slightly lower for the second line
    f"{mean_aortic}", 
    fontsize=35, 
    fontweight='bold',
    color='#FF0000', 
    ha='right', 
    transform=plt.gca().transAxes
)
plt.text(
    0.80, 0.80,  # Third line for Pd
    "Pd", 
    fontsize=12, 
    color='#12F508', 
    ha='left', 
    transform=plt.gca().transAxes
)
plt.text(
    0.98, 0.72,  # Final line for mean_distal
    f"{mean_distal}", 
    fontsize=35, 
    fontweight='bold',
    color='#12F508', 
    ha='right', 
    transform=plt.gca().transAxes,
)

# Add a single rounded bounding box enclosing all the text
plt.gca().add_patch(FancyBboxPatch(
    (0.80, 0.7),  # Bottom-left corner relative to axes
    0.19, 0.28,  # Width and height of the box
    boxstyle="round,pad=0.01",  # Rounded corners with padding
    transform=plt.gca().transAxes,
    facecolor='black',  # Black fill
    edgecolor='white',  # Black outline
    linewidth=1,
    zorder=2
))

plt.text(
    0.80, 0.60,  # X, Y position (upper right corner relative to axes)
    "FFR",
    fontsize=12,
    color='#FBFB00',
    ha='left',
    transform=plt.gca().transAxes
)

plt.text(
    0.98, 0.50,  # Slightly lower for the second line
    f"{ffr}",
    fontsize=35,
    fontweight='bold',
    color='#FBFB00',
    ha='right',
    transform=plt.gca().transAxes
)

# Add another rounded bounding box below the first one
plt.gca().add_patch(FancyBboxPatch(
    (0.80, 0.45),  # Bottom-left corner relative to axes
    0.19, 0.20,  # Width and height of the box
    boxstyle="round,pad=0.01",  # Rounded corners with padding
    transform=plt.gca().transAxes,
    facecolor='black',  # Black fill
    edgecolor='#FBFB00',  # Yellow outline
    linewidth=1,
    zorder=2
))

# Add titles and labels
plt.title("Pressure Curves", fontsize=16)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Pressure (mmHg)", fontsize=12)
plt.savefig(path)
# Show the plot
plt.show()
