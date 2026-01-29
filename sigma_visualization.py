"""
Visualization of EMF distribution over an actual DMFB
"""

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import csv 
import os 
import re 
import matplotlib.patches as patches

"_____________________________________________________________ EMF from raw data _____________________________________________________________"

def extract_average_top100(filename, top_n=100):
    """
    Reads a Tektronix CSV and returns the average of the top N voltage values.
    """
    metadata = {}
    adc = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        data_section = False
        for row in reader:
            if not row:
                continue
            if row[0] == "# Scope metadata":
                continue
            if row[0] == "# Waveform data":
                data_section = True
                continue
            if not data_section:
                metadata[row[0]] = row[1]
            else:
                if row[0] == "PointIndex":
                    continue
                adc.append(float(row[1]))

    adc = np.array(adc)

    # Apply Tektronix scaling
    YMULT = float(metadata["YMULT"])
    YOFF  = float(metadata["YOFF"])
    YZERO = float(metadata["YZERO"])
    voltage = (adc - YOFF) * YMULT + YZERO

    top_values = np.sort(voltage)[-top_n:]  # np.sort ascending, take last top_n
    return np.mean(top_values)
"______________________________________________________________________________________________________________________________________________"



"_____________________________________________________________ collect CSV files _________________________________________________________"


data_dir = "/Users/[]/Desktop/research/Droplet_Tracker_Modularized/official_repo/O_Scope_Data/OpenDroplet/43_actuated"

def extract_index(filename):
    # Extract number after last underscore and before .csv
    match = re.search(r'_(\d+)\.csv$', filename)
    return int(match.group(1)) if match else -1

# Sort files numerically based on index
files = sorted(
    [f for f in os.listdir(data_dir) if f.endswith(".csv")],
    key=extract_index
)

# Extract top 100 average for each CSV
Z_values = []
for f in files:
    filepath = os.path.join(data_dir, f)
    avg_high = extract_average_top100(filepath)
    Z_values.append(avg_high)

Z_values = np.array(Z_values)

"______________________________________________________________________________________________________________________________________________"

X_points = np.array([21.4, 23.6, 25.8, 28.25, 15.5, 19.2, 21.3, 23.6, 26, 28.2, 30.3, 33.3, 21.4, 23.6, 25.8, 28.25, 21.4, 23.6, 25.8, 28.25, 21.4, 23.6, 25.8, 28.25, 15.5, 19.2, 21.3, 23.6, 26, 28.2, 30.3, 33.3, 21.4, 23.6, 25.8, 28.25, 21.4, 23.6, 25.8, 28.25, 21.4, 23.6, 25.8, 28.25, 15.5, 19.2, 21.3, 23.6, 26, 28.2, 30.3, 33.3, 21.4, 23.6, 25.8, 28.25, 21.4, 23.6, 25.8, 28.25, 21.4, 23.6, 25.8, 28.25, 15.5, 19.2, 21.3, 23.6, 26, 28.2, 30.3, 33.3, 21.4, 23.6, 25.8, 28.25, 0, 25, 49, 49, 25, 0])  # x-coordinates for measurements 
Y_points = np.array([40.7, 40.7, 40.7, 40.7, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 36.1, 36.1, 36.1, 36.1, 34, 34, 34, 34, 31.6, 31.6, 31.6, 31.6,29.5,29.5,29.5,29.5,29.5,29.5,29.5,29.5,27.4,27.4,27.4,27.4,25.1,25.1,25.1,25.1,22.9, 22.9, 22.9, 22.9, 20.75, 20.75, 20.75, 20.75, 20.75, 20.75, 20.75, 20.75, 18.4, 18.4, 18.4, 18.4, 16.25, 16.25, 16.25, 16.25, 14, 14, 14, 14, 11.6, 11.6, 11.6, 11.6, 11.6, 11.6, 11.6, 11.6, 9.5, 9.5, 9.5, 9.5, 0, 0, 0, 50, 50, 50 ]) # y-coordinates for measurements 

"_____________________________________________________________ interpolate between pts _________________________________________________________"
from scipy.interpolate import griddata

# Define full grid for plotting
x_grid = np.linspace(0, 49, 50)  # mm
y_grid = np.linspace(0, 50, 50)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Interpolate sparse points over full grid
Z_grid = griddata(
    points=(X_points, Y_points),
    values=Z_values,
    xi=(X_grid, Y_grid),
    method='nearest'  
)

"__________________________________________________________________________________________________________________________________________________"

# load image
DMFB = imread("actual_chip.jpeg")
x0, y0 = 65, 1471
xEnd, yEnd = 1530, 0
DMFB_crop = DMFB[yEnd:y0, x0:xEnd]
Z_grid_mV = Z_grid * 1e3  # convert to mV

plt.figure(figsize=(6, 6), constrained_layout=True)

# Background DMFB image
plt.imshow(
    DMFB_crop,
    extent=[0, 49, 50, 0],
    origin="lower"
)

# Overlay interpolated EMF heatmap
contour = plt.contourf(
    X_grid,
    Y_grid,
    Z_grid_mV,
    levels=40,
    cmap="inferno",
    alpha=0.5
)

center_x, center_y = 26, 25.25
square_size = 2
bl_x = center_x - square_size / 2
bl_y = center_y - square_size / 2

rect = patches.Rectangle(
    (bl_x, bl_y),      
    square_size,       
    square_size,       
    linewidth=2,
    edgecolor='green',
    facecolor='none'   
)

# Axes formatting
plt.xlabel("X Position (mm)", labelpad=15, fontsize=14)
plt.ylabel("Y Position (mm)", labelpad=15, fontsize=14)
plt.xlim(0, 49)
plt.ylim(0, 50)
plt.tick_params(axis='both', which='major', labelsize=12, length=8, width=2)

# Colorbar
cbar = plt.colorbar(contour)
cbar.set_label("EMF (mV)", fontsize=14, rotation=270, labelpad=25)

# Title with bold top line
plt.title(
    r"$\mathbf{EMF\ Distribution\ over\ DMFB}$"
    "\n(Measured by Probe ~3 mm Above Grid Surface)", fontsize=14
)

plt.show()