import csv
import numpy as np
import matplotlib.pyplot as plt

filename = "/Users/[]/Desktop/research/Droplet_Tracker_Modularized/official_repo/O_Scope_Data/TopPlateDroplet/Region 1/tds7104_TopPlateDroplet_region1_1.csv"

metadata = {} 
adc = []

with open(filename, "r") as f:
    reader = csv.reader(f)

    data_section = False
    for row in reader:
        if not row:
            continue

        # Read metadata
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

# Extract scaling parameters
YMULT = float(metadata["YMULT"])
YOFF  = float(metadata["YOFF"])
YZERO = float(metadata["YZERO"])

XINCR = float(metadata["XINCR"])
XZERO = float(metadata["XZERO"])

# Apply Tektronix scaling
voltage = (adc - YOFF) * YMULT + YZERO
time = np.arange(len(voltage)) * XINCR + XZERO

# Plot
plt.figure(figsize=(10, 4))
plt.plot(time * 1e6, voltage)  # microseconds 
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (V)")
plt.title("TDS7104 CH1 Waveform (Scaled)")
plt.grid(True)
plt.tight_layout()
plt.show()
