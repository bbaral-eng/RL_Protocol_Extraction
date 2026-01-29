"""
Triggered data logging for Tektronix TDS7104
Single channel (CH1), ASCII waveform, CSV storage
"""
import pyvisa
import time
import numpy as np
import csv
import os

# connect to scope 
rm = pyvisa.ResourceManager('@py')
print(rm.list_resources())

scope = rm.open_resource()
print(scope.query("*IDN?"))

# Scope configuration
scope.write("*CLS")                    # Clear status
scope.write("DATA:SOURCE CH1")         # Channel 1 only
scope.write("DATA:ENCdg ASCii")        # ASCII waveform
scope.write("HDR ON")                  # Include headers
scope.write("VERBose ON")              # Verbose responses
scope.write("WFMOUTPRE:BYT_NR 1")      # 1 byte per sample
scope.write("DATA:START 1")            # First point
scope.write("DATA:STOP 10000")         # Number of points
scope.write("ACQ:STOPA SEQ")           # Single acquisition
scope.write("TRIG:A:MODE NORM")        # Normal trigger


def parse_preamble(preamble):
    """
    Convert Tektronix preamble into a dictionary of key-value pairs.
    Leaves values as strings for full fidelity.
    """
    preamble = preamble.replace(":WFMOUTPRE:", "").strip()
    parsed = {}
    for field in preamble.split(";"):
        field = field.strip()
        if not field:
            continue
        parts = field.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts
        parsed[key] = value.strip().strip('"')
    return parsed


nums = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84]

output_dir = "/Users/[]/Desktop/research/Droplet_Tracker_Modularized/official_repo/O_Scope_Data/TopPlateDroplet/Region 10" # specify output directory here


for num in nums:
    input(f"\nPress ENTER to capture and save probe {num} (Ctrl+C to quit)")

    # Triggered acquisition
    print("Arming trigger...")
    scope.write("ACQ:STATE ON")

    # Wait until trigger fires
    while scope.query("ACQ:STATE?").strip() == '1':
        time.sleep(0.01)

    print("Triggered. Reading waveform...")

    # Get scaling info (unchanged behavior)
    raw_preamble = scope.query("WFMOUTPRE?")
    preamble = parse_preamble(raw_preamble)
    print("Preamble keys:", preamble.keys())

    scope.write("CURVE?")
    raw = scope.read_raw()
    data_str = raw.decode('ascii').strip()  # convert to string

    # Remove the leading ':CURVE ...' text
    if data_str.startswith(":CURVE"):
        data_str = data_str.split(' ', 1)[1]  # keep only the numbers

    # Convert to floats
    waveform = np.array([float(x) for x in data_str.split(',')])

    filename = os.path.join(
        output_dir,
        f"tds7104_TopPlateDroplet_region10_{num}.csv"
    )

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write preamble metadata at the top
        writer.writerow(["# Scope metadata"])
        for k, v in preamble.items():
            writer.writerow([k, v])

        writer.writerow([])  # empty line
        writer.writerow(["# Waveform data"])
        writer.writerow(["PointIndex", "Voltage"])  
        for i, v in enumerate(waveform, start=1):
            writer.writerow([i, v])

    print(f"Data saved to {filename}")
