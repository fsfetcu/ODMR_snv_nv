import numpy as np
import matplotlib.pyplot as plt
from utils import plotting, math_functions
from measurements import cwODMR, pulsedODMR
from vectors import vec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Constants
MW_freq_range = np.linspace(2800, 2940, 500)  # Adjust range and points as needed
B0 = 10  # Magnetic field strength in G
# thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0  # Electric field strength in V/m (assuming no electric field for simplicity)
thetaE, phiE = np.pi / 2, 0  # Direction of the electric field (not relevant here)

# Linewidth of the transitions in MHz
Linewidth = 1

# Magnetic and electric field vectors in Cartesian coordinates
MWvec = vec.getAllframesCartesian(1, 0, np.pi/2)[0]
Evec = vec.getAllframesCartesian(E0, thetaE, phiE)[0]

# Prepare arrays to collect ODMR data
thetaB_range = np.linspace(0, np.pi, 20)  # Adjust the range and number of points as needed
phiB_range = np.linspace(0, 2*2*np.pi, 20)  # Adjust the range and number of points as needed
ODMR_data = np.zeros((len(thetaB_range), len(phiB_range), len(MW_freq_range)))

# Create an array to store the frequency differences
freq_diffs = np.zeros((len(thetaB_range), len(phiB_range)))

for i, thetaB in enumerate(thetaB_range):
    for j, phiB in enumerate(phiB_range):
        # Invert the ODMR signal to find dips as peaks
        Bvec = vec.getAllframesCartesian(10, thetaB, phiB)[0]

        ODMR_data[i, j, :] = cwODMR.ODMRsingleNV(MW_freq_range, MWvec, Bvec, Evec, Linewidth)
        inverted_signal = -ODMR_data[i, j, :]

        # Find peaks (which correspond to dips in the original signal)
        peaks, properties = find_peaks(inverted_signal, prominence=0.01)  # Adjust prominence as needed
        
        # Check if we have at least two peaks
        if len(peaks) >= 2:
            # Sort peaks by prominence and select the two most prominent
            prominences = properties['prominences']
            sorted_peaks = peaks[np.argsort(-prominences)][:2]
            
            # Calculate the frequency difference
            freq_diff = abs(MW_freq_range[sorted_peaks[0]] - MW_freq_range[sorted_peaks[1]])
            freq_diffs[i, j] = freq_diff
        else:
            # Handle cases with fewer than 2 peaks
            freq_diffs[i, j] = np.nan  # Or some other default value
        print(i, j)
# Plotting the frequency differences
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(8, 6))
plt.imshow(freq_diffs, extent=[phiB_range.min(), phiB_range.max(), thetaB_range.min(), thetaB_range.max()],
           aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Frequency difference $\\frac{\\Delta E}{h}$ (MHz)')
plt.xlabel('$\\phi_{B}$ (radians)')
plt.ylabel('$\\theta_{B}$ (radians)')
plt.show()