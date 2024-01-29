import numpy as np
import matplotlib.pyplot as plt
from utils import plotting, math_functions
from measurements import cwODMR, pulsedODMR
from vectors import vec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Constants
MW_freq_range = np.linspace(2870, 2870, 1)  # MW frequency range in MHz
B0 = 0  # Magnetic field strength in G
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0  # Electric field strength in V/m (assuming no electric field for simplicity)
thetaE, phiE = np.pi / 2, 0  # Direction of the electric field (not relevant here)

# Linewidth of the transitions in MHz
Linewidth = 1

# Magnetic and electric field vectors in Cartesian coordinates
Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]
Evec = vec.getAllframesCartesian(E0, thetaE, phiE)[0]

# Prepare arrays to collect ODMR data
thetaMW_range = np.linspace(0, np.pi, 250)  # Adjust the range and number of points as needed
phiMW_range = np.linspace(0, 2*2*np.pi, 250)  # Adjust the range and number of points as needed
ODMR_data = np.zeros((len(thetaMW_range), len(phiMW_range), len(MW_freq_range)))

# Nested loop over thetaMW and phiMW
for i, thetaMW in enumerate(thetaMW_range): 
    for j, phiMW in enumerate(phiMW_range):
        # Calculate MW vector for current angles
        MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]
        
        # Simulate ODMR measurement
        ODMR_data[i, j, :] = cwODMR.ODMRsingleNV(MW_freq_range, MWvec, Bvec, Evec, Linewidth)
        # ODMR_data[i, j, :] = pulsedODMR.pulsedODMRsingleNV(MW_freq_range, MWvec, Bvec, Evec, Linewidth)
        
        ODMR_data[i, j, :] = 1 - (ODMR_data[i, j, :]) 
        print(i, j)
# Now ODMR_data contains the simulated ODMR response for different MW field directions

# Plotting - creating a 2D color plot for each frequency slice
for freq_index in range(len(MW_freq_range)):
    # Reshape data for the current frequency into a 2D array
    data_2D = ODMR_data[:, :, freq_index].reshape(len(thetaMW_range), len(phiMW_range))
    plt.rcParams.update({'font.size': 22})

    plt.figure()
    plt.imshow(data_2D, extent=[phiMW_range.min(), phiMW_range.max(), thetaMW_range.min(), thetaMW_range.max()],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Fluoresence intensity (a.u.)')
    plt.xlabel('$\\phi_{{MW}}$ (radians)')
    plt.ylabel('$\\theta_{{MW}}$ (radians)')
    plt.title(f'ODMR Signal at Frequency resonance frequency (2870 MHz) for single NV center')
    plt.show()