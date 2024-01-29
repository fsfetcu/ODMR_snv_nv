import numpy as np
from utils import plotting
from measurements import cwODMR, pulsedODMR
import matplotlib.pyplot as plt

#constants
MW_freq_range = np.linspace(2840, 2900, 500)  # Adjust range and points as needed
B0 = 1000 # Magnetic field strength in G
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 100 # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = np.pi / 2, 0   # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi/2 , 0  # Direction of MW field
Linewidth = 6  # Linewidth of the transitions (in MHz)

# Define the range of free precession times
tau_range = np.linspace(1e-9, 10e-6, 200)  # Up to one period of the Rabi oscillation

# Get the simulated fluorescence signal as a function of free precession time
fluorescence_signal = cwODMR.nvODMR_with_dephasing(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
print(fluorescence_signal)
plt.style.use("classic")
plt.plot(MW_freq_range, fluorescence_signal)
plt.xlabel('Free Precession Time (s)')
plt.ylabel('Fluorescence Intensity (Arbitrary Units)')
plt.title('Simulated Ramsey Fringes')
plt.show()




