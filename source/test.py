import numpy as np
from utils import plotting, math_functions
from measurements import cwODMR, pulsedODMR
import matplotlib.pyplot as plt
from vectors import vec
#constants
MW_freq_range = np.linspace(2840, 2900, 500)  # Adjust range and points as needed
B0 = 0 # Magnetic field strength in G
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0 # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = np.pi / 2, 0   # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi/2 , 0  # Direction of MW field
Linewidth = 6  # Linewidth of the transitions (in MHz)
T2_star = 1/(np.pi*Linewidth)
data = []

MWvec_x = [1, 0, 0]

# Linear polarization along y-axis
MWvec_y = [0, 1, 0]

# Right circular polarization (in xy-plane)
MWvec_rcp = [1/np.sqrt(2), 1j/np.sqrt(2), 0]

# Left circular polarization (in xy-plane)
MWvec_lcp = [1/np.sqrt(2), -1j/np.sqrt(2), 0]

MWvec = [MWvec_x, MWvec_y, MWvec_rcp, MWvec_lcp]
Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]
Evec = vec.getAllframesCartesian(E0, thetaE, phiE)[0]
print(Bvec)

# Define the range of free precession times
tau_range = np.linspace(1e-8, 1.5*10e-2, 100)  # Up to one period of the Rabi oscillation

# Get the simulated fluorescence signal as a function of free precession time
# fluorescence_signal = pulsedODMR.ramseySingleNV3(tau_range, MWvec[3],Bvec, Evec, Linewidth)
# plt.style.use("classic")
# #plot dots 
# plt.plot(tau_range, fluorescence_signal/max(fluorescence_signal), 'o', color='blue', markersize=3)
# # plt.plot(tau_range, raw_data)
# plt.xlabel('Free Precession Time ')
# plt.ylabel('Fluorescence Intensity (Arbitrary Units)')
# plt.title('Simulated Ramsey Fringes')
# plt.show()
transition_probabilities = pulsedODMR.ramseySingleNV3(tau_range, MWvec[3],Bvec, Evec, Linewidth)
ms_0_indices = [0]  # Example indices for ms=0 states
fluorescence_signal = transition_probabilities[:, ms_0_indices].sum(axis=1) * np.exp(-tau_range / T2_star)

# Normalize the fluorescence signal if needed
fluorescence_signal_normalized = fluorescence_signal / max(fluorescence_signal)

# Plotting the normalized fluorescence signal
plt.plot(tau_range, fluorescence_signal_normalized, 'o-', color='blue', markersize=3)
plt.xlabel('Free Precession Time (s)')
plt.ylabel('Fluorescence Intensity (Arbitrary Units)')
plt.title('Simulated Ramsey Fringes')
plt.show()


