import numpy as np
from utils import plotting, math_functions
from measurements import cwODMR, pulsedODMR, SnV_ODMR
import matplotlib.pyplot as plt
from vectors import vec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# Example of generating mock data for the purpose of illustration
# This should be replaced with actual calls to the simulation functions
# that produce the required ODMR data.
thetaB, phiB = np.pi, np.pi  # Direction of the magnetic field in spherical coordinates
E0 = 0 # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = np.pi / 2, 0   # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi/2 , 0  # Direction of MW field
Linewidth = 5e6  # Linewidth of the transitions (in MHz)
# Constants
t_points = 150
tau_range = np.linspace(1e-9, 100e-9, t_points)  # Free precession time (tau) from 0 to 200 ns
B_range = np.linspace(0.12, 0.14, t_points)  # Magnetic field strength B from 0.05 to 0.15 T
MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]
# Initialize an empty array to store the simulated data
probabilities = np.zeros((len(B_range), len(tau_range)))

# Populate the array with mock data for demonstration purposes
for i, B in enumerate(B_range):
    Bvec = vec.getAllframesCartesian(B, thetaB, phiB)[0]
    # Mock probability calculation: sinusoidal variation with decay over tau_range
    t2, result, _ = SnV_ODMR.ram2(tau_range,MWvec,Bvec,Linewidth,t_points)
    probabilities[i, :] = result[1]/max(result[1])
    print(i)
# Normalize the probabilities along the frequency axis
# norm = Normalize(vmin=probabilities.min(), vmax=probabilities.max())
# probabilities = norm(probabilities)

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(probabilities, extent=[tau_range.min()*1e9, tau_range.max()*1e9, B_range.min(), B_range.max()],
           aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Population')
plt.xlabel('Ramsey delay (ns)')
plt.ylabel('Magnitude of B field (T)')
plt.show()
