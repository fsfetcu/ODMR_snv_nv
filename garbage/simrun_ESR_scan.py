import numpy as np
import matplotlib.pyplot as plt
import simulate_TStrength_NV_ensemble as simnv
# Define MW frequency range (in MHz) for the ODMR spectrum
MW_freq_range = np.linspace(2800, 2950, 1000)  # Adjust range and points as needed

# Define the parameters for the NV center environment
# Example values: Magnetic field (B0), electric field (E0), etc.
B0 = 0.01  # Magnetic field strength in Tesla
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0  # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = 0, 0  # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi / 2, 0  # Direction of MW field
Linewidth = 10  # Linewidth of the transitions (in MHz)

# Calculate the ODMR spectrum
ODMR_spectrum = simnv.ESR_NVensemble(MW_freq_range, thetaMW, phiMW, 
                                   B0, thetaB, phiB, 
                                   E0, thetaE, phiE, 
                                   Linewidth)
data = -ODMR_spectrum/np.max(ODMR_spectrum) + 1


# Plot the ODMR spectrum
plt.figure(figsize=(10, 6))
plt.plot(MW_freq_range, data, label='Photoluminescence Intensity')
plt.xlabel('Microwave Frequency (MHz)')
plt.ylabel('Fluorescence Intensity (Arbitrary Units)')
plt.title('Simulated ODMR Spectrum of NV Center')
plt.legend()
plt.grid(True)
plt.show()
