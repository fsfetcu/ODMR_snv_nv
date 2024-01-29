import numpy as np
import matplotlib.pyplot as plt
from measurements import SnV_ODMR
from vectors import vec

# Constants
MW_freq_range = np.linspace(-1e9, 1e9, 1000)  # Frequency range for ODMR sweep
B0 = 0.015 # Magnetic field strength in Tesla
thetaB, phiB = np.pi, np.pi  # Direction of the magnetic field in spherical coordinates
Linewidth = 40e6  # Linewidth of the transitions (in Hz)
Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]

# Define MW field vectors, perpendicular to B-field for simplicity
# You might use x or y polarization depending on your system setup; here we'll use x-polarization for illustration.
thetaMW, phiMW = np.pi/2, np.pi  # Direction of the magnetic field in spherical coordinates

MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]
# Call the function to calculate the ODMR spectrum
snV_ODMR_spectrum = SnV_ODMR.singleSnVodmr(MW_freq_range, MWvec, Bvec, Linewidth)
snV_ODMR_spectrum1 = SnV_ODMR.snv_pulsed(MW_freq_range, MWvec,Bvec, Linewidth)
print(max(snV_ODMR_spectrum))
print(max(snV_ODMR_spectrum1))
# Plotting
plt.figure()
plt.rcParams.update({'font.size': 22})

plt.style.use("classic")
plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum, label="SnV-cw")
plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum1, label="SnV-pulsed")
plt.xlabel('$\\Delta \\omega$ (MHZ)')
plt.ylabel('ODMR Signal (Arbitrary Units)')
plt.legend()
plt.tight_layout()
plt.show()