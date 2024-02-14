import numpy as np
import matplotlib.pyplot as plt
from measurements import SnV_ODMR
from hamiltonian import SingleSnVHamiltonian as ssnvh
from vectors import vec
from scipy.integrate import simps
from scipy.signal import find_peaks, peak_widths

def calculate_fwhm(data, frequency):
    """
    Calculate the Full Width at Half Maximum (FWHM) of peaks in the given data.

    Parameters:
    data (np.ndarray): The ODMR signal data.
    frequency (np.ndarray): The frequency range corresponding to the data.

    Returns:
    fwhms (list): A list of FWHM values for each detected peak in Hz.
    """

    # Find peaks in the data
    peaks, _ = find_peaks(data, height=0)

    # Calculate the full width at half maximum (FWHM) for each peak
    results_half = peak_widths(data, peaks, rel_height=0.5)

    # Convert the width from data points to frequency units
    fwhms = np.diff(frequency)[0] * results_half[0]

    return fwhms.tolist()


# Constants
MW_freq_range = np.linspace(-1e9, 1e9, 2000)  # Frequency range for ODMR sweep
B0 = 0.001 # Magnetic field strength in Tesla
thetaB, phiB = np.pi/2, 0  # Direction of the magnetic field in spherical coordinates
Linewidth = 10e6  # Linewidth of the transitions (in Hz)
#

Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]

eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates(Bvec)
print( (eigenenergies[1] - eigenenergies[0])/1e9) # 1/(2*ssnvh.gamma_e) *

# Define MW field vectors, perpendicular to B-field for simplicity
# You might use x or y polarization depending on your system setup; here we'll use x-polarization for illustration.
thetaMW, phiMW = np.pi/2, np.pi # Direction of the magnetic field in spherical coordinates

MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]
# Call the function to calculate the ODMR spectrum
snV_ODMR_spectrum = SnV_ODMR.singleSnVodmr(MW_freq_range, MWvec, Bvec, Linewidth)
snV_ODMR_spectrum1 = SnV_ODMR.snv_pulsed(MW_freq_range, MWvec,Bvec, Linewidth)
cw_odmr_area = simps(snV_ODMR_spectrum, MW_freq_range)
pulsed_odmr_area = simps(snV_ODMR_spectrum1, MW_freq_range)

snV_ODMR_spectrum2 = SnV_ODMR.snv_pulsed2(MW_freq_range, MWvec,Bvec, Linewidth)
# print(max(snV_ODMR_spectrum))
# print(max(snV_ODMR_spectrum1))
# Plotting
plt.figure()
plt.rcParams.update({'font.size': 22})

plt.style.use("classic")
plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum/max(snV_ODMR_spectrum), label="cw")
# plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum1/max(snV_ODMR_spectrum1), label="pulsed")
plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum2/max(snV_ODMR_spectrum2), label="pulsed")
plt.xlabel('$\\Delta \\omega$ (MHZ)')
plt.ylabel('Fluoresence (Arbitrary Units)')
plt.legend()
plt.tight_layout()
plt.show()

fwhms = calculate_fwhm(snV_ODMR_spectrum/max(snV_ODMR_spectrum), MW_freq_range)
for idx, fwhm in enumerate(fwhms):
    print(f"Peak snV_ODMR_spectrum {idx+1}: FWHM = {fwhm/1e6} MHz")


fwhms = calculate_fwhm(snV_ODMR_spectrum1/max(snV_ODMR_spectrum1), MW_freq_range)
for idx, fwhm in enumerate(fwhms):
    print(f"Peak snV_ODMR_spectrum1 {idx+1}: FWHM = {fwhm/1e6} MHz")