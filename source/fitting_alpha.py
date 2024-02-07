import numpy as np
import matplotlib.pyplot as plt
from utils import plotting, math_functions
from measurements import cwODMR, pulsedODMR
from vectors import vec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import quad
from measurements import SnV_ODMR
from hamiltonian import SingleSnVHamiltonian as ssnvh
from scipy.integrate import simps
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import minimize

MW_freq_range = np.linspace(-1e9, 1e9, 1500)  # Frequency range for ODMR sweep
B0 = 0 # Magnetic field strength in Tesla
thetaB, phiB = np.pi, np.pi  # Direction of the magnetic field in spherical coordinates
Linewidth = 50e6  # Linewidth of the transitions (in Hz)
Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]
#
# eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates_T(Bvec,alpha)
# print(ssnvh.gamma_e * (eigenenergies[1] - eigenenergies[0]))
thetaMW, phiMW = np.pi/2, np.pi # Direction of the magnetic field in spherical coordinates
MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]
# Call the function to calculate the ODMR spectrum
# T = 296


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

def objective_function(inC, MW_freq_range, MWvec, Bvec, Linewidth, Tlist):
    """
    Objective function to minimize. Calculates the difference between
    fwhm_T and Linewidth_ODMR for a given C and a given T.

    Parameters:
    C (float): The parameter to fit.
    Other parameters are used to calculate fwhm_T and Linewidth_ODMR.

    Returns:
    float: The sum of squared differences between fwhm_T and Linewidth_ODMR.
    """
    sum_of_squares = 0
    C,beta = inC
    
    for T in Tlist:
        print(T)
        # Calculate fwhm_T for this value of C and T
        data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, Linewidth, [C, T],beta)
        fwhm_T = calculate_fwhm(data/max(data), MW_freq_range)

        # Calculate Linewidth_ODMR for this T
        snV_ODMR_spectrum_T = SnV_ODMR.singleSnVodmr_L(MW_freq_range, MWvec, Bvec, Linewidth, T)
        fwhms_Linewidth_ODMR = calculate_fwhm(snV_ODMR_spectrum_T/max(snV_ODMR_spectrum_T), MW_freq_range)

        # Assuming both fwhm_T and fwhms_Linewidth_ODMR are lists of FWHM values
        # for each peak, and have the same length
        for fwhm1, fwhm2 in zip(fwhm_T, fwhms_Linewidth_ODMR):
            sum_of_squares += (fwhm1 - fwhm2) ** 2
    return sum_of_squares


initial_C = [1e4,1.259]
Tlist = np.linspace(100,300,20)
# Perform the optimization
result = minimize(objective_function, initial_C, args=(MW_freq_range, MWvec, Bvec, Linewidth, Tlist))
print(result)
C_optimal, beta_optimal = result.x
print(result.x)
data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, Linewidth, [C_optimal, 100],beta_optimal)
fwhm_T = calculate_fwhm(data/max(data), MW_freq_range)

# Calculate Linewidth_ODMR for this T
snV_ODMR_spectrum_T = SnV_ODMR.singleSnVodmr_L(MW_freq_range, MWvec, Bvec, Linewidth, 100)
fwhms_Linewidth_ODMR = calculate_fwhm(snV_ODMR_spectrum_T/max(snV_ODMR_spectrum_T), MW_freq_range)
print("T difference",(fwhms_Linewidth_ODMR[-1] - fwhm_T[-1])/1e6)
print("difference with 0",fwhms_Linewidth_ODMR[-1]/1e6 - Linewidth/1e6)
print()






snV_ODMR_spectrum = SnV_ODMR.singleSnVodmr(MW_freq_range, MWvec, Bvec, Linewidth)
# fwhm_T0 = calculate_fwhm(snV_ODMR_spectrum, MW_freq_range)

# # fitting with different C
T = 100
# data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, Linewidth,[[result.fun],T])
# fwhm_T = calculate_fwhm(data, MW_freq_range)
# snV_ODMR_spectrum_T = SnV_ODMR.singleSnVodmr_L(MW_freq_range, MWvec, Bvec, Linewidth, T)
# fwhms_Linewidth_ODMR = calculate_fwhm(snV_ODMR_spectrum_T, MW_freq_range)

plt.figure()
plt.style.use("classic")
plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum_T, label=f"T = {T}K")
plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum, label="T = 0K")
plt.plot(MW_freq_range / 1e6, data, label="alpha")
plt.xlabel('$\\Delta \\omega$ (MHZ)')
plt.ylabel('Fluoresence (Arbitrary Units)')
plt.legend()
plt.tight_layout()
plt.show()
