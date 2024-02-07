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

def objective_function(C, MW_freq_range, MWvec, Bvec, Linewidth, Tlist):
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

    for T in Tlist:
        # Calculate fwhm_T for this value of C and T
        data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, Linewidth, [C, T])
        fwhm_T = calculate_fwhm(data, MW_freq_range)

        # Calculate Linewidth_ODMR for this T
        snV_ODMR_spectrum_T = SnV_ODMR.singleSnVodmr_L(MW_freq_range, MWvec, Bvec, Linewidth, T)
        fwhms_Linewidth_ODMR = calculate_fwhm(snV_ODMR_spectrum_T, MW_freq_range)

        # Assuming both fwhm_T and fwhms_Linewidth_ODMR are lists of FWHM values
        # for each peak, and have the same length
        for fwhm1, fwhm2 in zip(fwhm_T, fwhms_Linewidth_ODMR):
            sum_of_squares += (fwhm1 - fwhm2) ** 2

    return sum_of_squares

# Constants
MW_freq_range = np.linspace(-1e9, 1e9, 1500)  # Frequency range for ODMR sweep
B0 = 0.000 # Magnetic field strength in Tesla
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


# plt.figure()
# plt.style.use("classic")
# plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum/max(snV_ODMR_spectrum), label="cw")
# plt.xlabel('$\\Delta \\omega$ (MHZ)')
# plt.ylabel('Fluoresence (Arbitrary Units)')
# plt.legend()
# plt.tight_layout()
# plt.show()


Debye = 28.8e13  # Hz
hbar = 1.0545718e-34  # J*s
k_b = 1.380649e-23  # J/K
c = 299792458  # m/s
omega_0 = 1332.7 * c * 100#32.1 * 1.60218e-19 / hbar * 1e-3  # Convert from meV to Hz
def X(T):
    return np.exp(hbar*Debye/(k_b*T))
def integrand(x):
    return 0.05*(2 * x**2 * (np.log(x))**2) / ((x - 1) * (x + 3)**2) # (k_b/hbar)**3*
Tlist = np.linspace(300, 300, 1)

# def gamma(T):
#     return Gamma_0 *(1+2/(np.exp(hbar*omega_0/(k_b*T))-1))

result = np.zeros(len(Tlist))
error = np.zeros(len(Tlist))

# for i,T1 in enumerate(Tlist):
#     integral, error_est = quad(integrand, 0, X(T1), limit = 1000)
#     result[i] = integral
#     error[i] = error_est
# print(result)



snV_ODMR_spectrum = SnV_ODMR.singleSnVodmr(MW_freq_range, MWvec, Bvec, Linewidth)
fwhm_T0 = calculate_fwhm(snV_ODMR_spectrum, MW_freq_range)
T=100
integral, error_est = quad(integrand, 0, X(T), limit = 1000)
print(integral)
# fitting with different C
def gamma(T):
            return Linewidth *(1+2/(np.exp(hbar*omega_0/(k_b*T))-1))
print(gamma(T)/1e6)
data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, gamma(T),(integral *T**3))
snV_ODMR_spectrum_T = SnV_ODMR.singleSnVodmr_L(MW_freq_range, MWvec, Bvec, Linewidth, T)
fwhm_T = calculate_fwhm(data, MW_freq_range)

plt.figure()
plt.rcParams.update({'font.size': 22})
diff = np.abs(max(snV_ODMR_spectrum)-max(data))
plt.style.use("classic")
# plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum_T, label=f"T = {T}K")
plt.plot(MW_freq_range/1e6 , snV_ODMR_spectrum/max(snV_ODMR_spectrum), label="T = 0K")
plt.plot(MW_freq_range/1e6, data/max(data), label=f"T = {T}K")
# plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum, label="T = 0K")
# plt.plot(MW_freq_range / 1e6, data, label=f"T = {T}K")
plt.xlabel('$\\Delta \\omega$ (MHZ)')
plt.ylabel('Fluoresence (Arbitrary Units)')
plt.legend()
plt.tight_layout()
plt.show()



# MW_freq_range = np.linspace(-1e9, 1e9, 1000)  # Frequency range for ODMR sweep
# B0 = 0 # Magnetic field strength in Tesla
# thetaB, phiB = np.pi, np.pi  # Direction of the magnetic field in spherical coordinates
# Linewidth = 50e6  # Linewidth of the transitions (in Hz)
# Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]
# #
# # eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates_T(Bvec,alpha)
# # print(ssnvh.gamma_e * (eigenenergies[1] - eigenenergies[0]))
# thetaMW, phiMW = np.pi/2, np.pi # Direction of the magnetic field in spherical coordinates
# MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]
# # Call the function to calculate the ODMR spectrum
# # T = 296
# Linewidth_ODMR = np.zeros(len(Tlist))
# phonon_ODMR = np.zeros(len(Tlist))
# for i,T in enumerate(Tlist):
#     snV_ODMR_spectrum_T = SnV_ODMR.singleSnVodmr_L(MW_freq_range, MWvec, Bvec, Linewidth,T)

#     fwhms = calculate_fwhm(snV_ODMR_spectrum_T, MW_freq_range)
#     for idx, fwhm in enumerate(fwhms):
#         print(f"Peak snV_ODMR_spectrum {idx+1}: FWHM = {fwhm/1e6} MHz")
#         Linewidth_ODMR[i] = fwhm
#     # fwhms = calculate_fwhm(snV_ODMR_spectrum_T, MW_freq_range)
#     # for idx, fwhm in enumerate(fwhms):
#     #     print(f"Peak snV_ODMR_spectrum {idx+1}: FWHM = {fwhm/1e6} MHz")
#     #     phonon_ODMR[i] = fwhm
# # FWHM at T = 0:
# snV_ODMR_spectrum = SnV_ODMR.singleSnVodmr(MW_freq_range, MWvec, Bvec, Linewidth)
# fwhm_T0 = calculate_fwhm(snV_ODMR_spectrum, MW_freq_range)
# T=200
# # fitting with different C
# C = [-1e12]
# data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, Linewidth,[C,T])

# fwhm_T = calculate_fwhm(data, MW_freq_range)

# plt.figure()
# plt.rcParams.update({'font.size': 22})

# plt.style.use("classic")
# plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum_T, label=f"T = {T}K")
# plt.plot(MW_freq_range / 1e6, snV_ODMR_spectrum, label="T = 0K")
# plt.plot(MW_freq_range / 1e6, data, label="alpha")
# plt.xlabel('$\\Delta \\omega$ (MHZ)')
# plt.ylabel('Fluoresence (Arbitrary Units)')
# plt.legend()
# plt.tight_layout()
# plt.show()

# print(Linewidth_ODMR)


# initial_C = 1

# # Perform the optimization
# result = minimize(objective_function, initial_C, args=(MW_freq_range, MWvec, Bvec, Linewidth, Tlist))