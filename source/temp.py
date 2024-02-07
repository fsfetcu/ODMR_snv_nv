import numpy as np
import matplotlib.pyplot as plt
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

# Assuming SnV_ODMR.singleSnVodmr_T and other necessary functions are defined correctly in your environment

# Constants
MW_freq_range = np.linspace(-0.5e8, 1e8, 1500)  # Frequency range for ODMR sweep
B0 = 0.0001  # Magnetic field strength in Tesla
thetaB, phiB = np.pi, np.pi  # Direction of the magnetic field in spherical coordinates
Linewidth_base = 10e6  # Base Linewidth of the transitions (in Hz), may be adjusted by temperature
Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]
thetaMW, phiMW = np.pi / 2, np.pi  # Direction of the magnetic field in spherical coordinates
MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]
Debye = 28.8e11 # Hz
hbar = 1.0545718e-34  # J*s
k_b = 1.380649e-23  # J/K
c = 299792458  # m/s
omega_0 = 1332.7 * c * 100#32.1 * 1.60218e-19 / hbar * 1e-3  # Convert from meV to Hz
def X(T):
    return np.exp(hbar*Debye/(k_b*T))

# Temperature range for the plots
temperatures = [50, 200, 296,400]  # Example temperatures in Kelvin


plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 12})
plt.style.use("classic")


for i,T in enumerate(temperatures):
    def integrand(x):
        return 0.1*(2 * x**2 * (np.log(x))**2) / ((x - 1) * (x + 3)**2) # (k_b/hbar)**3*
    def integrand2(omega):
        """
        The integrand of the given integral.
        
        Parameters:
        omega (float): Frequency (rad/s)
        T (float): Temperature (K)
        """
        exponent = np.exp(hbar * omega / (k_b * T))
        return 0.5e-29*(2 * exponent * omega**2) / ((exponent - 1) * (exponent + 3)**2)
    # integral, error_est = quad(integrand2, 0, Debye, limit=1000)
    # adjusted_linewidth = Linewidth_base * (1 + 2/(np.exp(hbar * omega_0 / (k_b * T)) - 1))  # Example adjustment, replace with your model
    # alpha = 12000* T**3  # Calculate alpha for the current temperature
    # data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, adjusted_linewidth, -alpha)  # Assuming this function accepts alpha directly

    # # Normalizing data for plotting
    # normalized_data = data / max(data)
    # plt.plot(MW_freq_range, normalized_data + (i) , label=f"T = {T}K with approximation")
    integral, error_est = quad(integrand2, 0, Debye, limit=1000)
    adjusted_linewidth = Linewidth_base * (1 + 2/(np.exp(hbar * omega_0 / (k_b * T)) - 1))  # Example adjustment, replace with your model
    alpha = integral  # Calculate alpha for the current temperature
    data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, adjusted_linewidth, -alpha)  # Assuming this function accepts alpha directly
    # Normalizing data for plotting
    normalized_data = data / max(data)
    plt.plot(MW_freq_range, normalized_data + (i+1) , label=f"T = {T}K")
SnV_ODMR_spectrum = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, Linewidth_base, 0)
plt.plot(MW_freq_range,SnV_ODMR_spectrum/max(SnV_ODMR_spectrum),label="T=0 K")

plt.xlabel('Frequency shift compared to $T=0$ K (arbitrary units of frequency)')
plt.ylabel('Fluorescence (arbitrary units)')
plt.legend()
plt.tight_layout()
plt.show()