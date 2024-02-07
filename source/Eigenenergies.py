import numpy as np
import matplotlib.pyplot as plt
from vectors import vec
from scipy.integrate import quad
hbar = 1.0545718e-34  # J*s
k_b = 1.380649e-23  # J/K
c = 299792458  # m/s
omega_0 = 1332.7 * c * 100
def calculate_integral(T):
    def integrand2(omega):
        """
        The integrand of the given integral.
        
        Parameters:
        omega (float): Frequency (rad/s)
        T (float): Temperature (K)
        """
        exponent = np.exp(hbar * omega / (k_b * T))
        return (2 * exponent * omega**2) / ((exponent - 1) * (exponent + 3)**2)
    # integral, error_est = quad(integrand2, 0, Debye, limit=1000)
    # adjusted_linewidth = Linewidth_base * (1 + 2/(np.exp(hbar * omega_0 / (k_b * T)) - 1))  # Example adjustment, replace with your model
    # alpha = 12000* T**3  # Calculate alpha for the current temperature
    # data = SnV_ODMR.singleSnVodmr_T(MW_freq_range, MWvec, Bvec, adjusted_linewidth, -alpha)  # Assuming this function accepts alpha directly

    # # Normalizing data for plotting
    # normalized_data = data / max(data)
    # plt.plot(MW_freq_range, normalized_data + (i) , label=f"T = {T}K with approximation")
    integral, error_est = quad(integrand2, 0, 28.8e11, limit=1000)
    return integral
# Constants
gamma_e = 2 * 9.274e-24 / 6.626e-34 # Gyromagnetic ratio for electron in rad/s/T
lambda_SO = 850e9  # Spin-orbit coupling constant in Hz
B_magnitude = 0.05e27  # Magnetic field strength in Tesla
thetaB, phiB = 0, np.pi/2  # Direction of the magnetic field in spherical coordinates
Bvec = vec.getAllframesCartesian(B_magnitude, thetaB, phiB)[0]
B_z = Bvec[2]  # Z component of the magnetic field
# Function to calculate alpha(T), example linear dependency on T for simplicity


# Temperature range
T_range = np.linspace(0, 400, 400)  # From 1 K to 300 K

# Eigenenergies calculation
E_1 = np.zeros_like(T_range)
E_2 = np.zeros_like(T_range)
E_3 = np.zeros_like(T_range)
E_4 = np.zeros_like(T_range)

for i, T in enumerate(T_range):
    alpha_T = calculate_integral(T)
    print(alpha_T)
    E_1[i] = -B_z - np.sqrt(4*B_magnitude**2*gamma_e**2 + 4*B_z*gamma_e*(lambda_SO + alpha_T) + (lambda_SO + alpha_T)**2) / 2
    E_2[i] = -B_z + np.sqrt(4*B_magnitude**2*gamma_e**2 + 4*B_z*gamma_e*(lambda_SO + alpha_T) + (lambda_SO + alpha_T)**2) / 2
    E_3[i] = B_z - np.sqrt(4*B_magnitude**2*gamma_e**2 - 4*B_z*gamma_e*(lambda_SO + alpha_T) + (lambda_SO + alpha_T)**2) / 2
    E_4[i] = B_z + np.sqrt(4*B_magnitude**2*gamma_e**2 - 4*B_z*gamma_e*(lambda_SO + alpha_T) + (lambda_SO + alpha_T)**2) / 2

    
# Plotting

plt.figure(figsize=(10, 6))
plt.style.use("classic")
plt.plot(T_range, E_1/1e37, label='$E_1$')
plt.plot(T_range, E_2/1e37, label='$E_2$')
plt.plot(T_range, E_3/1e37, label='$E_3$')
plt.plot(T_range, E_4/1e37, label='$E_4$')
plt.xlabel('Temperature (K)')
plt.ylabel('Eigenenergies (arbitrary units of energy$/\\hbar$)')
plt.legend()
plt.grid(True)
plt.show()