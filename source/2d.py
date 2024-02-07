import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
hbar = 1.0545718e-34  # J*s
k_b = 1.380649e-23  # J/K
c = 299792458  # m/s
omega_0 = 1332.7 * c * 100
gamma_e = 2 * 9.274e-24 / 6.626e-34  # Gyromagnetic ratio for electron in rad/s/T
lambda_SO = 850e9  # Spin-orbit coupling constant in Hz
thetaB, phiB = 0, np.pi/2  # Direction of the magnetic field in spherical coordinates

# Calculate integral function
def calculate_integral(T):
    def integrand2(omega):
        exponent = np.exp(hbar * omega / (k_b * T))
        return (2 * exponent * omega**2) / ((exponent - 1) * (exponent + 3)**2)
    integral, error_est = quad(integrand2, 0, 28.8e11, limit=1000)
    return integral

# B_magnitude range
B_magnitude_range = np.linspace(0.01e27, 0.15e27, 50)

# Temperature range
T_range = np.linspace(1, 400, 50)  # Adjusted for computational efficiency

# Initialize 2D arrays for E_1 energies
E_1_matrix = np.zeros((len(T_range), len(B_magnitude_range)))

# Calculate energies for each B_magnitude and temperature
for j, B_magnitude in enumerate(B_magnitude_range):
    Bvec = np.array([np.sin(thetaB) * np.cos(phiB), np.sin(thetaB) * np.sin(phiB), np.cos(thetaB)]) * B_magnitude
    B_z = Bvec[2]
    for i, T in enumerate(T_range):
        alpha_T = calculate_integral(T)
        E_1_matrix[i, j] = -B_z + np.sqrt(4*B_magnitude**2*gamma_e**2 + 4*B_z*gamma_e*(lambda_SO + alpha_T) + (lambda_SO + alpha_T)**2) / 2
# Plotting
plt.figure(figsize=(10, 6))
plt.contourf(B_magnitude_range/1e27, T_range, E_1_matrix/1e37, 50, cmap='viridis')
plt.colorbar(label='Eigenenergy $E_1$ (arbitrary units)')
plt.xlabel('Magnetic Field Magnitude ($10^{27}$ Tesla)')
plt.ylabel('Temperature (K)')
plt.title('Eigenenergy $E_1$ as a function of Temperature and Magnetic Field Magnitude')
plt.show()