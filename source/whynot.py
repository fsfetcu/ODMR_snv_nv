import numpy as np
from hamiltonian import singleNVhamiltonian as snvh
from hamiltonian import SingleSnVHamiltonian as ssnvh
from vectors import vec as vec
from utils import noise, math_functions , operators
from qutip import Qobj, tensor, qeye, mesolve, basis, expect, ket2dm,sigmax, sigmay, sigmaz
from qutip import Options
import scipy.linalg as la
from qutip.qip.operations import snot
from qutip.qip.device import Processor

from alive_progress import alive_it
from alive_progress import alive_bar
import matplotlib.pyplot as plt
# T2_star = 1 / (np.pi * Linewidth)  # T2 is in microseconds
# nMW = len(tau)  # Number of microwave frequency points
# Tstrength = np.zeros(nMW)  # Transition strength

# Eigenenergies and eigenvectors
#simple hamiltonian
# Hint = snvh.simpleMWint(MWvec)
B0 = 0# Magnetic field strength in G
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0 # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = np.pi / 2, 0   # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi/2 , 0  # Direction of MW field
Linewidth = 6  # Linewidth of the transitions (in MHz)

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
H_free = snvh.simpleFree(Bvec, Evec)
eigenenergies, eigenstates = snvh.simpleHamiltonEigen(Bvec, Evec)
resonant = 2870
# Constants
omega_0 = 2870  # Resonant frequency of the qubit in Hz
tau_max = 0.1  # Maximum free precession time in seconds
points = 500  # Number of points in the time range
phi = 0  # Phase of the second π/2 pulse

# Define the qubit states |0> and |1>
state_0 = basis(2, 0)
state_1 = basis(2, 1)
H = omega_0 * state_1 * state_1.dag()
print(H)
# Define the π/2 pulse operator (Pauli-X rotation)
U_pi2 = (-1j*sigmax() * (np.pi / 4)).expm()

# Define the free evolution unitary operator for time tau
# The Hamiltonian is omega_0 * sigmaz() / 2, so the unitary operator is as follows:
def U_free_evolution(tau_max):
    return (-1j * H * tau_max).expm()


# Define the range of free precession times
tau_range = np.linspace(0, tau_max, points)

# Prepare the storage for probabilities of being in |0>
probabilities = []

# Ramsey sequence simulation
for tau in tau_range:
    # Apply the first π/2 pulse
    psi_after_first_pulse = U_pi2 * state_0
    print(psi_after_first_pulse)
    # Free evolution for time tau
    print(U_free_evolution(tau))
    psi_after_evolution = U_free_evolution(tau) * psi_after_first_pulse

    # Apply the second π/2 pulse with phase phi
    U_pi2_phi = U_pi2 * np.exp(1j * phi)
    psi_final = U_pi2 * psi_after_evolution

    # Probability of being in |0>
    prob = abs(psi_final.overlap(state_0))**2
    probabilities.append(prob)

# Plotting the Ramsey fringes
plt.plot(tau_range , probabilities, label='Ramsey Fringes')
plt.xlabel('Free Precession Time (µs)')
plt.ylabel('Probability of Being in |0>')
plt.title('Simulated Ramsey Fringes')
plt.legend()
plt.grid(True)
plt.show()